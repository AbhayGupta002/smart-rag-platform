package com.nextgem.smartrag.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nextgem.smartrag.query.*;
import com.nextgem.smartrag.vectorstore.ChromaVectorStoreService;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;

/**
 * Enterprise Bounded RAG Generation & Synthesis Service.
 * Engineered for ~1,000 Concurrent Users:
 * - Multi-tier L1/L2 Redis caching
 * - SingleFlight in-flight query deduplication
 * - Fair FIFO Semaphore backpressure & queue timeout
 * - Multi-Tier Failover Synthesis (Cloud RestClient -> Local Ollama RestClient -> Deterministic Extractive)
 * - Circuit breaker with graceful fast-fallback
 * - Query relevance filtering (eliminates boilerplate preambles & code noise)
 * - Server-Sent Events (SSE) token streaming to frontend
 */
@Service
public class RagGenerationService {

    private static final Logger log = LoggerFactory.getLogger(RagGenerationService.class);

    private final ChromaVectorStoreService vectorStoreService;
    private final RagCacheService cacheService;
    private final QueryRelevanceFilter relevanceFilter;
    private final QueryConcurrencyGuard concurrencyGuard;
    private final QueryCircuitBreaker circuitBreaker;
    private final QueryMetricsTracker metricsTracker;
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;
    private final RestClient restClient;
    private final ExecutorService streamingExecutor;

    @Value("${rag.llm.provider:AUTO}")
    private String configuredProvider;

    @Value("${rag.llm.endpoint:http://localhost:11434/api/generate}")
    private String configuredEndpoint;

    @Value("${rag.llm.model:}")
    private String configuredModel;

    @Value("${rag.llm.groq-api-key:${GROQ_API_KEY:}}")
    private String configuredGroqApiKey;

    @Value("${rag.llm.gemini-api-key:${RAG_LLM_GEMINI_API_KEY:${GEMINI_API_KEY:}}}")
    private String configuredGeminiApiKey;

    @Value("${rag.llm.openrouter-api-key:${OPENROUTER_API_KEY:}}")
    private String configuredOpenrouterApiKey;

    private final java.util.concurrent.atomic.AtomicReference<String> currentProvider = new java.util.concurrent.atomic.AtomicReference<>();
    private final java.util.concurrent.atomic.AtomicReference<String> currentGroqKey = new java.util.concurrent.atomic.AtomicReference<>();
    private final java.util.concurrent.atomic.AtomicReference<String> currentGeminiKey = new java.util.concurrent.atomic.AtomicReference<>();
    private final java.util.concurrent.atomic.AtomicReference<String> currentOpenrouterKey = new java.util.concurrent.atomic.AtomicReference<>();
    private final java.util.concurrent.atomic.AtomicReference<String> currentModel = new java.util.concurrent.atomic.AtomicReference<>();
    private final java.util.concurrent.atomic.AtomicReference<String> currentEndpoint = new java.util.concurrent.atomic.AtomicReference<>();

    public RagGenerationService(
            ChromaVectorStoreService vectorStoreService,
            RagCacheService cacheService,
            QueryRelevanceFilter relevanceFilter,
            QueryConcurrencyGuard concurrencyGuard,
            QueryCircuitBreaker circuitBreaker,
            QueryMetricsTracker metricsTracker,
            ObjectMapper objectMapper
    ) {
        this.vectorStoreService = vectorStoreService;
        this.cacheService = cacheService;
        this.relevanceFilter = relevanceFilter;
        this.concurrencyGuard = concurrencyGuard;
        this.circuitBreaker = circuitBreaker;
        this.metricsTracker = metricsTracker;
        this.objectMapper = objectMapper;

        this.streamingExecutor = Executors.newFixedThreadPool(64, r -> {
            Thread t = new Thread(r, "rag-query-worker");
            t.setDaemon(true);
            return t;
        });

        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_2)
                .connectTimeout(Duration.ofSeconds(6))
                .executor(streamingExecutor)
                .build();

        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(4000);
        requestFactory.setReadTimeout(15000);
        this.restClient = RestClient.builder()
                .requestFactory(requestFactory)
                .build();
    }

    @PostConstruct
    public void warmup() {
        currentProvider.set((configuredProvider != null && !configuredProvider.isBlank()) ? configuredProvider.trim().toUpperCase() : "AUTO");
        currentGroqKey.set(configuredGroqApiKey != null ? configuredGroqApiKey.trim() : "");
        currentGeminiKey.set(configuredGeminiApiKey != null ? configuredGeminiApiKey.trim() : "");
        currentOpenrouterKey.set(configuredOpenrouterApiKey != null ? configuredOpenrouterApiKey.trim() : "");
        currentModel.set(configuredModel != null ? configuredModel.trim() : "");
        currentEndpoint.set(configuredEndpoint != null ? configuredEndpoint.trim() : "http://localhost:11434/api/generate");

        log.info("[RAG-QA] Pre-warming query pipeline, caches, and dependency connections... Active Free LLM Provider: {}", currentProvider.get());
        try {
            String probe = "warmup probe";
            cacheService.putEmbedding(probe, vectorStoreService.computeLightweightEmbedding(probe));
            log.info("[RAG-QA] Warmup complete. Query pipeline ready for high-concurrency traffic.");
        } catch (Exception e) {
            log.warn("[RAG-QA] Warmup encountered non-critical notice: {}", e.getMessage());
        }
    }

    public record LlmStatus(
            String activeProvider,
            String activeModel,
            String endpoint,
            boolean hasGroqKey,
            boolean hasGeminiKey,
            boolean hasOpenrouterKey,
            String statusDescription
    ) {}

    public LlmStatus getLlmConfig() {
        String prov = currentProvider.get();
        String groq = currentGroqKey.get();
        String gem = currentGeminiKey.get();
        String openr = currentOpenrouterKey.get();

        String statusDesc;
        if ("GROQ".equals(prov) || ("AUTO".equals(prov) && groq != null && !groq.isBlank())) {
            statusDesc = "Groq Cloud (Free Llama 3.3 / 3.1 Active)";
        } else if ("GEMINI".equals(prov) || ("AUTO".equals(prov) && gem != null && !gem.isBlank())) {
            statusDesc = "Google Gemini (Free Flash Active)";
        } else if ("OPENROUTER".equals(prov) || ("AUTO".equals(prov) && openr != null && !openr.isBlank())) {
            statusDesc = "OpenRouter (Free Models Active)";
        } else if ("OLLAMA".equals(prov)) {
            statusDesc = "Local Ollama Active";
        } else {
            statusDesc = "Intelligent Extractive Synthesis (Add free Groq/Gemini key for AI explanations)";
        }

        return new LlmStatus(
                prov,
                getEffectiveModel(),
                currentEndpoint.get(),
                groq != null && !groq.isBlank(),
                gem != null && !gem.isBlank(),
                openr != null && !openr.isBlank(),
                statusDesc
        );
    }

    public synchronized void updateLlmConfig(Map<String, String> config) {
        if (config == null) return;
        if (config.containsKey("provider") && config.get("provider") != null && !config.get("provider").isBlank()) {
            currentProvider.set(config.get("provider").trim().toUpperCase());
        }
        if (config.containsKey("groqApiKey") && config.get("groqApiKey") != null) {
            currentGroqKey.set(config.get("groqApiKey").trim());
        }
        if (config.containsKey("geminiApiKey") && config.get("geminiApiKey") != null) {
            currentGeminiKey.set(config.get("geminiApiKey").trim());
        }
        if (config.containsKey("openrouterApiKey") && config.get("openrouterApiKey") != null) {
            currentOpenrouterKey.set(config.get("openrouterApiKey").trim());
        }
        if (config.containsKey("apiKey") && config.get("apiKey") != null && !config.get("apiKey").isBlank()) {
            String prov = currentProvider.get();
            String key = config.get("apiKey").trim();
            if ("GROQ".equals(prov) || key.startsWith("gsk_")) {
                currentGroqKey.set(key);
                if (!"GROQ".equals(prov) && "AUTO".equals(prov)) currentProvider.set("GROQ");
            } else if ("OPENROUTER".equals(prov) || key.startsWith("sk-or-")) {
                currentOpenrouterKey.set(key);
                if (!"OPENROUTER".equals(prov) && "AUTO".equals(prov)) currentProvider.set("OPENROUTER");
            } else {
                currentGeminiKey.set(key);
                if (!"GEMINI".equals(prov) && "AUTO".equals(prov)) currentProvider.set("GEMINI");
            }
        }
        if (config.containsKey("model") && config.get("model") != null && !config.get("model").isBlank()) {
            currentModel.set(config.get("model").trim());
        }
        if (config.containsKey("endpoint") && config.get("endpoint") != null && !config.get("endpoint").isBlank()) {
            currentEndpoint.set(config.get("endpoint").trim());
        }
        log.info("[RAG-QA] Updated LLM configuration: provider={}, model={}", currentProvider.get(), getEffectiveModel());
    }

    public Map<String, Object> testLlmConnection(Map<String, String> testConfig) {
        long start = System.currentTimeMillis();
        String testProvider = (testConfig != null && testConfig.get("provider") != null && !testConfig.get("provider").isBlank())
                ? testConfig.get("provider").trim().toUpperCase() : currentProvider.get();
        String apiKey = (testConfig != null) ? testConfig.get("apiKey") : null;
        String model = (testConfig != null) ? testConfig.get("model") : null;
        String endpoint = (testConfig != null) ? testConfig.get("endpoint") : null;

        String effectiveKey = (apiKey != null && !apiKey.isBlank()) ? apiKey.trim() :
                ("GROQ".equals(testProvider) ? currentGroqKey.get() :
                 "OPENROUTER".equals(testProvider) ? currentOpenrouterKey.get() : currentGeminiKey.get());

        try {
            String testPrompt = "Respond with exactly 1 sentence: 'Verified: Free AI model is connected successfully.'";
            String reply = null;

            if ("GROQ".equals(testProvider) || ("AUTO".equals(testProvider) && effectiveKey != null && effectiveKey.startsWith("gsk_"))) {
                String mod = (model != null && !model.isBlank()) ? model : "llama-3.3-70b-versatile";
                reply = directCallGroq(effectiveKey, mod, testPrompt);
            } else if ("GEMINI".equals(testProvider) || ("AUTO".equals(testProvider) && effectiveKey != null && !effectiveKey.isBlank())) {
                String mod = (model != null && !model.isBlank()) ? model : "gemini-1.5-flash";
                reply = directCallGemini(effectiveKey, mod, testPrompt);
            } else if ("OPENROUTER".equals(testProvider)) {
                String mod = (model != null && !model.isBlank()) ? model : "meta-llama/llama-3.2-3b-instruct:free";
                reply = directCallOpenRouter(effectiveKey, mod, testPrompt);
            } else if ("OLLAMA".equals(testProvider)) {
                String endp = (endpoint != null && !endpoint.isBlank()) ? endpoint : currentEndpoint.get();
                String mod = (model != null && !model.isBlank()) ? model : "llama3.2";
                reply = directCallOllama(endp, mod, testPrompt);
            }

            if (reply != null && !reply.isBlank()) {
                return Map.of(
                        "success", true,
                        "provider", testProvider,
                        "model", (model != null && !model.isBlank()) ? model : getEffectiveModel(),
                        "latencyMs", (System.currentTimeMillis() - start),
                        "reply", reply.trim()
                );
            }
            return Map.of(
                    "success", false,
                    "provider", testProvider,
                    "error", "Provider did not return a valid response. Please check your free API key or network connectivity."
            );
        } catch (Exception e) {
            return Map.of(
                    "success", false,
                    "provider", testProvider,
                    "error", e.getMessage() != null ? e.getMessage() : e.toString()
            );
        }
    }

    private String getEffectiveModel() {
        String configured = currentModel.get();
        if (configured != null && !configured.isBlank()) return configured;
        String prov = currentProvider.get();
        if ("GROQ".equals(prov)) return "llama-3.3-70b-versatile";
        if ("GEMINI".equals(prov)) return "gemini-1.5-flash";
        if ("OPENROUTER".equals(prov)) return "meta-llama/llama-3.2-3b-instruct:free";
        return "llama3.2";
    }

    /**
     * Answers a query using context retrieved from the vector store with multi-tier caching,
     * in-flight deduplication, bounded concurrency, relevance filtering, and Free LLM explanation.
     */
    public RagAnswer ask(String query, int topK) {
        long startTime = System.currentTimeMillis();
        String normQuery = (query != null) ? query.trim().replaceAll("\\s+", " ") : "";
        int safeK = Math.max(1, Math.min(20, topK));

        if (normQuery.isBlank()) {
            return new RagAnswer(query, "Please enter a valid query.", List.of(), 0);
        }

        // Instant conversational response for greetings, pleasantries, and meta-questions
        if (relevanceFilter.isConversationalQuery(normQuery)) {
            String conversationalReply = relevanceFilter.getConversationalResponse(normQuery);
            metricsTracker.recordSuccess(System.currentTimeMillis() - startTime);
            return new RagAnswer(query, cleanAnswerFormatting(conversationalReply), List.of(), 0);
        }

        // 1. Check L1 / L2 Distributed Cache First
        RagAnswer cached = cacheService.getAnswer(normQuery, safeK);
        if (cached != null) {
            metricsTracker.recordSuccess(System.currentTimeMillis() - startTime);
            return cached;
        }

        // 2. SingleFlight Deduplication (Coalesces concurrent identical queries)
        String dedupKey = normQuery.toLowerCase() + ":" + safeK;
        try {
            CompletableFuture<RagAnswer> future = concurrencyGuard.executeCoalesced(dedupKey, () -> {
                // 3. Acquire Bounded Execution Permit with Backpressure Queue Protection
                concurrencyGuard.acquirePermit();
                try {
                    return doAskInternal(normQuery, safeK);
                } finally {
                    concurrencyGuard.releasePermit();
                }
            });

            RagAnswer answer = future.get(20, TimeUnit.SECONDS);
            // Cache computed answer
            cacheService.putAnswer(normQuery, safeK, answer);
            metricsTracker.recordSuccess(System.currentTimeMillis() - startTime);
            return answer;
        } catch (QueryConcurrencyGuard.QueryBackpressureException e) {
            metricsTracker.recordThrottled();
            throw e;
        } catch (Exception e) {
            metricsTracker.recordFailure(System.currentTimeMillis() - startTime);
            Throwable cause = (e instanceof ExecutionException && e.getCause() != null) ? e.getCause() : e;
            log.error("[RAG-QA] Query execution failed for '{}': {}", normQuery, cause.getMessage(), cause);
            try {
                return doAskInternal(normQuery, safeK);
            } catch (Exception fallbackEx) {
                return new RagAnswer(
                        normQuery,
                        "We encountered an issue processing your query under current system load. Please try again in a few moments.",
                        List.of(),
                        0
                );
            }
        }
    }

    private RagAnswer doAskInternal(String query, int topK) {
        String searchExpQuery = relevanceFilter.normalizeAndExpandQuery(query);
        int candidatePoolSize = Math.max(topK, 5) * 3;
        List<ChromaVectorStoreService.SearchResult> rawResults = vectorStoreService.search(searchExpQuery, candidatePoolSize);

        // Apply Relevance & Boilerplate Filtering
        List<ChromaVectorStoreService.SearchResult> cleanResults = relevanceFilter.filterRelevantResults(rawResults, searchExpQuery, 0.25);
        int targetK = Math.max(topK, 5);
        if (cleanResults.size() > targetK) {
            cleanResults = cleanResults.subList(0, targetK);
        }

        if (cleanResults.isEmpty()) {
            return new RagAnswer(
                    query,
                    "No verified document context found in the vector index matching your query. Please ensure relevant documents have been uploaded and ingested.",
                    List.of(),
                    0
            );
        }

        List<Citation> citations = new ArrayList<>();
        StringBuilder contextBuilder = new StringBuilder();

        for (int i = 0; i < cleanResults.size(); i++) {
            ChromaVectorStoreService.SearchResult res = cleanResults.get(i);
            String docLabel = (res.sources() != null && res.sources().size() > 1)
                    ? res.document() + " (also in: " + String.join(", ", res.sources()) + ")"
                    : (res.document() != null ? res.document() : "document");

            String rawText = (res.text() != null) ? res.text() : "";
            String previewText = rawText.replaceAll("\\s+", " ").trim();
            if (previewText.length() > 250) {
                previewText = previewText.substring(0, 250) + "...";
            }

            citations.add(new Citation(
                    docLabel,
                    res.page(),
                    res.heading() != null ? res.heading() : "",
                    res.score(),
                    previewText
            ));

            String cleanedChunk = relevanceFilter.stripOutlineAndHeaderArtifacts(rawText);
            if (!cleanedChunk.isBlank()) {
                contextBuilder.append("---\n").append(cleanedChunk).append("\n\n");
            }
        }

        // Execute Multi-Tier Failover Synthesis Loop:
        // Tier 1 (External Cloud LLM via Spring RestClient) -> Tier 2 (Local Ollama via Spring RestClient) -> Tier 3 (Deterministic Extractive)
        String answer = executeMultiTierSynthesis(query, contextBuilder.toString(), cleanResults);

        return new RagAnswer(query, answer, citations, cleanResults.size());
    }

    /**
     * Server-Sent Events (SSE) Streaming query processing.
     * Streams citations context and tokens in real-time to the frontend.
     */
    public void askStream(String query, int topK, SseEmitter emitter) {
        long startTime = System.currentTimeMillis();
        streamingExecutor.submit(() -> {
            try {
                RagAnswer answer = ask(query, topK);
                metricsTracker.recordFirstTokenLatency(System.currentTimeMillis() - startTime);

                // 1. Send context metadata event
                emitter.send(SseEmitter.event().name("context").data(objectMapper.writeValueAsString(Map.of(
                        "totalChunks", answer.totalContextChunks(),
                        "citations", answer.citations()
                ))));

                // 2. Stream tokens in small chunks for responsive UI typing effect
                String[] words = answer.answer().split(" ");
                StringBuilder buffer = new StringBuilder();
                for (int i = 0; i < words.length; i++) {
                    buffer.append(words[i]).append(" ");
                    if (i % 3 == 0 || i == words.length - 1) {
                        emitter.send(SseEmitter.event().name("token").data(buffer.toString()));
                        buffer.setLength(0);
                        Thread.sleep(15);
                    }
                }

                // 3. Send done event
                emitter.send(SseEmitter.event().name("done").data(objectMapper.writeValueAsString(answer)));
                emitter.complete();
            } catch (Exception e) {
                try {
                    emitter.send(SseEmitter.event().name("error").data(e.getMessage()));
                } catch (IOException ignored) {}
                emitter.completeWithError(e);
            }
        });
    }

    /**
     * Executes the Multi-Tier Failover Synthesis Loop:
     * - Tier 1: Primary Cloud LLM via Spring RestClient (Groq, Gemini, OpenRouter) guarded by Circuit Breaker
     * - Tier 2: Local Ollama instance (http://localhost:11434 / host.docker.internal:11434) via Spring RestClient
     * - Tier 3: Local Deterministic Extractive Synthesis component
     */
    public String executeMultiTierSynthesis(String query, String context, List<ChromaVectorStoreService.SearchResult> results) {
        String prompt = buildSystemPrompt(query, context);

        // ==============================================================
        // TIER 1: External Cloud LLM Provider (Guarded by Circuit Breaker)
        // ==============================================================
        if (circuitBreaker.allowExecution()) {
            try {
                String cloudResponse = callTier1CloudLlm(prompt);
                if (cloudResponse != null && !cloudResponse.isBlank()) {
                    circuitBreaker.recordSuccess();
                    log.info("[SYNTHESIS-TIER1] Primary Cloud LLM generation succeeded.");
                    return cleanAnswerFormatting(cloudResponse);
                }
            } catch (Exception ex) {
                circuitBreaker.recordFailure();
                log.warn("[SYNTHESIS-TIER1] Cloud LLM provider failed or throttled: {}. Triggering failover...", ex.getMessage());
            }
        } else {
            log.warn("[SYNTHESIS-TIER1] Circuit Breaker is OPEN. Bypassing cloud provider to prevent pool exhaustion.");
        }

        // ==============================================================
        // TIER 2: Local Ollama Instance (http://localhost:11434)
        // ==============================================================
        try {
            log.info("[SYNTHESIS-TIER2] Routing request to Tier 2: Local Ollama instance via Spring RestClient...");
            String ollamaResponse = callTier2Ollama(prompt);
            if (ollamaResponse != null && !ollamaResponse.isBlank()) {
                log.info("[SYNTHESIS-TIER2] Local Ollama fallback succeeded.");
                return cleanAnswerFormatting(ollamaResponse);
            }
        } catch (Exception ex) {
            log.warn("[SYNTHESIS-TIER2] Local Ollama fallback unavailable: {}. Degrading to Tier 3...", ex.getMessage());
        }

        // ==============================================================
        // TIER 3: Local Deterministic Extractive Synthesis Component
        // ==============================================================
        log.info("[SYNTHESIS-TIER3] Routing transaction to Tier 3: Local Deterministic Extractive Synthesis.");
        String extractiveAnswer = relevanceFilter.synthesizeExtractiveAnswer(query, results);
        return cleanAnswerFormatting(extractiveAnswer);
    }

    private String callTier1CloudLlm(String prompt) throws Exception {
        String provider = currentProvider.get();
        if ("OLLAMA".equalsIgnoreCase(provider)) {
            // Ollama configured as primary; handled by Tier 2 logic
            return null;
        }

        if ("GROQ".equalsIgnoreCase(provider)) {
            return callGroqWithRestClient(prompt);
        } else if ("GEMINI".equalsIgnoreCase(provider)) {
            return callGeminiWithRestClient(prompt);
        } else if ("OPENROUTER".equalsIgnoreCase(provider)) {
            return callOpenRouterWithRestClient(prompt);
        } else {
            // AUTO Mode: Groq -> Gemini -> OpenRouter
            String groqKey = currentGroqKey.get();
            if (groqKey != null && !groqKey.isBlank()) {
                String ans = callGroqWithRestClient(prompt);
                if (ans != null && !ans.isBlank()) return ans;
            }
            String geminiKey = currentGeminiKey.get();
            if (geminiKey != null && !geminiKey.isBlank()) {
                String ans = callGeminiWithRestClient(prompt);
                if (ans != null && !ans.isBlank()) return ans;
            }
            String openRouterKey = currentOpenrouterKey.get();
            if (openRouterKey != null && !openRouterKey.isBlank()) {
                String ans = callOpenRouterWithRestClient(prompt);
                if (ans != null && !ans.isBlank()) return ans;
            }
        }
        return null;
    }

    private String callGroqWithRestClient(String prompt) {
        String key = currentGroqKey.get();
        if (key == null || key.isBlank()) return null;
        try {
            String model = (currentModel.get() != null && !currentModel.get().isBlank())
                    ? currentModel.get() : "llama-3.3-70b-versatile";

            Map<String, Object> body = Map.of(
                    "model", model,
                    "messages", List.of(
                            Map.of("role", "system", "content", "You are an intelligent, articulate enterprise AI document assistant. Deeply analyze the user's query and provide a comprehensive, natural, and helpful explanation. Strict formatting rule: NEVER use asterisks (* or **), NEVER include page numbers, and NEVER include chunk or reference numbers."),
                            Map.of("role", "user", "content", prompt)
                    ),
                    "temperature", 0.3,
                    "max_tokens", 1500
            );

            String responseJson = restClient.post()
                    .uri("https://api.groq.com/openai/v1/chat/completions")
                    .header("Authorization", "Bearer " + key)
                    .header("Content-Type", "application/json")
                    .body(objectMapper.writeValueAsString(body))
                    .retrieve()
                    .body(String.class);

            if (responseJson != null && !responseJson.isBlank()) {
                JsonNode root = objectMapper.readTree(responseJson);
                String text = root.path("choices").path(0).path("message").path("content").asText();
                if (text != null && !text.isBlank()) {
                    return text.trim();
                }
            }
        } catch (Exception e) {
            log.warn("[GROQ-REST-CLIENT] Call failed: {}", e.getMessage());
            throw new RuntimeException("Groq API error: " + e.getMessage(), e);
        }
        return null;
    }

    private String callGeminiWithRestClient(String prompt) {
        String key = currentGeminiKey.get();
        if (key == null || key.isBlank()) return null;
        try {
            String model = (currentModel.get() != null && !currentModel.get().isBlank())
                    ? currentModel.get() : "gemini-1.5-flash";
            String url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + key;

            Map<String, Object> body = Map.of(
                    "contents", List.of(Map.of("parts", List.of(Map.of("text", prompt)))),
                    "generationConfig", Map.of("temperature", 0.3, "maxOutputTokens", 1500)
            );

            String responseJson = restClient.post()
                    .uri(url)
                    .header("Content-Type", "application/json")
                    .body(objectMapper.writeValueAsString(body))
                    .retrieve()
                    .body(String.class);

            if (responseJson != null && !responseJson.isBlank()) {
                JsonNode root = objectMapper.readTree(responseJson);
                String text = root.path("candidates").path(0).path("content").path("parts").path(0).path("text").asText();
                if (text != null && !text.isBlank()) {
                    return text.trim();
                }
            }
        } catch (Exception e) {
            log.warn("[GEMINI-REST-CLIENT] Call failed: {}", e.getMessage());
            throw new RuntimeException("Gemini API error: " + e.getMessage(), e);
        }
        return null;
    }

    private String callOpenRouterWithRestClient(String prompt) {
        String key = currentOpenrouterKey.get();
        if (key == null || key.isBlank()) return null;
        try {
            String model = (currentModel.get() != null && !currentModel.get().isBlank())
                    ? currentModel.get() : "meta-llama/llama-3.2-3b-instruct:free";

            Map<String, Object> body = Map.of(
                    "model", model,
                    "messages", List.of(
                            Map.of("role", "system", "content", "You are an intelligent, articulate enterprise AI document assistant. Deeply analyze the user's query and provide a comprehensive, natural, and helpful explanation. Strict formatting rule: NEVER use asterisks (* or **), NEVER include page numbers, and NEVER include chunk or reference numbers."),
                            Map.of("role", "user", "content", prompt)
                    ),
                    "temperature", 0.3,
                    "max_tokens", 1500
            );

            String responseJson = restClient.post()
                    .uri("https://openrouter.ai/api/v1/chat/completions")
                    .header("Authorization", "Bearer " + key)
                    .header("Content-Type", "application/json")
                    .body(objectMapper.writeValueAsString(body))
                    .retrieve()
                    .body(String.class);

            if (responseJson != null && !responseJson.isBlank()) {
                JsonNode root = objectMapper.readTree(responseJson);
                String text = root.path("choices").path(0).path("message").path("content").asText();
                if (text != null && !text.isBlank()) {
                    return text.trim();
                }
            }
        } catch (Exception e) {
            log.warn("[OPENROUTER-REST-CLIENT] Call failed: {}", e.getMessage());
            throw new RuntimeException("OpenRouter API error: " + e.getMessage(), e);
        }
        return null;
    }

    private String callTier2Ollama(String prompt) {
        String endpoint = currentEndpoint.get();
        String model = (currentModel.get() != null && !currentModel.get().isBlank()) ? currentModel.get() : "llama3.2";

        List<String> targetEndpoints = new ArrayList<>();
        if (endpoint != null && !endpoint.isBlank()) {
            targetEndpoints.add(endpoint);
        }
        if (!targetEndpoints.contains("http://localhost:11434/api/generate")) {
            targetEndpoints.add("http://localhost:11434/api/generate");
        }
        targetEndpoints.add("http://host.docker.internal:11434/api/generate");

        Map<String, Object> body = Map.of(
                "model", model,
                "prompt", prompt,
                "stream", false
        );

        for (String targetUrl : targetEndpoints) {
            try {
                String requestJson = objectMapper.writeValueAsString(body);
                String responseJson = restClient.post()
                        .uri(targetUrl)
                        .header("Content-Type", "application/json")
                        .body(requestJson)
                        .retrieve()
                        .body(String.class);

                if (responseJson != null && !responseJson.isBlank()) {
                    JsonNode root = objectMapper.readTree(responseJson);
                    String text = root.path("response").asText();
                    if (text != null && !text.isBlank()) {
                        return text.trim();
                    }
                }
            } catch (Exception e) {
                log.debug("[OLLAMA-REST-CLIENT] Endpoint {} failed: {}", targetUrl, e.getMessage());
            }
        }
        return null;
    }

    private String directCallGroq(String apiKey, String model, String prompt) throws Exception {
        Map<String, Object> body = Map.of(
                "model", model,
                "messages", List.of(Map.of("role", "user", "content", prompt)),
                "max_tokens", 80
        );
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create("https://api.groq.com/openai/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)))
                .timeout(Duration.ofSeconds(8))
                .build();
        HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() == 200) {
            JsonNode root = objectMapper.readTree(resp.body());
            return root.path("choices").path(0).path("message").path("content").asText();
        }
        throw new IOException("Groq returned status " + resp.statusCode() + ": " + resp.body());
    }

    private String directCallGemini(String apiKey, String model, String prompt) throws Exception {
        String url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + apiKey;
        Map<String, Object> body = Map.of(
                "contents", List.of(Map.of("parts", List.of(Map.of("text", prompt))))
        );
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)))
                .timeout(Duration.ofSeconds(10))
                .build();
        HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() == 200) {
            JsonNode root = objectMapper.readTree(resp.body());
            return root.path("candidates").path(0).path("content").path("parts").path(0).path("text").asText();
        }
        throw new IOException("Gemini returned status " + resp.statusCode() + ": " + resp.body());
    }

    private String directCallOpenRouter(String apiKey, String model, String prompt) throws Exception {
        Map<String, Object> body = Map.of(
                "model", model,
                "messages", List.of(Map.of("role", "user", "content", prompt)),
                "max_tokens", 80
        );
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create("https://openrouter.ai/api/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)))
                .timeout(Duration.ofSeconds(10))
                .build();
        HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() == 200) {
            JsonNode root = objectMapper.readTree(resp.body());
            return root.path("choices").path(0).path("message").path("content").asText();
        }
        throw new IOException("OpenRouter returned status " + resp.statusCode() + ": " + resp.body());
    }

    private String directCallOllama(String endpoint, String model, String prompt) throws Exception {
        Map<String, Object> body = Map.of(
                "model", model,
                "prompt", prompt,
                "stream", false
        );
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)))
                .timeout(Duration.ofSeconds(6))
                .build();
        HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() == 200) {
            JsonNode root = objectMapper.readTree(resp.body());
            return root.path("response").asText();
        }
        throw new IOException("Ollama returned status " + resp.statusCode() + ": " + resp.body());
    }

    private String buildSystemPrompt(String query, String context) {
        return "You are an intelligent, articulate enterprise AI document assistant.\n\n" +
                "The user is asking: \"" + query + "\"\n\n" +
                "INSTRUCTIONS FOR YOUR RESPONSE:\n" +
                "1. ANALYZE USER QUERY & INTENT: Deeply understand what the user wants to know. When the user asks 'tell me about X' or asks an open-ended question, provide a comprehensive, well-structured, and insightful explanation synthesizing all relevant history, key figures, governance, geography, concepts, and factual details from the context.\n" +
                "2. NATURAL EXPLANATION: Write fluent, cohesive paragraphs and organized plain bullet points. Explain concepts clearly and professionally as a knowledgeable expert.\n" +
                "3. STRICT FORMATTING RULES (CRITICAL):\n" +
                "   - Do NOT use asterisks (*) or markdown bold (**) anywhere in the response. Output strictly plain readable text.\n" +
                "   - Do NOT include any page numbers (e.g. never write 'Page 8', 'p. 20', '(Page 15)', etc.).\n" +
                "   - Do NOT include chunk numbers, bracketed numbers, or index references (e.g. never write '[1]', 'Chunk 1', or '[2]').\n" +
                "   - Do NOT mention document filenames, PDF titles, or source paths.\n" +
                "   - Do NOT output raw OCR junk, scanner phone numbers, or outline codes.\n" +
                "   - For bullet points, start lines with '• ' followed by plain text without bolding.\n\n" +
                "DOCUMENT CONTEXT:\n" + context + "\n\n" +
                "DETAILED EXPLANATION (Plain text, no asterisks, no page or chunk numbers):";
    }

    /**
     * Cleans and sanitizes answer text ensuring no asterisks (* or **), page numbers,
     * chunk brackets, outline noise, or publisher watermarks leak to the user.
     */
    public String cleanAnswerFormatting(String answer) {
        if (answer == null || answer.isBlank()) return "";

        String cleaned = answer;

        // 1. Remove all markdown bold and italic asterisks
        cleaned = cleaned.replaceAll("\\*{1,}", "");

        // 2. Remove markdown header hashes: ### Heading -> Heading
        cleaned = cleaned.replaceAll("(?m)^\\s*#{1,6}\\s*", "");

        // 3. Remove chunk references: [1], [2], [1, 2], [Chunk 1], Chunk 1, chunk 2
        cleaned = cleaned.replaceAll("\\[\\s*\\d+\\s*(?:,\\s*\\d+\\s*)*\\]", "");
        cleaned = cleaned.replaceAll("(?i)\\[?\\bchunk\\s*\\d+\\b\\]?", "");

        // 4. Remove page references: (Page 8), (Page 8-10), Page 8, p. 8, (p. 20), (Pages 12-15)
        cleaned = cleaned.replaceAll("(?i)\\(?\\s*\\b(?:pages?|p\\.)\\s*\\d+(?:\\s*[-–—to]+\\s*\\d+)?\\s*\\)?", "");

        // 5. Remove source/document references: "Source: ...", "**Source: ...**", "Document: ..."
        cleaned = cleaned.replaceAll("(?im)^\\s*(?:source|document|ref|reference):\\s*[^\\r\\n]+", "");

        // 6. Remove exam/syllabus outline headers: "PAPER - I", "PAPER-II", "Section-A", "Section-B"
        cleaned = cleaned.replaceAll("(?i)\\b(?:PAPER|Paper)\\s*[-–—]?\\s*[I|V0-9]+", "");
        cleaned = cleaned.replaceAll("(?i)\\b(?:Section|SECTION)\\s*[-–—]?\\s*[A-Z0-9]+", "");

        // 7. Remove OCR junk / publisher watermarks: e.g. "ToppersNotes / 9828-286-909 5"
        cleaned = cleaned.replaceAll("(?i)T\\s*oppersNotes[^\\r\\n]*", "");
        cleaned = cleaned.replaceAll("\\b\\d{4,5}[-\\s]?\\d{3}[-\\s]?\\d{3,4}\\b", "");

        // 8. Standardize bullet points: "- ", "* " -> "• "
        cleaned = cleaned.replaceAll("(?m)^\\s*[-*]\\s+", "• ");

        // 9. Clean up empty parentheses left behind: "()", "( )"
        cleaned = cleaned.replaceAll("\\(\\s*\\)", "");
        cleaned = cleaned.replaceAll("\\s+,", ",");
        cleaned = cleaned.replaceAll("\\s+\\.", ".");

        // 10. Clean up multiple empty lines or dangling spaces
        cleaned = cleaned.replaceAll("[ \\t]+", " ");
        cleaned = cleaned.replaceAll("(\\r?\\n){3,}", "\n\n");
        cleaned = cleaned.replaceAll("(?m)^[ \\t]+", "");

        return cleaned.trim();
    }

    public record Citation(String document, int page, String heading, double similarityScore, String preview) {}

    public record RagAnswer(String query, String answer, List<Citation> citations, int totalContextChunks) {}
}
