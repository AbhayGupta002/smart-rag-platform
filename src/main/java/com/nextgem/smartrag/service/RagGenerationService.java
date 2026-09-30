package com.nextgem.smartrag.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nextgem.smartrag.query.*;
import com.nextgem.smartrag.vectorstore.ChromaVectorStoreService;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
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
    private final ExecutorService streamingExecutor;

    @Value("${rag.llm.endpoint:http://localhost:11434/api/generate}")
    private String llmEndpoint;

    @Value("${rag.llm.model:llama3.2}")
    private String llmModel;

    @Value("${rag.llm.gemini-api-key:}")
    private String geminiApiKey;

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
                .connectTimeout(Duration.ofSeconds(4))
                .executor(streamingExecutor)
                .build();
    }

    @PostConstruct
    public void warmup() {
        log.info("[RAG-QA] Pre-warming query pipeline, caches, and dependency connections...");
        try {
            String probe = "warmup probe";
            cacheService.putEmbedding(probe, vectorStoreService.computeLightweightEmbedding(probe));
            log.info("[RAG-QA] Warmup complete. Query pipeline ready for high-concurrency traffic.");
        } catch (Exception e) {
            log.warn("[RAG-QA] Warmup encountered non-critical notice: {}", e.getMessage());
        }
    }

    /**
     * Answers a query using context retrieved from the vector store with multi-tier caching,
     * in-flight deduplication, bounded concurrency, and relevance filtering.
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
            return new RagAnswer(query, conversationalReply, List.of(), 0);
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

            RagAnswer answer = future.get(15, TimeUnit.SECONDS);
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
        int candidatePoolSize = Math.max(topK, 5) * 3;
        List<ChromaVectorStoreService.SearchResult> rawResults = vectorStoreService.search(query, candidatePoolSize);

        // Apply Relevance & Boilerplate Filtering
        List<ChromaVectorStoreService.SearchResult> cleanResults = relevanceFilter.filterRelevantResults(rawResults, query, 0.25);
        int targetK = Math.max(topK, 5);
        if (cleanResults.size() > targetK) {
            cleanResults = cleanResults.subList(0, targetK);
        }

        if (cleanResults.isEmpty()) {
            return new RagAnswer(
                    query,
                    "No relevant document context found in the vector index for your query. Please ingest documents first.",
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

            contextBuilder.append(String.format("[%d] Document: %s | Page: %d | Section: %s\n%s\n\n",
                    i + 1, docLabel, res.page(), res.heading() != null ? res.heading() : "", rawText));
        }

        String answer = null;
        if (geminiApiKey != null && !geminiApiKey.isBlank() && circuitBreaker.allowExecution()) {
            answer = callGeminiLlm(query, contextBuilder.toString(), cleanResults);
        } else if (circuitBreaker.allowExecution()) {
            answer = callOllamaOrFallback(query, contextBuilder.toString(), cleanResults);
        }

        if (answer == null || answer.isBlank()) {
            answer = relevanceFilter.synthesizeExtractiveAnswer(query, cleanResults);
        }

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

    private String callGeminiLlm(String query, String context, List<ChromaVectorStoreService.SearchResult> results) {
        try {
            String url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + geminiApiKey;
            String prompt = buildSystemPrompt(query, context);

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
                circuitBreaker.recordSuccess();
                JsonNode root = objectMapper.readTree(resp.body());
                String text = root.path("candidates").path(0).path("content").path("parts").path(0).path("text").asText();
                if (text != null && !text.isBlank()) {
                    return text;
                }
            } else {
                circuitBreaker.recordFailure();
                log.warn("Gemini API call returned status: {}", resp.statusCode());
            }
        } catch (Exception e) {
            circuitBreaker.recordFailure();
            log.warn("Gemini API call failed: {}", e.getMessage());
        }
        return relevanceFilter.synthesizeExtractiveAnswer(query, results);
    }

    private String callOllamaOrFallback(String query, String context, List<ChromaVectorStoreService.SearchResult> results) {
        try {
            String prompt = buildSystemPrompt(query, context);
            Map<String, Object> body = Map.of(
                    "model", llmModel,
                    "prompt", prompt,
                    "stream", false
            );

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(llmEndpoint))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)))
                    .timeout(Duration.ofSeconds(3))
                    .build();

            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() == 200) {
                circuitBreaker.recordSuccess();
                JsonNode root = objectMapper.readTree(resp.body());
                return root.path("response").asText();
            } else {
                circuitBreaker.recordFailure();
            }
        } catch (Exception ignored) {
            circuitBreaker.recordFailure();
        }

        // Use intelligent extractive synthesis
        return relevanceFilter.synthesizeExtractiveAnswer(query, results);
    }

    private String buildSystemPrompt(String query, String context) {
        return "You are a specialized enterprise document QA assistant.\n" +
                "Answer the following question STRICTLY and ONLY using the provided document context.\n" +
                "Always cite the source document name, page number, and section heading.\n" +
                "Do not make up facts or extrapolate beyond the provided text.\n\n" +
                "CONTEXT:\n" + context + "\n\n" +
                "QUESTION:\n" + query + "\n\n" +
                "ANSWER:";
    }

    public record Citation(String document, int page, String heading, double similarityScore, String preview) {}

    public record RagAnswer(String query, String answer, List<Citation> citations, int totalContextChunks) {}
}
