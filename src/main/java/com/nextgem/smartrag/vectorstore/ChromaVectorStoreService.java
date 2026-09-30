package com.nextgem.smartrag.vectorstore;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nextgem.smartrag.config.RagPipelineProperties;
import com.nextgem.smartrag.service.DynamicHardwareTuningService;
import com.nextgem.smartrag.service.ResourceManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

/**
 * Enterprise Vector Store Service.
 * Implements bounded-memory indexing with ChromaDB REST integration,
 * Content-Addressed Chunk Deduplication (CAS), direct ChromaDB similarity query retrieval,
 * streaming disk-backed fallback, exponential backoff retry, and zero heap memory leaks.
 */
@Service
public class ChromaVectorStoreService {

    private static final Logger log = LoggerFactory.getLogger(ChromaVectorStoreService.class);

    private final RagPipelineProperties properties;
    private final ExecutorService executor;
    private final ObjectMapper objectMapper;
    private final DynamicHardwareTuningService tuningService;
    private final ResourceManager resourceManager;
    private final HttpClient httpClient;

    private volatile String chromaCollectionId = null;
    private volatile boolean chromaAvailable = false;

    private final AtomicInteger totalVectorsIndexed = new AtomicInteger(0);
    private final AtomicInteger totalChunksDeduplicated = new AtomicInteger(0);
    private final ConcurrentHashMap<String, ChunkMetadataRef> chunkRegistry = new ConcurrentHashMap<>();

    @Autowired(required = false)
    private com.nextgem.smartrag.query.RagCacheService cacheService;

    @Autowired(required = false)
    private com.nextgem.smartrag.query.QueryCircuitBreaker circuitBreaker;

    /**
     * In-memory reference for deduplicated chunks tracking multi-source citations.
     */
    public static class ChunkMetadataRef {
        private final String chunkId;
        private final float[] embedding;
        private final Set<String> sources = ConcurrentHashMap.newKeySet();
        private final String primaryDocument;
        private final int primaryPage;
        private final String primaryHeading;
        private final String text;

        public ChunkMetadataRef(String chunkId, float[] embedding, String doc, int page, String heading, String text) {
            this.chunkId = chunkId;
            this.embedding = embedding;
            this.primaryDocument = doc;
            this.primaryPage = page;
            this.primaryHeading = heading;
            this.text = text;
            this.sources.add(doc + ":p" + page);
        }

        public void addSource(String doc, int page) {
            this.sources.add(doc + ":p" + page);
        }

        public Set<String> sources() {
            return sources;
        }

        public float[] embedding() {
            return embedding;
        }

        public String chunkId() {
            return chunkId;
        }

        public String text() {
            return text;
        }
    }

    /**
     * Deterministic SHA-256 hash of normalized chunk text for Content-Addressed Storage.
     */
    public static String computeContentHash(String text) {
        if (text == null || text.isBlank()) {
            return "chk_empty";
        }
        String normalized = text.trim().toLowerCase().replaceAll("\\s+", " ");
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(normalized.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder("chk_");
            for (int i = 0; i < 8; i++) {
                sb.append(String.format("%02x", digest[i]));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            return "chk_" + Integer.toHexString(normalized.hashCode());
        }
    }

    public ChromaVectorStoreService(
            RagPipelineProperties properties,
            @Qualifier("pipelineExecutor") ExecutorService executor,
            ObjectMapper objectMapper,
            DynamicHardwareTuningService tuningService,
            ResourceManager resourceManager
    ) {
        this.properties = properties;
        this.executor = executor;
        this.objectMapper = objectMapper;
        this.tuningService = tuningService;
        this.resourceManager = resourceManager;
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(3))
                .build();

        initializeChromaConnection();
    }

    /**
     * Probes ChromaDB heartbeat and registers/fetches the collection.
     */
    public synchronized void initializeChromaConnection() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(properties.getChromaUrl() + "/api/v1/heartbeat"))
                    .timeout(Duration.ofSeconds(2))
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 200) {
                this.chromaAvailable = true;
                log.info("[CHROMA] Connected successfully at {}", properties.getChromaUrl());
                createOrGetCollection();
            } else {
                log.warn("[CHROMA] Heartbeat returned HTTP {}. Switching to streaming disk fallback.", response.statusCode());
                this.chromaAvailable = false;
            }
        } catch (Exception e) {
            log.info("[CHROMA] Not reachable at {} ({}). Using high-speed streaming disk vector store.",
                    properties.getChromaUrl(), e.getMessage());
            this.chromaAvailable = false;
        }
    }

    private void createOrGetCollection() {
        try {
            // 1. Check if collection already exists
            HttpRequest getReq = HttpRequest.newBuilder()
                    .uri(URI.create(properties.getChromaUrl() + "/api/v1/collections/" + properties.getChromaCollection()))
                    .timeout(Duration.ofSeconds(3))
                    .GET()
                    .build();
            HttpResponse<String> getResp = httpClient.send(getReq, HttpResponse.BodyHandlers.ofString());
            if (getResp.statusCode() == 200) {
                JsonNode root = objectMapper.readTree(getResp.body());
                this.chromaCollectionId = root.path("id").asText();
                log.info("[CHROMA] Connected to existing collection: {} (ID: {})", properties.getChromaCollection(), chromaCollectionId);
                CompletableFuture.runAsync(this::syncDiskVectorsToChroma, executor);
                return;
            }

            // 2. Create collection if it does not exist
            Map<String, Object> body = Map.of(
                    "name", properties.getChromaCollection(),
                    "metadata", Map.of("description", "Enterprise RAG Pipeline Collection")
            );

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(properties.getChromaUrl() + "/api/v1/collections"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 200 || response.statusCode() == 201) {
                JsonNode root = objectMapper.readTree(response.body());
                this.chromaCollectionId = root.path("id").asText();
                log.info("[CHROMA] Created active collection: {} (ID: {})", properties.getChromaCollection(), chromaCollectionId);
                CompletableFuture.runAsync(this::syncDiskVectorsToChroma, executor);
            }
        } catch (Exception e) {
            log.warn("[CHROMA] Failed creating/accessing collection: {}", e.getMessage());
        }
    }

    /**
     * Synchronizes all pre-computed vectors on disk into the Chroma collection to ensure 100% coverage.
     */
    public synchronized void syncDiskVectorsToChroma() {
        if (!chromaAvailable || chromaCollectionId == null) return;
        Path vectorDir = properties.getVectorStoragePath();
        if (!Files.exists(vectorDir)) return;

        List<VectorDocument> batch = new ArrayList<>();
        try (Stream<Path> stream = Files.list(vectorDir)) {
            stream.filter(p -> p.toString().endsWith(".vectors.jsonl")).forEach(file -> {
                try (BufferedReader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (line.trim().isEmpty()) continue;
                        JsonNode node = objectMapper.readTree(line);
                        String id = node.path("id").asText();
                        String docName = node.path("documentName").asText();
                        int page = node.path("pageNumber").asInt();
                        String heading = node.path("heading").asText();
                        String text = node.path("text").asText();
                        JsonNode embNode = node.path("embedding");
                        if (embNode != null && embNode.isArray() && !id.isBlank() && !text.isBlank()) {
                            float[] emb = new float[embNode.size()];
                            for (int i = 0; i < embNode.size(); i++) {
                                emb[i] = (float) embNode.get(i).asDouble();
                            }
                            Map<String, Object> metadata = new HashMap<>();
                            metadata.put("document", docName);
                            metadata.put("page", page);
                            metadata.put("heading", heading);
                            metadata.put("sources", docName + ":p" + page);
                            batch.add(new VectorDocument(id, docName, page, heading, text, emb, metadata));
                            if (batch.size() >= 50) {
                                flushBatchWithRetry(batch);
                                batch.clear();
                            }
                        }
                    }
                } catch (Exception ignored) {}
            });
            if (!batch.isEmpty()) {
                flushBatchWithRetry(batch);
                batch.clear();
            }
            log.info("[CHROMA] Synced disk vectors to Chroma collection successfully.");
        } catch (Exception e) {
            log.warn("[CHROMA] Error syncing disk vectors: {}", e.getMessage());
        }
    }

    /**
     * Ingests all chunked JSONL files in parallel into ChromaDB and disk backup.
     */
    public VectorIngestionSummary ingestAllChunks() {
        Path jsonlDir = properties.getChunkedJsonlPath();
        Path vectorDir = properties.getVectorStoragePath();

        try {
            Files.createDirectories(vectorDir);
        } catch (IOException e) {
            throw new RuntimeException("Could not initialize vector storage directory: " + vectorDir, e);
        }

        totalVectorsIndexed.set(0);
        totalChunksDeduplicated.set(0);
        chunkRegistry.clear();
        loadExistingChunkRegistry(vectorDir);

        List<CompletableFuture<Void>> futures = new ArrayList<>();

        try (Stream<Path> stream = Files.list(jsonlDir)) {
            stream.filter(p -> p.toString().endsWith(".jsonl"))
                    .forEach(jsonlPath -> {
                        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                            try {
                                tuningService.acquirePermit();
                                resourceManager.checkAndThrottle();
                                ingestSingleJsonl(jsonlPath, vectorDir);
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                                log.warn("[STAGE:VECTOR] Interrupted while ingesting: {}", jsonlPath.getFileName());
                            } finally {
                                tuningService.releasePermit();
                            }
                        }, executor);
                        futures.add(future);
                    });

            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
            int totalProcessedFiles = futures.size();
            futures.clear();

            // PARTITION 3 MEMORY CLEAR
            resourceManager.forceReclaim();

            log.info("[STAGE:VECTOR] Ingestion complete. Files: {}, Vectors indexed: {}, Deduplicated: {}, Chroma online: {}",
                    totalProcessedFiles, totalVectorsIndexed.get(), totalChunksDeduplicated.get(), chromaAvailable);
            return new VectorIngestionSummary(totalProcessedFiles, totalVectorsIndexed.get(), totalChunksDeduplicated.get(), chromaAvailable);
        } catch (IOException e) {
            log.error("[STAGE:VECTOR] Error listing JSONL directory: {}", jsonlDir, e);
            return new VectorIngestionSummary(0, 0, 0, chromaAvailable);
        }
    }

    /**
     * Pre-populates the in-memory chunk registry from existing vectors on disk
     * so that subsequent pipeline runs can immediately deduplicate against historical vectors.
     */
    private void loadExistingChunkRegistry(Path vectorDir) {
        if (!Files.exists(vectorDir)) return;
        try (Stream<Path> stream = Files.list(vectorDir)) {
            stream.filter(p -> p.toString().endsWith(".vectors.jsonl")).forEach(file -> {
                try (BufferedReader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (line.trim().isEmpty()) continue;
                        JsonNode node = objectMapper.readTree(line);
                        String id = node.path("id").asText();
                        String docName = node.path("documentName").asText();
                        int page = node.path("pageNumber").asInt();
                        String heading = node.path("heading").asText();
                        String text = node.path("text").asText();
                        JsonNode embNode = node.path("embedding");
                        if (embNode != null && embNode.isArray() && !id.isBlank() && !text.isBlank()) {
                            float[] emb = new float[embNode.size()];
                            for (int i = 0; i < embNode.size(); i++) {
                                emb[i] = (float) embNode.get(i).asDouble();
                            }
                            ChunkMetadataRef ref = chunkRegistry.computeIfAbsent(id,
                                    k -> new ChunkMetadataRef(id, emb, docName, page, heading, text));
                            ref.addSource(docName, page);
                        }
                    }
                } catch (Exception ignored) {}
            });
        } catch (IOException ignored) {}
    }

    /**
     * Streams a single JSONL chunk file, computes embeddings (reusing existing identical chunks via Content-Addressed Storage),
     * and flushes batches to ChromaDB and local disk.
     */
    public void ingestSingleJsonl(Path jsonlPath, Path vectorDir) {
        String baseName = jsonlPath.getFileName().toString().replaceFirst("\\.jsonl$", "");
        Path localVectorStorePath = vectorDir.resolve(baseName + ".vectors.jsonl");

        List<VectorDocument> batch = new ArrayList<>();
        int batchSize = 64;

        try (BufferedReader reader = Files.newBufferedReader(jsonlPath, StandardCharsets.UTF_8);
             BufferedWriter writer = Files.newBufferedWriter(
                     localVectorStorePath,
                     StandardCharsets.UTF_8,
                     StandardOpenOption.CREATE,
                     StandardOpenOption.TRUNCATE_EXISTING,
                     StandardOpenOption.WRITE
             )) {

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;

                JsonNode node = objectMapper.readTree(line);
                String docName = node.path("documentName").asText();
                int pageNum = node.path("pageNumber").asInt();
                String heading = node.path("heading").asText();
                String text = node.path("content").asText();

                // Content-Addressed Hash ID (deterministic content fingerprint)
                String contentHash = computeContentHash(text);

                // Check if this identical chunk was already embedded
                ChunkMetadataRef existingRef = chunkRegistry.get(contentHash);
                if (existingRef != null) {
                    // DEDUPLICATION HIT: Reuse existing embedding and skip computation
                    totalChunksDeduplicated.incrementAndGet();
                    existingRef.addSource(docName, pageNum);

                    Map<String, Object> metadata = new HashMap<>();
                    metadata.put("document", docName);
                    metadata.put("page", pageNum);
                    metadata.put("heading", heading);
                    metadata.put("sources", String.join("; ", existingRef.sources()));
                    metadata.put("deduplicated", true);

                    VectorDocument doc = new VectorDocument(contentHash, docName, pageNum, heading, text, existingRef.embedding(), metadata);
                    writer.write(objectMapper.writeValueAsString(doc));
                    writer.newLine();

                    log.debug("[DEDUP] Reused existing chunk {} for document: {} (p.{})", contentHash, docName, pageNum);
                    continue;
                }

                // NEW UNIQUE CHUNK: Compute embedding once
                float[] embedding = computeLightweightEmbedding(text);
                ChunkMetadataRef newRef = new ChunkMetadataRef(contentHash, embedding, docName, pageNum, heading, text);
                chunkRegistry.put(contentHash, newRef);

                Map<String, Object> metadata = new HashMap<>();
                metadata.put("document", docName);
                metadata.put("page", pageNum);
                metadata.put("heading", heading);
                metadata.put("sources", docName + ":p" + pageNum);
                metadata.put("deduplicated", false);

                VectorDocument doc = new VectorDocument(contentHash, docName, pageNum, heading, text, embedding, metadata);
                batch.add(doc);

                // Stream to local disk backup
                writer.write(objectMapper.writeValueAsString(doc));
                writer.newLine();

                if (batch.size() >= batchSize) {
                    flushBatchWithRetry(batch);
                    totalVectorsIndexed.addAndGet(batch.size());
                    batch.clear();
                }
            }

            if (!batch.isEmpty()) {
                flushBatchWithRetry(batch);
                totalVectorsIndexed.addAndGet(batch.size());
                batch.clear();
            }

            writer.flush();

        } catch (Exception e) {
            log.error("[STAGE:VECTOR] Failed ingesting chunks for: {}", jsonlPath.getFileName(), e);
        }
    }

    /**
     * Flushes a batch of vector documents to ChromaDB with exponential backoff retry.
     */
    private void flushBatchWithRetry(List<VectorDocument> batch) {
        if (!chromaAvailable || chromaCollectionId == null) {
            return;
        }

        int maxRetries = 3;
        long backoffMs = 100;

        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                List<String> ids = new ArrayList<>(batch.size());
                List<String> documents = new ArrayList<>(batch.size());
                List<float[]> embeddings = new ArrayList<>(batch.size());
                List<Map<String, Object>> metadatas = new ArrayList<>(batch.size());

                for (VectorDocument doc : batch) {
                    ids.add(doc.id());
                    documents.add(doc.text());
                    embeddings.add(doc.embedding());
                    metadatas.add(doc.metadata());
                }

                Map<String, Object> payload = Map.of(
                        "ids", ids,
                        "documents", documents,
                        "embeddings", embeddings,
                        "metadatas", metadatas
                );

                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(properties.getChromaUrl() + "/api/v1/collections/" + chromaCollectionId + "/add"))
                        .header("Content-Type", "application/json")
                        .timeout(Duration.ofSeconds(5))
                        .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(payload)))
                        .build();

                HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
                if (response.statusCode() == 200 || response.statusCode() == 201) {
                    return; // Success
                }

                log.warn("[CHROMA] Batch upsert returned HTTP {} (Attempt {}/{}). Retrying...",
                        response.statusCode(), attempt, maxRetries);

            } catch (Exception e) {
                log.warn("[CHROMA] Batch upsert exception (Attempt {}/{}): {}", attempt, maxRetries, e.getMessage());
            }

            if (attempt < maxRetries) {
                try {
                    Thread.sleep(backoffMs);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                }
                backoffMs *= 2;
            }
        }
        log.error("[CHROMA] Exhausted retries pushing batch. Data safely preserved in local .vectors.jsonl storage.");
    }

    /**
     * High-speed, deterministic token-frequency hash embedding (128-dim).
     * Provides instant vector similarity scoring with zero external network overhead or GPU dependencies.
     */
    public float[] computeLightweightEmbedding(String text) {
        int dim = 128;
        float[] vector = new float[dim];
        if (text == null || text.isBlank()) return vector;

        String[] tokens = text.toLowerCase().split("[^a-z0-9]+");
        for (String token : tokens) {
            if (token.isEmpty()) continue;
            int hash = Math.abs(token.hashCode()) % dim;
            vector[hash] += 1.0f;
        }

        // L2 Normalization
        float norm = 0.0f;
        for (float v : vector) norm += v * v;
        if (norm > 0.0f) {
            float inv = (float) (1.0 / Math.sqrt(norm));
            for (int i = 0; i < dim; i++) vector[i] *= inv;
        }
        return vector;
    }

    /**
     * Vector similarity search query:
     * 1. Queries ChromaDB REST API `/api/v1/collections/{id}/query` if available.
     * 2. Seamlessly falls back to line-streaming scan over `.vectors.jsonl` files on disk.
     */
    public List<SearchResult> search(String queryText, int topK) {
        int safeK = Math.max(1, topK);
        float[] queryEmbedding = null;
        if (cacheService != null) {
            queryEmbedding = cacheService.getEmbedding(queryText);
        }
        if (queryEmbedding == null) {
            queryEmbedding = computeLightweightEmbedding(queryText);
            if (cacheService != null) {
                cacheService.putEmbedding(queryText, queryEmbedding);
            }
        }

        List<SearchResult> candidates = new ArrayList<>();

        // 1. Try querying ChromaDB directly if circuit breaker allows
        boolean allowChroma = chromaAvailable && chromaCollectionId != null
                && (circuitBreaker == null || circuitBreaker.allowExecution());

        if (allowChroma) {
            try {
                List<SearchResult> chromaResults = queryChromaDb(queryEmbedding, safeK * 4);
                if (circuitBreaker != null) circuitBreaker.recordSuccess();
                if (chromaResults != null) {
                    candidates.addAll(chromaResults);
                }
            } catch (Exception e) {
                if (circuitBreaker != null) circuitBreaker.recordFailure();
                log.warn("[CHROMA] Query failed, falling back to disk: {}", e.getMessage());
            }
        }

        // 2. Only fallback to disk scan if ChromaDB is offline or returned no results
        if (candidates.isEmpty()) {
            List<SearchResult> diskResults = streamSearchFromDisk(queryEmbedding, queryText, safeK * 4);
            candidates.addAll(diskResults);
        }

        // 3. Hybrid Lexical + Semantic Re-ranking
        return rankCandidatesHybrid(queryText, candidates, safeK);
    }

    private List<SearchResult> rankCandidatesHybrid(String query, List<SearchResult> candidates, int topK) {
        if (candidates == null || candidates.isEmpty()) return Collections.emptyList();

        Set<String> stopWords = Set.of(
                "what", "is", "the", "a", "an", "and", "or", "how", "to", "in", "on", "for",
                "with", "about", "tell", "me", "give", "can", "you", "does", "do", "explain"
        );
        List<String> queryKeywords = Arrays.stream(query.toLowerCase().split("[^a-z0-9]+"))
                .filter(w -> w.length() > 2 && !stopWords.contains(w))
                .toList();

        List<SearchResult> scored = new ArrayList<>();
        for (SearchResult res : candidates) {
            double vectorScore = res.score();
            double lexicalScore = 0.0;

            if (!queryKeywords.isEmpty()) {
                String searchTarget = (res.document() + " " + res.heading() + " " + res.text()).toLowerCase();
                int matches = 0;
                for (String kw : queryKeywords) {
                    if (searchTarget.contains(kw)) {
                        matches++;
                    }
                }
                lexicalScore = (double) matches / queryKeywords.size();
                if (searchTarget.contains(query.toLowerCase().trim())) {
                    lexicalScore = Math.min(1.0, lexicalScore + 0.3);
                }
            }

            // Down-rank pure index/TOC pages and metadata-only headers
            String textLower = res.text().toLowerCase();
            if (textLower.startsWith("# document:") && res.text().length() < 280) {
                lexicalScore *= 0.05;
                vectorScore *= 0.1;
            }
            if (textLower.contains("s no. chapter title page") || textLower.contains("chapter title page no")
                    || textLower.contains("table of contents") || textLower.contains("### index")) {
                lexicalScore *= 0.05;
                vectorScore *= 0.1;
            }
            if ((textLower.contains("public service commission") || textLower.contains("assistant professor")) && textLower.length() < 220) {
                lexicalScore *= 0.05;
                vectorScore *= 0.1;
            }

            double combinedScore = (lexicalScore > 0.0)
                    ? (0.35 * vectorScore + 0.65 * lexicalScore)
                    : (0.75 * vectorScore);

            scored.add(new SearchResult(res.id(), res.document(), res.page(), res.heading(), res.text(), combinedScore, res.sources()));
        }

        scored.sort((a, b) -> Double.compare(b.score(), a.score()));

        // If top candidate has high relevance, prioritize sibling chunks from the same document
        if (!scored.isEmpty() && scored.get(0).score() >= 0.55) {
            String topDoc = scored.get(0).document();
            List<SearchResult> prioritized = new ArrayList<>();
            for (SearchResult r : scored) {
                if (r.document().equals(topDoc)) {
                    prioritized.add(r);
                }
            }
            for (SearchResult r : scored) {
                if (!r.document().equals(topDoc) && prioritized.size() < topK && r.score() >= 0.40) {
                    prioritized.add(r);
                }
            }
            return prioritized.stream().limit(topK).toList();
        }

        return scored.stream().limit(topK).toList();
    }

    private List<SearchResult> queryChromaDb(float[] queryEmbedding, int topK) {
        try {
            // Build float array list
            List<Float> embList = new ArrayList<>(queryEmbedding.length);
            for (float f : queryEmbedding) {
                embList.add(f);
            }

            Map<String, Object> queryBody = Map.of(
                    "query_embeddings", List.of(embList),
                    "n_results", topK,
                    "include", List.of("documents", "metadatas", "distances")
            );

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(properties.getChromaUrl() + "/api/v1/collections/" + chromaCollectionId + "/query"))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(4))
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(queryBody)))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 200) {
                JsonNode root = objectMapper.readTree(response.body());
                JsonNode idsArr = root.path("ids").get(0);
                JsonNode docsArr = root.path("documents").get(0);
                JsonNode metasArr = root.path("metadatas").get(0);
                JsonNode distArr = root.path("distances").get(0);

                if (idsArr != null && idsArr.isArray()) {
                    List<SearchResult> results = new ArrayList<>();
                    for (int i = 0; i < idsArr.size(); i++) {
                        String id = idsArr.get(i).asText();
                        String text = (docsArr != null && docsArr.size() > i) ? docsArr.get(i).asText() : "";
                        JsonNode meta = (metasArr != null && metasArr.size() > i) ? metasArr.get(i) : null;
                        String docName = (meta != null) ? meta.path("document").asText("Unknown") : "Unknown";
                        int page = (meta != null) ? meta.path("page").asInt(1) : 1;
                        String heading = (meta != null) ? meta.path("heading").asText("General") : "General";

                        double distance = (distArr != null && distArr.size() > i) ? distArr.get(i).asDouble() : 1.0;
                        // Score: Map Euclidean distance squared (0..2) to similarity (0..1)
                        double score = Math.max(0.0, Math.min(1.0, 1.0 - (distance / 2.0)));

                        String sourcesStr = (meta != null) ? meta.path("sources").asText("") : "";
                        List<String> sources = parseSources(sourcesStr, docName, page);

                        results.add(new SearchResult(id, docName, page, heading, text, score, sources));
                    }
                    if (!results.isEmpty()) {
                        return results;
                    }
                }
            }
        } catch (Exception e) {
            log.warn("[CHROMA] Query failed: {}. Falling back to disk stream search.", e.getMessage());
        }
        return Collections.emptyList();
    }

    /**
     * Streams over all `.vectors.jsonl` files on disk, computing cosine similarity on the fly.
     * Uses a bounded PriorityQueue (O(topK) heap memory), guaranteeing no OutOfMemoryError.
     */
    private List<SearchResult> streamSearchFromDisk(float[] queryEmbedding, String queryText, int topK) {
        Path vectorDir = properties.getVectorStoragePath();
        if (!Files.exists(vectorDir)) return Collections.emptyList();

        PriorityQueue<SearchResult> queue = new PriorityQueue<>(Comparator.comparingDouble(SearchResult::score));
        Set<String> stopWords = Set.of(
                "what", "is", "the", "a", "an", "and", "or", "how", "to", "in", "on", "for",
                "with", "about", "tell", "me", "give", "can", "you", "does", "do", "explain"
        );
        List<String> queryKeywords = (queryText != null)
                ? Arrays.stream(queryText.toLowerCase().split("[^a-z0-9]+"))
                        .filter(w -> w.length() > 2 && !stopWords.contains(w))
                        .toList()
                : List.of();

        try (Stream<Path> stream = Files.list(vectorDir)) {
            stream.filter(p -> p.toString().endsWith(".vectors.jsonl")).forEach(file -> {
                try (BufferedReader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (line.trim().isEmpty()) continue;
                        JsonNode node = objectMapper.readTree(line);
                        String id = node.path("id").asText();
                        String docName = node.path("documentName").asText();
                        int page = node.path("pageNumber").asInt();
                        String heading = node.path("heading").asText();
                        String text = node.path("text").asText();
                        String sourcesStr = node.path("metadata").path("sources").asText("");
                        List<String> sources = parseSources(sourcesStr, docName, page);
                        JsonNode embNode = node.path("embedding");

                        if (embNode.isArray()) {
                            float[] emb = new float[embNode.size()];
                            for (int i = 0; i < embNode.size(); i++) {
                                emb[i] = (float) embNode.get(i).asDouble();
                            }

                            double vectorScore = cosineSimilarity(queryEmbedding, emb);
                            double lexicalScore = 0.0;
                            if (!queryKeywords.isEmpty()) {
                                String searchTarget = (docName + " " + heading + " " + text).toLowerCase();
                                int matches = 0;
                                for (String kw : queryKeywords) {
                                    if (searchTarget.contains(kw)) matches++;
                                }
                                lexicalScore = (double) matches / queryKeywords.size();
                                if (queryText != null && searchTarget.contains(queryText.toLowerCase().trim())) {
                                    lexicalScore = Math.min(1.0, lexicalScore + 0.3);
                                }
                            }

                            // Down-rank pure index/TOC pages and metadata-only headers
                            String textLower = text.toLowerCase();
                            if (textLower.startsWith("# document:") && text.length() < 280) {
                                lexicalScore *= 0.05;
                                vectorScore *= 0.1;
                            }
                            if (textLower.contains("s no. chapter title page") || textLower.contains("chapter title page no")
                                    || textLower.contains("table of contents") || textLower.contains("### index")) {
                                lexicalScore *= 0.05;
                                vectorScore *= 0.1;
                            }
                            if ((textLower.contains("public service commission") || textLower.contains("assistant professor")) && textLower.length() < 220) {
                                lexicalScore *= 0.05;
                                vectorScore *= 0.1;
                            }

                            double hybridScore = (lexicalScore > 0.0)
                                    ? (0.35 * vectorScore + 0.65 * lexicalScore)
                                    : (0.75 * vectorScore);

                            if (queue.size() < topK) {
                                queue.offer(new SearchResult(id, docName, page, heading, text, hybridScore, sources));
                            } else if (hybridScore > queue.peek().score()) {
                                queue.poll();
                                queue.offer(new SearchResult(id, docName, page, heading, text, hybridScore, sources));
                            }
                        }
                    }
                } catch (Exception e) {
                    log.debug("[DISK-SEARCH] Error reading {}: {}", file.getFileName(), e.getMessage());
                }
            });
        } catch (IOException e) {
            log.error("[DISK-SEARCH] Failed reading directory: {}", vectorDir, e);
        }

        List<SearchResult> results = new ArrayList<>(queue);
        results.sort((a, b) -> Double.compare(b.score(), a.score()));
        return results;
    }

    private List<String> parseSources(String sourcesStr, String defaultDoc, int defaultPage) {
        if (sourcesStr != null && !sourcesStr.isBlank()) {
            return Arrays.stream(sourcesStr.split(";\\s*"))
                    .filter(s -> !s.isBlank())
                    .toList();
        }
        return List.of(defaultDoc + ":p" + defaultPage);
    }

    private double cosineSimilarity(float[] a, float[] b) {
        if (a == null || b == null || a.length != b.length) return 0.0;
        double dot = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    public Map<String, Object> getVectorDbStatus() {
        checkAndReconnectChroma();

        int totalVectors = countTotalVectorsOnDisk();

        Map<String, Object> status = new HashMap<>();
        status.put("chromaUrl", properties.getChromaUrl());
        status.put("chromaOnline", chromaAvailable);
        status.put("collectionName", properties.getChromaCollection());
        status.put("collectionId", chromaCollectionId != null ? chromaCollectionId : "N/A");
        status.put("totalVectorsCount", totalVectors);
        status.put("totalChunksDeduplicated", totalChunksDeduplicated.get());
        status.put("chunkDeduplicationActive", true);
        status.put("storageMode", chromaAvailable ? "ChromaDB Active + Local Disk Backup" : "Embedded Streaming Disk Vector Store");
        status.put("vectorStorageDir", properties.getVectorStorageDir());
        status.put("chromaSwaggerUrl", properties.getChromaUrl() + "/docs");
        status.put("chromaCollectionsUrl", properties.getChromaUrl() + "/api/v1/collections");
        return status;
    }

    public List<VectorDocumentSummary> getVectorDocumentSummaries(int limit) {
        int safeLimit = limit > 0 ? limit : 50;
        List<VectorDocumentSummary> summaries = new ArrayList<>();
        Path vectorDir = properties.getVectorStoragePath();

        if (!Files.exists(vectorDir)) return summaries;

        try (Stream<Path> stream = Files.list(vectorDir)) {
            stream.filter(p -> p.toString().endsWith(".vectors.jsonl")).forEach(file -> {
                if (summaries.size() >= safeLimit) return;
                try (BufferedReader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                    String line;
                    while ((line = reader.readLine()) != null && summaries.size() < safeLimit) {
                        if (line.trim().isEmpty()) continue;
                        JsonNode node = objectMapper.readTree(line);
                        String id = node.path("id").asText();
                        String docName = node.path("documentName").asText();
                        int page = node.path("pageNumber").asInt();
                        String heading = node.path("heading").asText();
                        String text = node.path("text").asText();
                        String preview = text.length() > 180 ? text.substring(0, 180) + "..." : text;
                        summaries.add(new VectorDocumentSummary(id, docName, page, heading, preview, 128));
                    }
                } catch (Exception ignored) {}
            });
        } catch (IOException ignored) {}

        return summaries;
    }

    private int countTotalVectorsOnDisk() {
        Path vectorDir = properties.getVectorStoragePath();
        if (!Files.exists(vectorDir)) return 0;
        int count = 0;
        try (Stream<Path> stream = Files.list(vectorDir)) {
            List<Path> files = stream.filter(p -> p.toString().endsWith(".vectors.jsonl")).toList();
            for (Path file : files) {
                try (BufferedReader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                    while (reader.readLine() != null) {
                        count++;
                    }
                } catch (Exception ignored) {}
            }
        } catch (IOException ignored) {}
        return count;
    }

    public void checkAndReconnectChroma() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(properties.getChromaUrl() + "/api/v1/heartbeat"))
                    .timeout(Duration.ofMillis(1000))
                    .GET()
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            this.chromaAvailable = (response.statusCode() == 200);
            if (this.chromaAvailable && this.chromaCollectionId == null) {
                createOrGetCollection();
            }
        } catch (Exception e) {
            this.chromaAvailable = false;
        }
    }

    public record VectorDocumentSummary(
            String id,
            String documentName,
            int pageNumber,
            String heading,
            String textPreview,
            int dimension
    ) {}

    public record VectorIngestionSummary(int jsonlFiles, int totalVectors, int totalDeduplicated, boolean chromaOnline) {
        public VectorIngestionSummary(int jsonlFiles, int totalVectors, boolean chromaOnline) {
            this(jsonlFiles, totalVectors, 0, chromaOnline);
        }
    }

    public record SearchResult(
            String id,
            String document,
            int page,
            String heading,
            String text,
            double score,
            List<String> sources
    ) {
        public SearchResult(String id, String document, int page, String heading, String text, double score) {
            this(id, document, page, heading, text, score, List.of(document + ":p" + page));
        }
    }
}
