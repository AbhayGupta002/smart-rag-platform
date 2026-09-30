package com.nextgem.smartrag.controller;

import com.nextgem.smartrag.chunking.RagChunkingService;
import com.nextgem.smartrag.config.RagPipelineProperties;
import com.nextgem.smartrag.model.DocumentJob;
import com.nextgem.smartrag.orchestrator.RagPipelineOrchestrator;
import com.nextgem.smartrag.parser.PdfParallelParserService;
import com.nextgem.smartrag.repository.DocumentRepository;
import com.nextgem.smartrag.service.*;
import com.nextgem.smartrag.vectorstore.ChromaVectorStoreService;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/rag")
public class RagIngestionController {

    private final RagPipelineOrchestrator orchestrator;
    private final PdfParallelParserService parserService;
    private final RagChunkingService chunkingService;
    private final ChromaVectorStoreService vectorStoreService;
    private final RagGenerationService generationService;
    private final DocumentRepository documentRepository;
    private final DynamicHardwareTuningService tuningService;
    private final OperatingSystemDetector osDetector;
    private final ResourceManager resourceManager;
    private final PerformanceController performanceController;
    private final CheckpointService checkpointService;
    private final RagPipelineProperties properties;

    @org.springframework.beans.factory.annotation.Autowired(required = false)
    private com.nextgem.smartrag.query.QueryConcurrencyGuard concurrencyGuard;

    @org.springframework.beans.factory.annotation.Autowired(required = false)
    private com.nextgem.smartrag.query.QueryMetricsTracker metricsTracker;

    @org.springframework.beans.factory.annotation.Autowired(required = false)
    private com.nextgem.smartrag.query.QueryCircuitBreaker circuitBreaker;

    @org.springframework.beans.factory.annotation.Autowired(required = false)
    private com.nextgem.smartrag.query.RagCacheService cacheService;

    public RagIngestionController(
            RagPipelineOrchestrator orchestrator,
            PdfParallelParserService parserService,
            RagChunkingService chunkingService,
            ChromaVectorStoreService vectorStoreService,
            RagGenerationService generationService,
            DocumentRepository documentRepository,
            DynamicHardwareTuningService tuningService,
            OperatingSystemDetector osDetector,
            ResourceManager resourceManager,
            PerformanceController performanceController,
            CheckpointService checkpointService,
            RagPipelineProperties properties
    ) {
        this.orchestrator = orchestrator;
        this.parserService = parserService;
        this.chunkingService = chunkingService;
        this.vectorStoreService = vectorStoreService;
        this.generationService = generationService;
        this.documentRepository = documentRepository;
        this.tuningService = tuningService;
        this.osDetector = osDetector;
        this.resourceManager = resourceManager;
        this.performanceController = performanceController;
        this.checkpointService = checkpointService;
        this.properties = properties;
    }

    /**
     * Upload one or more PDF files (or an entire folder of PDFs) directly from the browser
     * and execute the zero-copy parallel RAG ingestion pipeline.
     */
    @PostMapping(value = "/upload-and-ingest", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<RagPipelineOrchestrator.PipelineExecutionReport> uploadAndIngest(
            @RequestParam("files") MultipartFile[] files
    ) throws IOException {
        Path rawPdfDir = properties.getRawPdfPath();
        Files.createDirectories(rawPdfDir);

        for (MultipartFile file : files) {
            if (file.isEmpty()) continue;
            String originalFilename = file.getOriginalFilename();
            if (originalFilename == null || originalFilename.isBlank()) continue;
            if (!originalFilename.toLowerCase().endsWith(".pdf")) continue;

            Path fileNameOnly = Paths.get(originalFilename).getFileName();
            Path destination = rawPdfDir.resolve(fileNameOnly.toString());
            try (java.io.InputStream in = file.getInputStream()) {
                Files.copy(in, destination, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        RagPipelineOrchestrator.PipelineExecutionReport report = orchestrator.runPipeline(rawPdfDir.toString());
        return ResponseEntity.ok(report);
    }

    /**
     * Ingests PDFs from a local folder path and runs the full parallel RAG pipeline.
     */
    @PostMapping("/ingest")
    public ResponseEntity<RagPipelineOrchestrator.PipelineExecutionReport> ingest(
            @RequestParam(required = false) String folderPath
    ) {
        RagPipelineOrchestrator.PipelineExecutionReport report = orchestrator.runPipeline(folderPath);
        return ResponseEntity.ok(report);
    }

    /**
     * Resumes an interrupted or partially-failed pipeline run from durable checkpoints.
     */
    @PostMapping("/pipeline/resume")
    public ResponseEntity<RagPipelineOrchestrator.PipelineExecutionReport> resumePipeline(
            @RequestParam(required = false) String folderPath
    ) {
        RagPipelineOrchestrator.PipelineExecutionReport report = orchestrator.resumePipeline(folderPath);
        return ResponseEntity.ok(report);
    }

    /**
     * Live Pipeline Status Snapshot for real-time dashboard progress polling.
     */
    @GetMapping("/pipeline-status")
    public ResponseEntity<RagPipelineOrchestrator.PipelineStatusSnapshot> getPipelineStatus() {
        return ResponseEntity.ok(orchestrator.getLiveStatus());
    }

    /**
     * Cross-platform System and Hardware Information (OS, CPU, RAM, GPU, Java).
     */
    @GetMapping("/system-info")
    public ResponseEntity<OperatingSystemDetector.SystemInfo> getSystemInfo() {
        return ResponseEntity.ok(osDetector.getSystemInfo());
    }

    /**
     * Override OS execution mode (AUTO, WINDOWS, MACOS, LINUX).
     */
    @PostMapping("/os-override")
    public ResponseEntity<OperatingSystemDetector.SystemInfo> setOsOverride(
            @RequestParam(defaultValue = "AUTO") String mode
    ) {
        if ("AUTO".equalsIgnoreCase(mode)) {
            osDetector.setOsOverride(null);
        } else {
            try {
                osDetector.setOsOverride(OperatingSystemDetector.OsType.valueOf(mode.toUpperCase()));
            } catch (IllegalArgumentException e) {
                return ResponseEntity.badRequest().build();
            }
        }
        return ResponseEntity.ok(osDetector.getSystemInfo());
    }

    /**
     * Adaptive execution plan calculated for the host environment.
     */
    @GetMapping("/execution-plan")
    public ResponseEntity<PerformanceController.ExecutionPlan> getExecutionPlan() {
        return ResponseEntity.ok(performanceController.getOptimalPlan());
    }

    /**
     * Resource Manager state check (NORMAL, WARNING, CRITICAL).
     */
    @GetMapping("/resource-state")
    public ResponseEntity<Map<String, Object>> getResourceState() {
        return ResponseEntity.ok(Map.of(
                "state", resourceManager.getCurrentState().name(),
                "usedHeapMb", resourceManager.getUsedHeapMb(),
                "maxHeapMb", resourceManager.getMaxHeapMb(),
                "usagePercent", Math.round(resourceManager.getHeapUsagePercentage() * 10.0) / 10.0,
                "throttles", resourceManager.getThrottleCount(),
                "criticalPauses", resourceManager.getCriticalPauseCount()
        ));
    }

    /**
     * Context-Grounded RAG QA Endpoint with Concurrency Guard, Rate Limiting & Backpressure.
     */
    @PostMapping("/ask")
    public ResponseEntity<?> askQuestion(
            @RequestParam String query,
            @RequestParam(defaultValue = "5") int topK,
            jakarta.servlet.http.HttpServletRequest request
    ) {
        String clientIp = getClientIp(request);
        if (concurrencyGuard != null && !concurrencyGuard.checkRateLimit(clientIp)) {
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                    .header("Retry-After", "1")
                    .body(Map.of(
                            "error", "Too Many Requests",
                            "message", "Rate limit exceeded. Please wait a moment before sending more queries.",
                            "status", 429
                    ));
        }

        try {
            RagGenerationService.RagAnswer answer = generationService.ask(query, topK);
            return ResponseEntity.ok(answer);
        } catch (com.nextgem.smartrag.query.QueryConcurrencyGuard.QueryBackpressureException e) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .header("Retry-After", "1")
                    .body(Map.of(
                            "error", "Service Overloaded",
                            "message", e.getMessage(),
                            "status", 503
                    ));
        }
    }

    /**
     * Real-time Server-Sent Events (SSE) Streaming QA Endpoint.
     */
    @GetMapping(value = "/ask/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public org.springframework.web.servlet.mvc.method.annotation.SseEmitter askStream(
            @RequestParam String query,
            @RequestParam(defaultValue = "5") int topK,
            jakarta.servlet.http.HttpServletRequest request
    ) {
        String clientIp = getClientIp(request);
        org.springframework.web.servlet.mvc.method.annotation.SseEmitter emitter = new org.springframework.web.servlet.mvc.method.annotation.SseEmitter(30_000L);

        if (concurrencyGuard != null && !concurrencyGuard.checkRateLimit(clientIp)) {
            try {
                emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event().name("error").data("Rate limit exceeded."));
                emitter.complete();
            } catch (IOException ignored) {}
            return emitter;
        }

        generationService.askStream(query, topK, emitter);
        return emitter;
    }

    /**
     * Get Current Active Free LLM Model Configuration & Status.
     */
    @GetMapping("/llm-config")
    public ResponseEntity<RagGenerationService.LlmStatus> getLlmConfig() {
        return ResponseEntity.ok(generationService.getLlmConfig());
    }

    /**
     * Update Free LLM Model Configuration at Runtime.
     */
    @PostMapping("/llm-config")
    public ResponseEntity<RagGenerationService.LlmStatus> updateLlmConfig(
            @RequestBody(required = false) Map<String, String> body,
            @RequestParam(required = false) String provider,
            @RequestParam(required = false) String apiKey,
            @RequestParam(required = false) String model,
            @RequestParam(required = false) String endpoint
    ) {
        Map<String, String> config = (body != null) ? new HashMap<>(body) : new HashMap<>();
        if (provider != null) config.put("provider", provider);
        if (apiKey != null) config.put("apiKey", apiKey);
        if (model != null) config.put("model", model);
        if (endpoint != null) config.put("endpoint", endpoint);

        generationService.updateLlmConfig(config);
        return ResponseEntity.ok(generationService.getLlmConfig());
    }

    /**
     * Live Connectivity Test for Free LLM Provider.
     */
    @PostMapping("/llm-test")
    public ResponseEntity<Map<String, Object>> testLlmConnection(
            @RequestBody(required = false) Map<String, String> body,
            @RequestParam(required = false) String provider,
            @RequestParam(required = false) String apiKey,
            @RequestParam(required = false) String model,
            @RequestParam(required = false) String endpoint
    ) {
        Map<String, String> config = (body != null) ? new HashMap<>(body) : new HashMap<>();
        if (provider != null) config.put("provider", provider);
        if (apiKey != null) config.put("apiKey", apiKey);
        if (model != null) config.put("model", model);
        if (endpoint != null) config.put("endpoint", endpoint);

        return ResponseEntity.ok(generationService.testLlmConnection(config));
    }

    /**
     * Real-time Query System Concurrency & Cache Metrics.
     */
    @GetMapping("/query-metrics")
    public ResponseEntity<?> getQueryMetrics() {
        if (metricsTracker == null) {
            return ResponseEntity.ok(Map.of("status", "Metrics tracker initializing"));
        }
        return ResponseEntity.ok(Map.of(
                "metrics", metricsTracker.getSnapshot(),
                "circuitBreakerState", circuitBreaker != null ? circuitBreaker.getState().name() : "CLOSED",
                "redisOnline", cacheService != null && cacheService.isRedisOperational(),
                "l1CacheSize", cacheService != null ? cacheService.getL1CacheSize() : 0,
                "availablePermits", concurrencyGuard != null ? concurrencyGuard.getAvailablePermits() : 128
        ));
    }

    /**
     * List all document ingestion job records from H2/database audit ledger.
     */
    @GetMapping("/jobs")
    public ResponseEntity<List<DocumentJob>> getJobs() {
        return ResponseEntity.ok(documentRepository.findAll());
    }

    /**
     * Stage 1 only: Zero-copy PDF parallel extraction to Markdown.
     */
    @PostMapping("/parse-only")
    public ResponseEntity<PdfParallelParserService.ParsingJobSummary> parseOnly(
            @RequestParam(required = false) String folderPath
    ) {
        return ResponseEntity.ok(parserService.parseAllPdfs(folderPath));
    }

    /**
     * Stage 2 only: Line-stream Markdown chunking with page markers into JSONL.
     */
    @PostMapping("/chunk-only")
    public ResponseEntity<RagChunkingService.ChunkingSummary> chunkOnly() {
        return ResponseEntity.ok(chunkingService.chunkAllStagedDocuments());
    }

    /**
     * Stage 3 only: Batch vector indexing from JSONL into ChromaDB.
     */
    @PostMapping("/vectorize-only")
    public ResponseEntity<ChromaVectorStoreService.VectorIngestionSummary> vectorizeOnly() {
        return ResponseEntity.ok(vectorStoreService.ingestAllChunks());
    }

    /**
     * Direct similarity search in vector index.
     */
    @GetMapping("/search")
    public ResponseEntity<List<ChromaVectorStoreService.SearchResult>> searchVectors(
            @RequestParam String query,
            @RequestParam(defaultValue = "5") int topK
    ) {
        return ResponseEntity.ok(vectorStoreService.search(query, topK));
    }

    /**
     * System metrics telemetry with real-time CPU and JVM Heap load.
     */
    @GetMapping("/system-metrics")
    public ResponseEntity<Map<String, Object>> systemMetrics() {
        Runtime runtime = Runtime.getRuntime();
        double cpuPercent = 0.0;
        try {
            java.lang.management.OperatingSystemMXBean osBean = java.lang.management.ManagementFactory.getOperatingSystemMXBean();
            if (osBean instanceof com.sun.management.OperatingSystemMXBean sunBean) {
                double load = sunBean.getCpuLoad();
                if (load >= 0.0) {
                    cpuPercent = Math.round(load * 1000.0) / 10.0;
                } else {
                    double procLoad = sunBean.getProcessCpuLoad();
                    if (procLoad >= 0.0) {
                        cpuPercent = Math.round(procLoad * 1000.0) / 10.0;
                    }
                }
            }
        } catch (Throwable ignored) {}

        Map<String, Object> metrics = new HashMap<>();
        metrics.put("availableProcessors", runtime.availableProcessors());
        metrics.put("freeMemoryMb", runtime.freeMemory() / (1024 * 1024));
        metrics.put("totalMemoryMb", runtime.totalMemory() / (1024 * 1024));
        metrics.put("maxMemoryMb", runtime.maxMemory() / (1024 * 1024));
        metrics.put("usedMemoryMb", (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024));
        metrics.put("cpuLoadPercent", cpuPercent);
        metrics.put("resourceState", resourceManager.getCurrentState().name());

        if (metricsTracker != null) {
            metrics.put("queryMetrics", metricsTracker.getSnapshot());
        }
        if (circuitBreaker != null) {
            metrics.put("circuitBreaker", circuitBreaker.getState().name());
        }
        if (cacheService != null) {
            metrics.put("redisOnline", cacheService.isRedisOperational());
            metrics.put("l1CacheEntries", cacheService.getL1CacheSize());
        }
        return ResponseEntity.ok(metrics);
    }

    private String getClientIp(jakarta.servlet.http.HttpServletRequest request) {
        if (request == null) return "127.0.0.1";
        String ip = request.getHeader("X-Forwarded-For");
        if (ip == null || ip.isBlank() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("X-Real-IP");
        }
        if (ip == null || ip.isBlank() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getRemoteAddr();
        }
        if (ip != null && ip.contains(",")) {
            ip = ip.split(",")[0].trim();
        }
        return ip != null ? ip : "127.0.0.1";
    }

    /**
     * Manual Garbage Collection sweep.
     */
    @PostMapping("/gc")
    public ResponseEntity<Map<String, Object>> triggerGc() {
        resourceManager.forceReclaim();
        return systemMetrics();
    }

    /**
     * Health status of Vector DB engine.
     */
    @GetMapping("/vector-db/status")
    public ResponseEntity<Map<String, Object>> vectorDbStatus() {
        return ResponseEntity.ok(vectorStoreService.getVectorDbStatus());
    }

    /**
     * Browse Stored Vector Documents.
     */
    @GetMapping("/vector-db/documents")
    public ResponseEntity<List<ChromaVectorStoreService.VectorDocumentSummary>> vectorDbDocuments(
            @RequestParam(defaultValue = "50") int limit
    ) {
        return ResponseEntity.ok(vectorStoreService.getVectorDocumentSummaries(limit));
    }

    /**
     * Dynamic Hardware Tuning Profile.
     */
    @GetMapping("/hardware-tuning")
    public ResponseEntity<DynamicHardwareTuningService.HardwareTuningProfile> getHardwareTuning() {
        return ResponseEntity.ok(tuningService.getCurrentProfile());
    }

    /**
     * Update Dynamic Hardware Tuning Profile.
     */
    @PostMapping("/hardware-tuning")
    public ResponseEntity<DynamicHardwareTuningService.HardwareTuningProfile> updateHardwareTuning(
            @RequestParam(defaultValue = "8") int ramCapacityGb,
            @RequestParam(defaultValue = "8") int allocatedCores
    ) {
        DynamicHardwareTuningService.HardwareTuningProfile updated = tuningService.tuneHardware(ramCapacityGb, allocatedCores);
        return ResponseEntity.ok(updated);
    }

    /**
     * Non-destructive inspection of a PDF file's properties and recommended page batch size.
     */
    @PostMapping("/inspect-file")
    public ResponseEntity<Map<String, Object>> inspectFile(
            @RequestParam(required = false) String filePath
    ) {
        Path targetPath = (filePath != null && !filePath.isBlank())
                ? Paths.get(filePath)
                : properties.getRawPdfPath();

        long sizeBytes = 0;
        try {
            if (Files.isRegularFile(targetPath)) {
                sizeBytes = Files.size(targetPath);
            }
        } catch (Exception ignored) {}

        return ResponseEntity.ok(Map.of(
                "filePath", targetPath.toString(),
                "sizeBytes", sizeBytes,
                "sizeMb", Math.round((sizeBytes / (1024.0 * 1024.0)) * 10.0) / 10.0,
                "recommendedPageBatchSize", properties.getPageBatchSize(),
                "strategy", sizeBytes > 50_000_000 ? "PAGE_BATCH_STREAMING" : "STANDARD_STREAMING"
        ));
    }

    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<Map<String, String>> handleIllegalState(IllegalStateException e) {
        return ResponseEntity.status(HttpStatus.CONFLICT).body(Map.of(
                "error", "Conflict",
                "message", e.getMessage()
        ));
    }
}
