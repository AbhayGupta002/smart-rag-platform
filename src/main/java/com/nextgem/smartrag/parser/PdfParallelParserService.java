package com.nextgem.smartrag.parser;

import com.nextgem.smartrag.config.RagPipelineProperties;
import com.nextgem.smartrag.model.DocumentJob;
import com.nextgem.smartrag.repository.DocumentRepository;
import com.nextgem.smartrag.service.CheckpointService;
import com.nextgem.smartrag.service.DynamicHardwareTuningService;
import com.nextgem.smartrag.service.ResourceManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.*;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Stream;

/**
 * Enterprise Parallel PDF Parser Service.
 * Implements bounded page-batch streaming, streaming SHA-256 fingerprinting,
 * resilient failure isolation, and durable checkpoint tracking.
 */
@Service
public class PdfParallelParserService {

    private static final Logger log = LoggerFactory.getLogger(PdfParallelParserService.class);

    private final RagPipelineProperties properties;
    private final ExecutorService executor;
    private final DynamicHardwareTuningService tuningService;
    private final DocumentRepository documentRepository;
    private final PageBatchProcessor pageBatchProcessor;
    private final CheckpointService checkpointService;
    private final ResourceManager resourceManager;

    // Real-time telemetry counters
    private final AtomicInteger totalFilesDiscovered = new AtomicInteger(0);
    private final AtomicInteger totalFilesProcessed = new AtomicInteger(0);
    private final AtomicInteger totalFilesFailed = new AtomicInteger(0);
    private final AtomicLong totalPagesExtracted = new AtomicLong(0);

    public PdfParallelParserService(
            RagPipelineProperties properties,
            @Qualifier("pipelineExecutor") ExecutorService executor,
            DynamicHardwareTuningService tuningService,
            DocumentRepository documentRepository,
            PageBatchProcessor pageBatchProcessor,
            CheckpointService checkpointService,
            ResourceManager resourceManager
    ) {
        this.properties = properties;
        this.executor = executor;
        this.tuningService = tuningService;
        this.documentRepository = documentRepository;
        this.pageBatchProcessor = pageBatchProcessor;
        this.checkpointService = checkpointService;
        this.resourceManager = resourceManager;
    }

    /**
     * Parses all PDFs in the given folder concurrently into Markdown files.
     * Uses lazy NIO.2 streaming and bounded worker permits.
     */
    public ParsingJobSummary parseAllPdfs(String inputFolderPath) {
        Path sourceDir = (inputFolderPath != null && !inputFolderPath.isBlank())
                ? Paths.get(inputFolderPath).toAbsolutePath()
                : properties.getRawPdfPath();

        Path stagingDir = properties.getStagingMdPath();

        try {
            Files.createDirectories(stagingDir);
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize staging directory: " + stagingDir, e);
        }

        String runId = checkpointService.getActiveRunId();
        log.info("[PIPELINE:{}] [STAGE:PARSE] Ingesting PDFs from [{}] to [{}] (Permits: {})",
                runId, sourceDir, stagingDir, properties.getMaxConcurrency());

        Instant startTime = Instant.now();
        resetMetrics();

        List<CompletableFuture<Void>> futures = new ArrayList<>();

        try (Stream<Path> pathStream = Files.walk(sourceDir)) {
            pathStream
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().toLowerCase().endsWith(".pdf"))
                    .forEach(pdfPath -> {
                        totalFilesDiscovered.incrementAndGet();

                        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                            try {
                                tuningService.acquirePermit();
                                resourceManager.checkAndThrottle();
                                processSinglePdf(pdfPath, stagingDir);
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                                log.warn("[PIPELINE:{}] Worker interrupted for: {}", runId, pdfPath.getFileName());
                            } finally {
                                tuningService.releasePermit();
                            }
                        }, executor);

                        futures.add(future);
                    });
        } catch (IOException e) {
            log.error("[PIPELINE:{}] Failed scanning directory [{}]: {}", runId, sourceDir, e.getMessage());
        }

        // Await batch completion
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();

        // Memory cleanup after partition 1
        resourceManager.forceReclaim();

        Duration elapsed = Duration.between(startTime, Instant.now());
        long elapsedMs = Math.max(1, elapsed.toMillis());
        double pagesPerSec = (totalPagesExtracted.get() * 1000.0) / elapsedMs;

        log.info("[PIPELINE:{}] [STAGE:PARSE] Completed in {}ms. Discovered: {}, Success: {}, Failed: {}, Pages: {}, Rate: {} p/s",
                runId, elapsedMs, totalFilesDiscovered.get(), totalFilesProcessed.get(), totalFilesFailed.get(),
                totalPagesExtracted.get(), String.format("%.2f", pagesPerSec));

        return new ParsingJobSummary(
                totalFilesDiscovered.get(),
                totalFilesProcessed.get(),
                totalFilesFailed.get(),
                totalPagesExtracted.get(),
                elapsedMs,
                pagesPerSec
        );
    }

    /**
     * Executes the strict "Read-Stream-Flush-Clear" lifecycle on a single PDF.
     * Guaranteed zero-copy read-only operations on source files.
     */
    public void processSinglePdf(Path pdfPath, Path stagingDir) {
        String filename = pdfPath.getFileName().toString();
        String baseName = getBaseName(filename);
        Path targetMdPath = stagingDir.resolve(baseName + ".md");

        // 1. Streaming SHA-256 (Bounded 8KB RAM, never loads entire file into memory)
        String checksum = computeStreamingChecksum(pdfPath);
        if (checksum == null) {
            log.error("[DOC:{}] [STAGE:PARSE] Failed computing checksum.", filename);
            totalFilesFailed.incrementAndGet();
            checkpointService.recordCheckpoint(filename, "unknown", 0, 0, DocumentJob.JobStatus.FAILED, "Checksum computation error");
            return;
        }

        // 2. Checkpoint Resume & Deduplication Guard
        if (checkpointService.isAlreadyCompleted(checksum) && Files.exists(targetMdPath)) {
            log.info("[DOC:{}] [STAGE:PARSE] Skipping already completed PDF (Checksum: {})", filename, checksum);
            totalFilesProcessed.incrementAndGet();
            checkpointService.recordCheckpoint(filename, checksum, 0, 0, DocumentJob.JobStatus.SKIPPED, null);
            return;
        }

        // 3. Mark as PROCESSING in checkpoint ledger
        checkpointService.recordCheckpoint(filename, checksum, 0, 0, DocumentJob.JobStatus.PROCESSING, null);

        // 4. Stream extraction in page batches (Read-only, preserves original PDF 100%)
        PageBatchProcessor.BatchProcessResult result = pageBatchProcessor.processDocumentInBatches(
                pdfPath,
                targetMdPath,
                properties.getPageBatchSize(),
                totalPagesExtracted
        );

        if (result.success()) {
            totalFilesProcessed.incrementAndGet();
            checkpointService.recordCheckpoint(
                    filename,
                    checksum,
                    result.totalPages(),
                    result.totalPages(),
                    DocumentJob.JobStatus.VALIDATED,
                    null
            );
            log.debug("[DOC:{}] [STAGE:PARSE] Successfully parsed {} pages in {}ms", filename, result.totalPages(), result.executionTimeMs());
        } else {
            totalFilesFailed.incrementAndGet();
            checkpointService.recordCheckpoint(
                    filename,
                    checksum,
                    0,
                    0,
                    DocumentJob.JobStatus.FAILED,
                    result.errorMessage()
            );
            log.warn("[DOC:{}] [STAGE:PARSE] Failed parsing PDF: {}", filename, result.errorMessage());
        }
    }

    /**
     * Computes SHA-256 fingerprint via streaming 8KB buffer.
     * Prevents OutOfMemoryError on 100MB-500MB+ PDF documents.
     */
    private String computeStreamingChecksum(Path path) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (InputStream is = new BufferedInputStream(Files.newInputStream(path), 8192);
                 DigestInputStream dis = new DigestInputStream(is, md)) {
                byte[] buffer = new byte[8192];
                while (dis.read(buffer) != -1) {
                    // Draining stream to update digest
                }
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (Exception e) {
            log.error("[CHECKSUM] Failed streaming checksum on [{}]: {}", path.getFileName(), e.getMessage());
            return null;
        }
    }

    private String getBaseName(String filename) {
        int dotIndex = filename.lastIndexOf('.');
        return (dotIndex == -1) ? filename : filename.substring(0, dotIndex);
    }

    private void resetMetrics() {
        totalFilesDiscovered.set(0);
        totalFilesProcessed.set(0);
        totalFilesFailed.set(0);
        totalPagesExtracted.set(0);
    }

    public record ParsingJobSummary(
            int discovered,
            int processed,
            int failed,
            long totalPages,
            long executionTimeMs,
            double pagesPerSecond
    ) {}
}
