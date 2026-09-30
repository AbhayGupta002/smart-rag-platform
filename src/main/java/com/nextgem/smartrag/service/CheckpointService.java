package com.nextgem.smartrag.service;

import com.nextgem.smartrag.model.DocumentJob;
import com.nextgem.smartrag.repository.DocumentRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Enterprise Checkpoint and Resume Service.
 * Tracks granular document and partition lifecycle states in durable storage
 * to guarantee zero silent data loss and crash recovery.
 */
@Service
public class CheckpointService {

    private static final Logger log = LoggerFactory.getLogger(CheckpointService.class);

    private final DocumentRepository documentRepository;
    private final AtomicReference<String> activePipelineRunId = new AtomicReference<>();

    public CheckpointService(DocumentRepository documentRepository) {
        this.documentRepository = documentRepository;
    }

    /**
     * Generates or retrieves the active pipeline run identifier.
     */
    public String startNewRun() {
        String runId = "run_" + System.currentTimeMillis() + "_" + UUID.randomUUID().toString().substring(0, 8);
        activePipelineRunId.set(runId);
        log.info("[CHECKPOINT] Initiated new pipeline run: {}", runId);
        return runId;
    }

    public String getActiveRunId() {
        String current = activePipelineRunId.get();
        if (current == null) {
            return startNewRun();
        }
        return current;
    }

    /**
     * Records or updates a document checkpoint.
     */
    @Transactional
    public DocumentJob recordCheckpoint(
            String filename,
            String checksum,
            int totalPages,
            int processedPages,
            DocumentJob.JobStatus status,
            String errorMessage
    ) {
        String safeChecksum = (checksum != null && !checksum.isBlank()) ? checksum : "unknown_" + filename;
        String runId = getActiveRunId();

        DocumentJob job = documentRepository.findByChecksum(safeChecksum)
                .orElseGet(() -> {
                    DocumentJob fresh = new DocumentJob();
                    fresh.setFilename(filename);
                    fresh.setChecksum(safeChecksum);
                    fresh.setPipelineRunId(runId);
                    return fresh;
                });

        job.setFilename(filename);
        job.setPipelineRunId(runId);
        job.setTotalPages(totalPages);
        job.setProcessedPages(processedPages);
        job.setJobStatus(status);
        job.setProcessedAt(LocalDateTime.now());

        if (errorMessage != null) {
            job.setErrorMessage(errorMessage);
            job.setFailureReason(errorMessage);
        }

        if (status == DocumentJob.JobStatus.COMPLETED) {
            job.setCompletedAt(LocalDateTime.now());
        }

        return documentRepository.save(job);
    }

    @Transactional
    public void markCompleted(String checksum, int totalPages, int chunks, long elapsedMs) {
        documentRepository.findByChecksum(checksum).ifPresent(job -> {
            job.setJobStatus(DocumentJob.JobStatus.COMPLETED);
            job.setTotalPages(totalPages);
            job.setProcessedPages(totalPages);
            job.setTotalChunks(chunks);
            job.setEmbeddedChunks(chunks);
            job.setExecutionTimeMs(elapsedMs);
            job.setCompletedAt(LocalDateTime.now());
            job.setErrorMessage(null);
            job.setFailureReason(null);
            documentRepository.save(job);
        });
    }

    @Transactional
    public void markFailed(String checksum, String errorMessage) {
        documentRepository.findByChecksum(checksum).ifPresent(job -> {
            job.setJobStatus(DocumentJob.JobStatus.FAILED);
            job.setErrorMessage(errorMessage);
            job.setFailureReason(errorMessage);
            job.setCompletedAt(LocalDateTime.now());
            documentRepository.save(job);
        });
    }

    /**
     * Checks if a document by checksum has already been successfully indexed.
     */
    public boolean isAlreadyCompleted(String checksum) {
        if (checksum == null) return false;
        return documentRepository.findByChecksum(checksum)
                .map(job -> job.getJobStatus() == DocumentJob.JobStatus.COMPLETED || "SUCCESS".equalsIgnoreCase(job.getStatus()))
                .orElse(false);
    }

    /**
     * Retrieves all incomplete or failed documents eligible for resumption.
     */
    public List<DocumentJob> findResumableJobs() {
        return documentRepository.findByJobStatusIn(List.of(
                DocumentJob.JobStatus.DISCOVERED,
                DocumentJob.JobStatus.VALIDATED,
                DocumentJob.JobStatus.PROCESSING,
                DocumentJob.JobStatus.CHUNKED,
                DocumentJob.JobStatus.EMBEDDED,
                DocumentJob.JobStatus.FAILED
        ));
    }

    public List<DocumentJob> getJobsByRunId(String runId) {
        return documentRepository.findByPipelineRunId(runId);
    }
}
