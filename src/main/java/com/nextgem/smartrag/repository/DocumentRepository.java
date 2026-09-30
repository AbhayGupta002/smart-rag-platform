package com.nextgem.smartrag.repository;

import com.nextgem.smartrag.model.DocumentJob;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface DocumentRepository extends JpaRepository<DocumentJob, Long> {
    Optional<DocumentJob> findByChecksum(String checksum);
    boolean existsByChecksumAndStatus(String checksum, String status);
    List<DocumentJob> findByPipelineRunId(String pipelineRunId);
    List<DocumentJob> findByJobStatusIn(List<DocumentJob.JobStatus> statuses);
    List<DocumentJob> findByStatusIn(List<String> statuses);
    Optional<DocumentJob> findTopByOrderByProcessedAtDesc();
}
