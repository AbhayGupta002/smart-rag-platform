package com.nextgem.smartrag;

import com.nextgem.smartrag.model.DocumentJob;
import com.nextgem.smartrag.orchestrator.RagPipelineOrchestrator;
import com.nextgem.smartrag.repository.DocumentRepository;
import com.nextgem.smartrag.service.RagGenerationService;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.Comparator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
public class EndToEndRagPipelineTest {

    @Autowired
    private RagPipelineOrchestrator orchestrator;

    @Autowired
    private RagGenerationService generationService;

    @Autowired
    private DocumentRepository jobRepository;

    private Path testDir;

    @BeforeEach
    void setUp() throws IOException {
        testDir = Files.createTempDirectory("rag_test_raw_");
        jobRepository.deleteAll();
        Files.deleteIfExists(Paths.get("data/staging_md/legal_order_1998.md"));
        Files.deleteIfExists(Paths.get("data/chunked_jsonl/legal_order_1998.jsonl"));
        Files.deleteIfExists(Paths.get("data/vector_storage/legal_order_1998.vectors.jsonl"));
        Files.deleteIfExists(Paths.get("data/staging_md/valid_doc.md"));
        Files.deleteIfExists(Paths.get("data/chunked_jsonl/valid_doc.jsonl"));
        Files.deleteIfExists(Paths.get("data/vector_storage/valid_doc.vectors.jsonl"));
    }

    @AfterEach
    void tearDown() throws IOException {
        if (testDir != null && Files.exists(testDir)) {
            Files.walk(testDir)
                    .sorted(Comparator.reverseOrder())
                    .map(Path::toFile)
                    .forEach(File::delete);
        }
    }

    @Test
    void testEndToEndPipelineExecutionAndRetrieval() throws IOException {
        // 1. Generate Synthetic PDF with legal clauses & table
        Path samplePdf = testDir.resolve("legal_order_1998.pdf");
        createSyntheticLegalPdf(samplePdf);

        assertTrue(Files.exists(samplePdf), "Synthetic PDF should exist");

        // 2. Execute End-to-End RAG Ingestion Pipeline
        RagPipelineOrchestrator.PipelineExecutionReport report = orchestrator.runPipeline(testDir.toString());

        assertNotNull(report, "Pipeline report must not be null");
        assertEquals(1, report.parsing().discovered(), "Should discover 1 PDF");
        assertEquals(1, report.parsing().processed(), "Should process 1 PDF successfully");
        assertEquals(0, report.parsing().failed(), "Should have 0 failures");
        assertEquals(2, report.parsing().totalPages(), "Should extract 2 pages");
        assertTrue(report.chunking().totalChunks() >= 1, "Should generate at least 1 RAG chunk");

        // 3. Verify Database Audit Record
        List<DocumentJob> jobs = jobRepository.findAll();
        assertFalse(jobs.isEmpty(), "DocumentJob repository must contain job records");
        DocumentJob job = jobs.stream()
                .filter(j -> j.getFilename().equals("legal_order_1998.pdf"))
                .findFirst()
                .orElse(null);
        assertNotNull(job, "Job record for legal_order_1998.pdf must exist");
        assertEquals("SUCCESS", job.getStatus());
        assertEquals(2, job.getTotalPages());

        // 4. Verify RAG Question-Answering Retrieval
        RagGenerationService.RagAnswer answer = generationService.ask("leave policy entitlement", 10);
        assertNotNull(answer);
        assertNotNull(answer.answer());
        assertFalse(answer.citations().isEmpty(), "Must return at least 1 citation");

        // The test document must appear somewhere in the retrieved citations
        boolean foundTestDoc = answer.citations().stream()
                .anyMatch(c -> "legal_order_1998".equals(c.document()));
        assertTrue(foundTestDoc, "Citations must include the test document 'legal_order_1998'");

        RagGenerationService.Citation testCitation = answer.citations().stream()
                .filter(c -> "legal_order_1998".equals(c.document()))
                .findFirst()
                .orElseThrow();
        assertEquals(1, testCitation.page(), "Leave policy should be located on page 1");

        // 5. Test Deduplication on repeat run
        RagPipelineOrchestrator.PipelineExecutionReport repeatReport = orchestrator.runPipeline(testDir.toString());
        assertEquals(1, repeatReport.parsing().processed(), "File should be recognized and skipped via checksum cache");
    }

    @Test
    void testFailureIsolationWithCorruptedPdf() throws IOException {
        // Create 1 valid PDF and 1 corrupted dummy PDF in the same directory
        Path validPdf = testDir.resolve("valid_doc.pdf");
        createSyntheticLegalPdf(validPdf);

        Path corruptPdf = testDir.resolve("corrupted_payload.pdf");
        Files.writeString(corruptPdf, "%PDF-1.4 THIS IS NOT A VALID PDF BINARY STREAM TRUNCATED");

        RagPipelineOrchestrator.PipelineExecutionReport report = orchestrator.runPipeline(testDir.toString());

        assertNotNull(report);
        assertEquals(2, report.parsing().discovered(), "Must discover both PDFs");
        assertEquals(1, report.parsing().processed(), "Valid PDF must succeed");
        assertEquals(1, report.parsing().failed(), "Corrupt PDF must fail gracefully");

        // Checkpoint verification in DB
        List<DocumentJob> jobs = jobRepository.findAll();
        DocumentJob failedJob = jobs.stream()
                .filter(j -> "corrupted_payload.pdf".equals(j.getFilename()))
                .findFirst()
                .orElse(null);
        assertNotNull(failedJob, "Corrupted PDF must have a job record in DB");
        assertEquals("FAILED", failedJob.getStatus(), "Status must be FAILED");
        assertNotNull(failedJob.getErrorMessage(), "Must contain descriptive error message");
    }

    private void createSyntheticLegalPdf(Path outputPath) throws IOException {
        try (PDDocument doc = new PDDocument()) {
            PDType1Font font = new PDType1Font(Standard14Fonts.FontName.HELVETICA);
            PDType1Font fontBold = new PDType1Font(Standard14Fonts.FontName.HELVETICA_BOLD);

            // Page 1: Title, Clause, and Tabular data
            PDPage page1 = new PDPage();
            doc.addPage(page1);

            try (PDPageContentStream cs = new PDPageContentStream(doc, page1)) {
                cs.beginText();
                cs.setFont(fontBold, 14);
                cs.newLineAtOffset(50, 720);
                cs.showText("Order No 1998 GAD Circular");

                cs.newLineAtOffset(0, -30);
                cs.setFont(font, 11);
                cs.showText("1.1 Leave Policy Entitlements");

                cs.newLineAtOffset(0, -20);
                cs.showText("All permanent employees are entitled to thirty days of earned leave annually.");

                cs.newLineAtOffset(0, -30);
                cs.showText("Grade    LeaveDays    Allowance");

                cs.newLineAtOffset(0, -20);
                cs.showText("Grade-A    30    5000");

                cs.newLineAtOffset(0, -20);
                cs.showText("Grade-B    25    3500");

                cs.endText();
            }

            // Page 2: Clause and bullet points
            PDPage page2 = new PDPage();
            doc.addPage(page2);

            try (PDPageContentStream cs = new PDPageContentStream(doc, page2)) {
                cs.beginText();
                cs.setFont(fontBold, 13);
                cs.newLineAtOffset(50, 720);
                cs.showText("2.1 Remote Working Guidelines");

                cs.newLineAtOffset(0, -25);
                cs.setFont(font, 11);
                cs.showText("- Employees must obtain prior written approval from head of department.");

                cs.newLineAtOffset(0, -20);
                cs.showText("- VPN access is mandatory for all internal database operations.");

                cs.endText();
            }

            doc.save(outputPath.toFile());
        }
    }
}
