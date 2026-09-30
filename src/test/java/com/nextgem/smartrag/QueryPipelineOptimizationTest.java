package com.nextgem.smartrag;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nextgem.smartrag.query.*;
import com.nextgem.smartrag.service.RagGenerationService;
import com.nextgem.smartrag.vectorstore.ChromaVectorStoreService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

public class QueryPipelineOptimizationTest {

    private QueryRelevanceFilter relevanceFilter;
    private QueryMetricsTracker metricsTracker;
    private QueryCircuitBreaker circuitBreaker;
    private QueryConcurrencyGuard concurrencyGuard;
    private RagCacheService cacheService;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper();
        relevanceFilter = new QueryRelevanceFilter();
        metricsTracker = new QueryMetricsTracker();
        circuitBreaker = new QueryCircuitBreaker();
        concurrencyGuard = new QueryConcurrencyGuard(4, 10, 2000, metricsTracker);
        cacheService = new RagCacheService(objectMapper, metricsTracker, null);
    }

    @Test
    void testRelevanceFilterDiscardsBoilerplatePreambles() {
        String preamble = "# Document: Test_Doc\n> Source: /app/data/raw_pdfs/Test_Doc.pdf\n> Total Pages: 5";
        assertTrue(relevanceFilter.isBoilerplateOrPreamble(preamble), "Preamble metadata should be identified as boilerplate");

        String validContent = "Frontend optimization requires reducing bundle sizes, deferring non-critical CSS, and using modern image formats like WebP.";
        assertFalse(relevanceFilter.isBoilerplateOrPreamble(validContent), "Valid prose must not be flagged as boilerplate");
    }

    @Test
    void testRelevanceFilterCleansPipelineMetadataLines() {
        String dirtyChunk = "# Document: Frontend_Guide\n> Source: /data/test.pdf\n> Total Pages: 2\nFrontend optimization improves user experience.";
        String cleaned = relevanceFilter.cleanChunkContent(dirtyChunk);

        assertFalse(cleaned.contains("# Document:"), "Cleaned text should not contain Document header");
        assertFalse(cleaned.contains("> Source:"), "Cleaned text should not contain Source line");
        assertTrue(cleaned.contains("Frontend optimization improves user experience."), "Core text must be preserved");
    }

    @Test
    void testExtractiveAnswerSynthesizesCleanFindings() {
        ChromaVectorStoreService.SearchResult res = new ChromaVectorStoreService.SearchResult(
                "chk_1",
                "Frontend_Guide",
                1,
                "Performance",
                "Frontend optimization reduces page load times. Lazy loading images saves network bandwidth.",
                0.85,
                List.of("Frontend_Guide:p1")
        );

        String answer = relevanceFilter.synthesizeExtractiveAnswer("frontend optimization", List.of(res));
        assertFalse(answer.contains("Page 1"), "Answer text must not contain page numbers");
        assertTrue(answer.contains("Frontend optimization reduces page load times."));
    }

    @Test
    void testConversationalQueryHandling() {
        assertTrue(relevanceFilter.isConversationalQuery("hi"));
        assertTrue(relevanceFilter.isConversationalQuery("Hello!"));
        assertTrue(relevanceFilter.isConversationalQuery("who are you"));
        assertTrue(relevanceFilter.isConversationalQuery("thanks"));

        assertFalse(relevanceFilter.isConversationalQuery("what is memory threshold"));
        assertFalse(relevanceFilter.isConversationalQuery("explain chunk deduplication"));

        String greetingResponse = relevanceFilter.getConversationalResponse("hi");
        assertTrue(greetingResponse.contains("Hello!") || greetingResponse.contains("Assistant"));
    }

    @Test
    void testCircuitBreakerTripsOnFailureBurstAndRecovers() {
        assertEquals(QueryCircuitBreaker.State.CLOSED, circuitBreaker.getState());

        // Trip with failures
        for (int i = 0; i < 15; i++) {
            circuitBreaker.recordFailure();
        }

        assertEquals(QueryCircuitBreaker.State.OPEN, circuitBreaker.getState());
        assertFalse(circuitBreaker.allowExecution(), "Circuit breaker should deny execution when OPEN");
    }

    @Test
    void testRateLimiterPerIp() {
        String ip = "192.168.1.50";
        // Default capacity is 30 tokens
        for (int i = 0; i < 30; i++) {
            assertTrue(concurrencyGuard.checkRateLimit(ip), "Should allow up to burst capacity");
        }
        // 31st request exceeds burst
        assertFalse(concurrencyGuard.checkRateLimit(ip), "Should reject request when burst bucket is exhausted");
    }

    @Test
    void testSingleFlightCoalescesConcurrentDuplicateQueries() throws Exception {
        AtomicInteger computeCount = new AtomicInteger(0);

        CompletableFuture<RagGenerationService.RagAnswer> f1 = concurrencyGuard.executeCoalesced("test-query", () -> {
            computeCount.incrementAndGet();
            Thread.sleep(100);
            return new RagGenerationService.RagAnswer("test-query", "Answer 1", List.of(), 1);
        });

        CompletableFuture<RagGenerationService.RagAnswer> f2 = concurrencyGuard.executeCoalesced("test-query", () -> {
            computeCount.incrementAndGet();
            return new RagGenerationService.RagAnswer("test-query", "Answer 2", List.of(), 1);
        });

        RagGenerationService.RagAnswer r1 = f1.get();
        RagGenerationService.RagAnswer r2 = f2.get();

        assertEquals(r1.answer(), r2.answer());
        assertEquals(1, computeCount.get(), "Only 1 computation should run for identical in-flight queries");
    }

    @Test
    void testL1CacheStoresAndRetrievesAnswers() {
        RagGenerationService.RagAnswer answer = new RagGenerationService.RagAnswer("what is rag", "Retrieval Augmented Generation", List.of(), 1);
        cacheService.putAnswer("what is rag", 5, answer);

        RagGenerationService.RagAnswer retrieved = cacheService.getAnswer("what is rag", 5);
        assertNotNull(retrieved);
        assertEquals("Retrieval Augmented Generation", retrieved.answer());

        // Test embedding caching
        float[] sampleEmbedding = new float[]{0.1f, 0.2f, 0.3f};
        cacheService.putEmbedding("sample query", sampleEmbedding);
        float[] cachedEmb = cacheService.getEmbedding("sample query");
        assertNotNull(cachedEmb);
        assertEquals(0.2f, cachedEmb[1]);
    }

    @Test
    void testCleanAnswerFormattingStripsAsterisksPageNumbersAndChunkArtifacts() {
        RagGenerationService genService = new RagGenerationService(null, cacheService, relevanceFilter, concurrencyGuard, circuitBreaker, metricsTracker, objectMapper);

        String rawLlmResponse = "Source: Volume-1-History-Geography-Politics-and-Economy-of-Madhya-Pradesh (Page 8)\n" +
                "**Ashoka**: Ashoka was one of the greatest rulers. Kakaniya (Page 8).\n" +
                "[1] Chunk 1: Ashoka married Shridevi/Mahadevi of Besnagar (Vidisha) (p. 20).\n" +
                "• **Dynasties**: Sunga and Maurya dynasties ruled central India.\n" +
                "ToppersNotes / 9828-286-909 5";

        String cleaned = genService.cleanAnswerFormatting(rawLlmResponse);

        assertFalse(cleaned.contains("**"), "Must not contain markdown bold asterisks");
        assertFalse(cleaned.contains("* *"), "Must not contain isolated asterisks");
        assertFalse(cleaned.contains("*"), "Must not contain any asterisks");
        assertFalse(cleaned.contains("Page 8"), "Must not contain page references");
        assertFalse(cleaned.contains("p. 20"), "Must not contain page abbreviations");
        assertFalse(cleaned.contains("[1]"), "Must not contain bracketed chunk numbers");
        assertFalse(cleaned.contains("Chunk 1"), "Must not contain chunk identifiers");
        assertFalse(cleaned.contains("ToppersNotes"), "Must not contain publisher watermarks");
        assertFalse(cleaned.contains("9828-286-909"), "Must not contain phone numbers");
        assertTrue(cleaned.contains("Ashoka was one of the greatest rulers"));
        assertTrue(cleaned.contains("• Dynasties: Sunga and Maurya dynasties ruled central India."));
    }

    @Test
    void testSynthesizeExtractiveAnswerContainsNoAsterisksOrPageNumbers() {
        ChromaVectorStoreService.SearchResult res1 = new ChromaVectorStoreService.SearchResult(
                "chk_mp_1",
                "Volume-1-History-Geography-Politics-and-Economy-of-Madhya-Pradesh",
                8,
                "CHAPTER",
                "Ashoka was a prominent ruler. Ashoka married Shridevi of Besnagar. Sanchi Stupa was restored between 1912 and 1920 CE.",
                0.85,
                List.of("Volume-1:p8")
        );

        String answer = relevanceFilter.synthesizeExtractiveAnswer("tell me about madhyapradesh", List.of(res1));

        assertFalse(answer.contains("**"), "Extractive answer must not contain markdown bold asterisks");
        assertFalse(answer.contains("* *"), "Extractive answer must not contain asterisks");
        assertFalse(answer.contains("Page 8"), "Extractive answer must not contain page numbers");
        assertFalse(answer.contains("(Page 8)"), "Extractive answer must not contain bracketed page numbers");
        assertTrue(answer.contains("Madhya Pradesh is located in central India"), "Should include synthesized foundational overview");
        assertTrue(answer.contains("Ashoka married Shridevi of Besnagar") || answer.contains("Sanchi Stupa"));
    }

    @Test
    void testQueryNormalizationForMadhyaPradesh() {
        String normalized = relevanceFilter.normalizeAndExpandQuery("tell me about madhyapradesh");
        assertEquals("tell me about madhya pradesh", normalized);
    }
}
