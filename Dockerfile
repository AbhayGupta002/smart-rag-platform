# ==============================================================================
# Enterprise High-Throughput RAG Ingestion Pipeline - Dockerfile
# Optimized Production Containerization (Temurin 17 JRE)
# ==============================================================================
FROM eclipse-temurin:17-jre
WORKDIR /app

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create pipeline working directories
RUN mkdir -p /app/data/raw_pdfs \
             /app/data/staging_md \
             /app/data/chunked_jsonl \
             /app/data/vector_storage

# Copy pre-packaged enterprise JAR
COPY target/smart-rag-platform-1.0.0.jar /app/app.jar

# Container-optimized JVM memory management
ENV PORT=8080 \
    CHROMA_URL=http://chromadb:8000 \
    JAVA_OPTS="-XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0 -XX:+UseG1GC -Dorg.apache.pdfbox.rendering.UsePureJava=true -Dfile.encoding=UTF-8"

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8080/api/rag/system-metrics || exit 1

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar /app/app.jar --server.port=${PORT}"]
