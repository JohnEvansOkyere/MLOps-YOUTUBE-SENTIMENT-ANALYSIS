# Dockerfile
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /app/reports /app/logs/inference && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/logs /app/reports# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /app/reports /app/logs/inference && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/logs /app/reports

# Copy requirements first (for caching)
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start application (NO --reload in production)
CMD ["uvicorn", "fastAPI_app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]