# Multi-stage Dockerfile for EHRX Gemini Hackathon Project
# This Dockerfile includes all dependencies from requirements.txt, setup.py,
# and system-level dependencies required by the application

# ========================================================================
# Stage 1: Builder - Install Python dependencies
# ========================================================================
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
# - gcc, g++: Required for compiling Python C extensions
# - poppler-utils: Required for pdf2image (PDF processing)
# - tesseract-ocr: Required for pytesseract (OCR processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt setup.py ./

# Install Python dependencies
# Using pip with --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install the package itself
COPY ehrx/ ./ehrx/
RUN pip install --no-cache-dir -e .

# ========================================================================
# Stage 2: Runtime - Create minimal runtime image
# ========================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Set default GCP location
    GCP_LOCATION=us-central1 \
    # Tesseract config
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Install runtime system dependencies
# - poppler-utils: PDF processing (pdf2image)
# - tesseract-ocr: OCR text extraction
# - libgl1, libglib2.0-0: Required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 ehrx && \
    mkdir -p /app /data /output && \
    chown -R ehrx:ehrx /app /data /output

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=ehrx:ehrx ehrx/ ./ehrx/
COPY --chown=ehrx:ehrx configs/ ./configs/

# Copy sample documents directory
# Note: Ensure SampleEHR_docs/ exists in the build context (not excluded by .dockerignore)
COPY --chown=ehrx:ehrx SampleEHR_docs/ ./SampleEHR_docs/

# Copy precomputed samples (if present) into image
COPY --chown=ehrx:ehrx precomputed_samples/ ./precomputed_samples/

# Copy precomputed answers (if present) into image
COPY --chown=ehrx:ehrx precomputed_answers/ ./precomputed_answers/

COPY --chown=ehrx:ehrx app.py run_mvp_pipeline.py test.py test_query_only.py ./
COPY --chown=ehrx:ehrx setup.py ./

# Switch to non-root user
USER ehrx

# Create necessary directories
RUN mkdir -p /app/output /app/logs

# Volume mounts for:
# - /data: Input PDF files
# - /output: Processing results
# - /credentials: Google Cloud credentials (optional)
VOLUME ["/data", "/output", "/credentials"]

# Default command: Start web server for Cloud Run
# Cloud Run sets PORT environment variable (default 8080)
CMD ["python", "app.py"]

# Health check (optional - useful for orchestration)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ehrx; import cv2; import pytesseract; print('OK')" || exit 1

# Labels for metadata
LABEL maintainer="PDF2EHR Team" \
      version="0.1.0" \
      description="EHR extraction tool for scanned PDFs using VLM and OCR" \
      python.version="3.11" \
      license="MIT"

# ========================================================================
# Build Instructions:
# ========================================================================
# docker build -t ehrx-pipeline:latest .
#
# Run Instructions:
# ========================================================================
# docker run -it --rm \
#   -v $(pwd)/SampleEHR_docs:/data:ro \
#   -v $(pwd)/output:/output \
#   -v /path/to/gcp-credentials.json:/credentials/gcp-key.json:ro \
#   -e GCP_PROJECT_ID=your-project-id \
#   -e GOOGLE_APPLICATION_CREDENTIALS=/credentials/gcp-key.json \
#   ehrx-pipeline:latest \
#   python run_mvp_pipeline.py
#
# Interactive Mode:
# ========================================================================
# docker run -it --rm \
#   -v $(pwd)/SampleEHR_docs:/data:ro \
#   -v $(pwd)/output:/output \
#   -v /path/to/gcp-credentials.json:/credentials/gcp-key.json:ro \
#   -e GCP_PROJECT_ID=your-project-id \
#   -e GOOGLE_APPLICATION_CREDENTIALS=/credentials/gcp-key.json \
#   ehrx-pipeline:latest \
#   /bin/bash

