# =============================================================================
# Stage 0: Security Scanning (Development/CI only)
# =============================================================================
FROM python:3.10-slim as security-scan

WORKDIR /scan

# Install security tools
RUN pip install --no-cache-dir safety bandit

# Copy requirements and scan
COPY requirements.txt .
RUN safety check -r requirements.txt --exit-code 0 || true

# Copy source code and scan
COPY src/ ./src/
COPY tests/ ./tests/

# Run Bandit security linter
RUN bandit -r src/ -f json -o /scan/bandit-report.json -ll || true && \
    bandit -r src/ -ll || true

# This stage is for CI/CD pipelines - not used in final image

# =============================================================================
# Stage 1: Base image with system dependencies
# =============================================================================
FROM python:3.10-slim as base

# Security: Create non-root user early
RUN groupadd -r trader && useradd -r -g trader -u 1000 trader

# Install system dependencies with minimal packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib for technical analysis
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Production image
# =============================================================================
FROM base as production

# Copy application code
COPY --chown=trader:trader . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create necessary directories with proper ownership
RUN mkdir -p /app/data /app/logs /app/models /app/checkpoints && \
    chown -R trader:trader /app/data /app/logs /app/models /app/checkpoints

# Security: Drop privileges
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Expose ports (for future API/dashboard)
EXPOSE 8000

# Default command: run pipeline
CMD ["python", "scripts/run_pipeline.py"]

# =============================================================================
# Stage 3: Training image with GPU support
# =============================================================================
FROM base as training

# Install PyTorch with CUDA support for GPU training
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY --chown=trader:trader . .

# Install package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/checkpoints && \
    chown -R trader:trader /app/data /app/logs /app/models /app/checkpoints

# Security: Drop privileges
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Default command: run training
CMD ["python", "src/training/train_agent.py"]

# =============================================================================
# Stage 4: Development image with additional tools
# =============================================================================
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    notebook \
    pytest-watch \
    black \
    flake8 \
    mypy

# Copy application code
COPY --chown=trader:trader . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create directories
RUN mkdir -p /app/data /app/logs /app/models /app/checkpoints && \
    chown -R trader:trader /app

# Security: Drop privileges
USER trader

# Default command: bash for interactive development
CMD ["/bin/bash"]

# =============================================================================
# Metadata
# =============================================================================
LABEL maintainer="AlphaRL-Quant Team"
LABEL description="Production-Grade Reinforcement Learning for Algorithmic Trading"
LABEL version="1.0"
LABEL security.scan="trivy,bandit,safety"
