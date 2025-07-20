# Multi-stage build for AIDA
FROM python:3.11-slim as builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Copy uv
COPY --from=builder /bin/uv /bin/uv

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN groupadd -r aida && useradd -r -g aida aida

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=aida:aida /app/.venv /app/.venv

# Copy application code
COPY --chown=aida:aida . .

# Install the application
RUN uv pip install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/tmp && \
    chown -R aida:aida /app

# Switch to non-root user
USER aida

# Expose port for web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["aida", "serve", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM builder as development

# Install development dependencies
RUN uv sync --frozen --no-install-project

# Copy application code
COPY . .

# Install the application in development mode
RUN uv pip install -e ".[dev]"

# Switch to non-root user
USER aida

# Default command for development
CMD ["aida", "interactive"]