# Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PORT=8080 \
    OPENAI_API_KEY="put your key or any keys, xai, deepseek, etc" \
    WORKSPACE_DIR="agent_workspace"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /app/venv

# Copy requirements and install dependencies
COPY api/requirements.txt .
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./api ./api

# Expose the port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start command
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8080"]
