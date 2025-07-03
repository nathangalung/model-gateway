FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy project files and install dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --locked || uv sync

# Copy application code
COPY app/ app/
COPY data/ data/
COPY tests/ tests/

# Set environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
EXPOSE 8000

# Default command
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]