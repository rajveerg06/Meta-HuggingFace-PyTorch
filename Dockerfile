FROM python:3.11-slim

# ── Build args ────────────────────────────────────────────────────────────────
ARG PORT=7860

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=${PORT}

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (security best practice) ────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# ── Python dependencies (separate layer for Docker cache efficiency) ───────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

# ── Switch to non-root user ───────────────────────────────────────────────────
USER appuser

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE ${PORT}

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
