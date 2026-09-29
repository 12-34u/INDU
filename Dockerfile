# Backend API image.
#
# Deliberately does NOT contain data/raw. That directory is 6.2 GB of mission
# archive, nothing the API serves reads it, and the demo and sector assets the
# app needs are three orders of magnitude smaller. The copy list below is
# explicit for that reason: a blanket `COPY . .` would pull the archive in.
# 3.13, not 3.11: numpy, scipy and rasterio are pinned to releases that
# declare requires-python >=3.12 and will not install on anything older.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=backend:src

WORKDIR /app

# Dependencies first, so application edits do not invalidate the layer.
# Everything is pinned; requirements-optional.txt (torch, lightglue) is
# deliberately not installed and the app runs without it.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Only what the API needs to run.
COPY backend/ ./backend/
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/demo/ ./data/demo/
COPY data/sector/ ./data/sector/
COPY data/reference_catalog/ ./data/reference_catalog/

# Written at runtime. Created here and owned by the unprivileged user, because
# the first request would otherwise try to create them as a user that cannot.
RUN mkdir -p data/cache/terrain data/working/registration outputs \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

# Nothing here needs root, and a compromised process should not have it.
USER appuser

EXPOSE 8000

# Render supplies its own health check; this one makes `docker run` locally
# report the same thing. Uses urllib rather than curl, which slim lacks.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/data/status',timeout=4)" || exit 1

# Shell form so ${PORT} expands. Render injects PORT and expects the process to
# bind it; a hardcoded port would be unreachable there.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
