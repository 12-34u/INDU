# Backend API image.
#
# Deliberately does NOT contain data/raw. That directory is 6.2 GB of mission
# archive, nothing the API serves reads it, and the demo and sector assets the
# app needs are three orders of magnitude smaller. The copy list below is
# explicit for that reason: a blanket `COPY . .` would pull the archive in.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=backend:src

WORKDIR /app

# Dependencies first, so application edits do not invalidate the layer.
# requirements-optional.txt (torch, lightglue) is deliberately not installed.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Only what the API needs to run.
COPY backend/ ./backend/
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/demo/ ./data/demo/
COPY data/sector/ ./data/sector/
COPY data/reference_catalog/ ./data/reference_catalog/

# Written at runtime; created here so the first request does not race to make them.
RUN mkdir -p data/cache/terrain data/working/registration outputs

EXPOSE 8000

# Shell form so ${PORT} expands. Render injects PORT and expects the process to
# bind it; a hardcoded port would be unreachable there.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
