# INDU - Lunar Visual Navigation MVP
#
# The reload watcher is scoped to source on purpose: watching the repo root
# makes uvicorn stat() ~21k files per tick (.venv + node_modules), which pins
# a core and eventually wedges the server.

VENV := backend/.venv/bin
PYTHONPATH := backend:src

.PHONY: help dev-backend dev-frontend test build data

help:
	@echo "INDU - available targets"
	@echo "  make dev-backend   Start the FastAPI backend on :8000 (scoped hot reload)"
	@echo "  make dev-frontend  Start the Vite dev server on :5173"
	@echo "  make test          Run the backend test suite"
	@echo "  make build         Type-check and build the frontend"
	@echo "  make data          Report what is present under data/raw (read-only)"

dev-backend:
	PYTHONPATH=$(PYTHONPATH) $(VENV)/uvicorn app.main:app \
		--port 8000 --app-dir backend/app/.. \
		--reload --reload-dir backend/app --reload-dir src

dev-frontend:
	cd frontend && npm run dev

test:
	PYTHONPATH=$(PYTHONPATH) $(VENV)/python -m pytest -q

build:
	cd frontend && npx tsc -b --noEmit && npm run build

# Report only. Writing anything requires an explicit flag on the script.
data:
	PYTHONPATH=src $(VENV)/python scripts/prepare_real_data.py
