"""
Entry point that cannot be started with an unscoped reload watcher.

`uvicorn app.main:app --reload` is the obvious command to type and it wedges
this project. Unscoped, the watcher stats the whole repository on every tick -
.venv, node_modules, and several gigabytes of DEM and SPICE data under
data/raw - which pins a core and eventually leaves the process holding port
8000 with a closed socket, answering nothing. The symptom is every request
failing with "failed to fetch" while `lsof` shows nothing listening, because
the socket is CLOSED rather than LISTEN.

Running through this module makes the scoping structural rather than something
to remember:

    python -m app                 # from backend/, with PYTHONPATH=backend:src

The Makefile's dev-backend target passes the same directories explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

# Only source is watched. Everything else in the tree is either enormous or
# irrelevant to the running server.
BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent.parent
RELOAD_DIRS = [str(BACKEND_ROOT), str(PROJECT_ROOT / "src")]


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("INDU_HOST", "127.0.0.1"),
        port=int(os.environ.get("INDU_PORT", "8000")),
        reload=os.environ.get("INDU_RELOAD", "1") != "0",
        reload_dirs=RELOAD_DIRS,
    )


if __name__ == "__main__":
    main()
