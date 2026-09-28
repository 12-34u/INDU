"""
Optional learned matcher, behind a lazy availability check.

LightGlue and its torch dependency are large and are deliberately excluded
from the deployed image. Nothing here is imported at module load: the import
happens inside `lightglue_available`, so a deployment without torch installed
starts normally and registration continues on its SIFT + MAGSAC++ baseline,
which is the configured default regardless.

This module intentionally does not register a matcher with the pipeline. It
exists so the optional dependency can be detected and reported rather than
discovered as an ImportError at request time.
"""

from __future__ import annotations

from typing import Optional


def lightglue_available() -> bool:
    """Whether the optional learned-matching dependencies are installed."""
    try:  # pragma: no cover - depends on the environment
        import torch  # noqa: F401
        import lightglue  # noqa: F401
    except Exception:
        return False
    return True


def unavailable_reason() -> Optional[str]:
    """Why the learned matcher cannot be used, or None when it can."""
    if lightglue_available():
        return None
    return (
        "torch and lightglue are not installed; they are optional and excluded "
        "from the deployed image. Registration uses its SIFT + MAGSAC++ baseline. "
        "Install with: pip install -r requirements-optional.txt"
    )
