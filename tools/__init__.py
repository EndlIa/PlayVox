"""PlayVox tooling package.

Puts the repository root on ``sys.path`` so ``import forehoi`` / ``import ui``
work regardless of the caller's cwd.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Default config location. The app passes its own path derived from
# ``PLAYVOX_DATA``; this is only a fallback for scripts.
_DATA = os.environ.get("PLAYVOX_DATA", "")
PIPELINE_CONFIG = os.environ.get(
    "PLAYVOX_PIPELINE_CONFIG",
    os.path.join(_DATA, "checkpoints", "pipeline.yaml") if _DATA else "",
)
