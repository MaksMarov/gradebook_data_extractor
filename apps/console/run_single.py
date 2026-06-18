from __future__ import annotations

import sys
from pathlib import Path


# When this file is launched as:
#   python apps/console/run_single.py
# Python adds apps/console to sys.path, but not the repository root.
# Add repository root manually so "apps.console..." imports work.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.console.run_extractor import main


if __name__ == "__main__":
    raise SystemExit(main(["single", *sys.argv[1:]]))
