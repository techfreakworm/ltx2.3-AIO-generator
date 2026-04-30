"""Materialize all LTX 2.3 model files for every mode by walking each template."""

from __future__ import annotations

import pathlib
import sys

# Ensure project root is on sys.path so `import models` / `import workflow` work
# when this script is invoked directly (e.g. `python tools/refresh_models.py`).
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import models  # noqa: E402
from workflow import VALID_MODES, load_template  # noqa: E402


def main() -> int:
    needed: set[str] = set()
    for mode in VALID_MODES:
        try:
            wf = load_template(mode)
            needed.update(models.walk_workflow_for_models(wf))
        except FileNotFoundError:
            print(f"  WARNING: workflows/{mode}.json missing — run tools/extract_modes.py first")
    if not needed:
        print("Nothing to do.")
        return 0
    print(f"Materializing {len(needed)} model files...")
    for event in models.ensure_models(needed):
        marker = "OK" if event.mb_done >= event.mb_total else "DL"
        print(f"  [{marker}] {event.filename}  {event.mb_done:.1f}/{event.mb_total:.1f} MB")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
