"""Tests for the workflow-mode extractor."""

import json
import subprocess
import sys

from tests.conftest import REPO_ROOT


def test_extract_creates_six_mode_files(master_workflow, tmp_path):
    """extract_modes.py emits six valid mode-specific JSON templates."""
    out_dir = tmp_path / "workflows"
    master_path = tmp_path / "master.json"
    master_path.write_text(json.dumps(master_workflow))

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "extract_modes.py"),
            "--master",
            str(master_path),
            "--out",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    expected = {"t2v.json", "a2v.json", "i2v.json", "lipsync.json", "keyframe.json", "style.json"}
    actual = {p.name for p in out_dir.iterdir()}
    assert actual == expected

    # Each file must be valid JSON with at least one node.
    for path in out_dir.iterdir():
        wf = json.loads(path.read_text())
        assert "nodes" in wf
        assert len(wf["nodes"]) > 0
