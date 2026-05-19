"""Tests for the output-dir TTL sweeper.

ComfyUI writes every generated video/audio into `<comfy_dir>/output/...` and
nothing in this app cleans them up — left alone, they accumulate until the
Spaces ephemeral disk fills and the container goes unhealthy. `_prune_old_outputs`
sweeps files older than a TTL each time the user clicks Generate.
"""

import os
import pathlib
import time

import app


def _touch(path: pathlib.Path, mtime: float | None = None) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_prune_old_outputs_deletes_files_older_than_threshold(tmp_path: pathlib.Path) -> None:
    stale = _touch(tmp_path / "old.mp4", mtime=time.time() - 5 * 3600)

    app._prune_old_outputs(tmp_path, max_age_seconds=4 * 3600)

    assert not stale.exists()


def test_prune_old_outputs_keeps_files_younger_than_threshold(tmp_path: pathlib.Path) -> None:
    fresh = _touch(tmp_path / "new.mp4", mtime=time.time() - 60)

    app._prune_old_outputs(tmp_path, max_age_seconds=4 * 3600)

    assert fresh.exists()


def test_prune_old_outputs_returns_deleted_count(tmp_path: pathlib.Path) -> None:
    _touch(tmp_path / "a.mp4", mtime=time.time() - 5 * 3600)
    _touch(tmp_path / "b.mp4", mtime=time.time() - 6 * 3600)
    _touch(tmp_path / "fresh.mp4", mtime=time.time() - 60)

    deleted = app._prune_old_outputs(tmp_path, max_age_seconds=4 * 3600)

    assert deleted == 2


def test_prune_old_outputs_recurses_into_subdirs(tmp_path: pathlib.Path) -> None:
    # ComfyUI writes under <output>/LTX2.3/<filename>.mp4 — the sweeper has
    # to descend or it'd never touch real outputs.
    stale = _touch(tmp_path / "LTX2.3" / "old.mp4", mtime=time.time() - 5 * 3600)

    app._prune_old_outputs(tmp_path, max_age_seconds=4 * 3600)

    assert not stale.exists()


def test_prune_old_outputs_handles_missing_directory(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "does-not-exist"

    # No-op, no crash. Returning 0 is the only sensible answer.
    assert app._prune_old_outputs(missing, max_age_seconds=4 * 3600) == 0


def test_prune_old_outputs_leaves_directories_alone(tmp_path: pathlib.Path) -> None:
    # The sweeper deletes *files*. Subdirs should remain so the next
    # generation can write into the same path layout.
    subdir = tmp_path / "LTX2.3"
    subdir.mkdir()

    app._prune_old_outputs(tmp_path, max_age_seconds=4 * 3600)

    assert subdir.exists() and subdir.is_dir()
