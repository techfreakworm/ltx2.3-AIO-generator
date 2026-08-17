"""Backend tests — most are smoke / structural since the real work is GPU."""

import pathlib
import sys
import types

import gradio as gr
import pytest

import backend

# One output node, no model filenames, so submit()'s pre-flight download pass
# finds nothing to fetch.
_MINIMAL_WORKFLOW = {"9": {"class_type": "SaveVideo", "inputs": {}}}


def test_backend_class_exists():
    assert hasattr(backend, "ComfyUILibraryBackend")


def test_progress_event_dataclasses_exist():
    assert hasattr(backend, "DownloadEvent")
    assert hasattr(backend, "ProgressEvent")
    assert hasattr(backend, "OutputEvent")
    assert hasattr(backend, "ErrorEvent")


def test_classify_detects_illegal_duration_from_the_error_title():
    """spaces raises gr.Error(message, title='ZeroGPU illegal duration').

    The discriminating words are only in the title; str(exc) is the message
    alone, so matching on the message can never see them.
    """
    exc = gr.Error(
        "The requested GPU duration (360s) is larger than the maximum allowed",
        title="ZeroGPU illegal duration",
    )
    assert backend._classify(exc) == "illegal_duration"


def test_classify_still_detects_gpu_timeout_from_the_message():
    assert backend._classify(gr.Error("GPU task aborted")) == "gpu_timeout"


def test_classify_still_detects_expired_token_from_the_message():
    exc = gr.Error("Expired ZeroGPU proxy token", title="ZeroGPU client error")
    assert backend._classify(exc) == "expired_token"


def _stub_comfy_utils(monkeypatch) -> None:
    """_worker imports comfy.utils for the progress hook; ComfyUI isn't installed in CI."""
    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    utils.PROGRESS_BAR_HOOK = None
    utils.set_progress_bar_global_hook = lambda _hook: None
    comfy.utils = utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)


def _detached_backend(comfy_dir: pathlib.Path):
    """A backend instance without __init__ — constructing one needs a real ComfyUI tree."""
    be = backend.ComfyUILibraryBackend.__new__(backend.ComfyUILibraryBackend)
    be._comfy_dir = comfy_dir
    be._executor = object()
    return be


async def _drain(be) -> list:
    return [event async for event in be.submit("t2v", _MINIMAL_WORKFLOW)]


@pytest.mark.asyncio
async def test_submit_reports_error_when_workflow_produces_no_video(tmp_path, monkeypatch):
    """A node that dies mid-graph leaves an empty path; that must not read as success.

    ComfyUI logs the failure and returns normally, so the run used to end in an
    OutputEvent carrying video_path='' — the UI showed no error and just reset.
    """
    _stub_comfy_utils(monkeypatch)
    monkeypatch.setattr(backend, "_execute_workflow", lambda *_a, **_kw: "")

    events = await _drain(_detached_backend(tmp_path / "comfy"))

    assert not any(isinstance(e, backend.OutputEvent) for e in events)
    errors = [e for e in events if isinstance(e, backend.ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].category == "no_output"


@pytest.mark.asyncio
async def test_submit_emits_output_event_when_executor_returns_a_path(tmp_path, monkeypatch):
    _stub_comfy_utils(monkeypatch)
    monkeypatch.setattr(backend, "_execute_workflow", lambda *_a, **_kw: "/tmp/out.mp4")

    events = await _drain(_detached_backend(tmp_path / "comfy"))

    assert [e.video_path for e in events if isinstance(e, backend.OutputEvent)] == ["/tmp/out.mp4"]
    assert not any(isinstance(e, backend.ErrorEvent) for e in events)


@pytest.mark.asyncio
async def test_submit_prefers_recent_output_file_over_erroring(tmp_path, monkeypatch):
    """The filesystem fallback still runs first; only a truly empty result errors."""
    _stub_comfy_utils(monkeypatch)
    monkeypatch.setattr(backend, "_execute_workflow", lambda *_a, **_kw: "")
    comfy_dir = tmp_path / "comfy"
    produced = comfy_dir / "output" / "run.mp4"
    produced.parent.mkdir(parents=True)
    produced.write_bytes(b"x")

    events = await _drain(_detached_backend(comfy_dir))

    assert [e.video_path for e in events if isinstance(e, backend.OutputEvent)] == [str(produced)]
    assert not any(isinstance(e, backend.ErrorEvent) for e in events)
