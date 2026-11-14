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


# Mirrors of how spaces 0.48.3 builds its errors, kept local so the tests do not
# import private spaces modules: HTMLError and error(..., html=True) from
# spaces/zero/gradio.py, html_string from spaces/zero/client.py.
class _HTMLError(gr.Error):
    def __str__(self):
        return str(self.message)


def _html_string(html_contents: str, text_contents: str) -> str:
    class _HTMLString(str):
        def __str__(self):
            return text_contents

    return _HTMLString(html_contents)


_QUOTA = "ZeroGPU quota exceeded"
_WAIT = "(240s requested vs. 12s left). Try again in 1:23:45"


# Each exception is built the way spaces/zero/client.py:schedule raises it.
@pytest.mark.parametrize(
    ("exc", "category"),
    [
        pytest.param(
            _HTMLError(
                _html_string(
                    f"You have exceeded your free GPU quota {_WAIT}",
                    "User is runnning out of daily ZeroGPU quotas. Visit "
                    "https://huggingface.co/subscribe/pro to get more ZeroGPU quota now.",
                ),
                title=_QUOTA,
            ),
            "quota_exceeded",
            id="quota-free-account",
        ),
        pytest.param(
            gr.Error(
                "Space app has reached its GPU limit. Try re-running outside of "
                "examples if it happened after clicking one",
                title=_QUOTA,
            ),
            "quota_exceeded",
            id="quota-no-token",
        ),
        pytest.param(
            gr.Error("You have exceeded your runs limit.", title=_QUOTA),
            "quota_exceeded",
            id="quota-runs-limit-plain",
        ),
        pytest.param(
            _HTMLError("You have exceeded your runs limit.", title=_QUOTA),
            "quota_exceeded",
            id="quota-runs-limit-html",
        ),
        pytest.param(
            _HTMLError(
                _html_string(
                    f"You have exceeded your GPU quota {_WAIT}",
                    "Unlogged user is runnning out of daily ZeroGPU quotas. Signup for "
                    "free on https://huggingface.co/join or login on "
                    "https://huggingface.co/login to get more ZeroGPU quota now.",
                ),
                title=_QUOTA,
            ),
            "unlogged",
            id="quota-signed-out-keeps-sign-in-card",
        ),
        pytest.param(
            _HTMLError(
                _html_string(
                    f"You have exceeded your Pro GPU quota {_WAIT}",
                    f"You have exceeded your Pro GPU quota {_WAIT}",
                ),
                title=_QUOTA,
            ),
            "quota_exceeded",
            id="quota-pro",
        ),
        pytest.param(
            gr.Error(
                "The requested GPU duration (360s) is larger than the maximum allowed",
                title="ZeroGPU illegal duration",
            ),
            "illegal_duration",
            id="illegal-duration",
        ),
        pytest.param(
            gr.Error("GPU task aborted", title="ZeroGPU worker error"),
            "gpu_timeout",
            id="worker-aborted",
        ),
        pytest.param(
            gr.Error("Expired ZeroGPU proxy token", title="ZeroGPU client error"),
            "expired_token",
            id="expired-token",
        ),
        pytest.param(
            gr.Error("No GPU was available", title="ZeroGPU client error"),
            "execution",
            id="no-gpu-available",
        ),
        pytest.param(
            _HTMLError(
                _html_string(
                    "No GPU was available after 60s. Retry later",
                    "No GPU was available after 60s Retry later",
                ),
                title="ZeroGPU queue timeout",
            ),
            "execution",
            id="queue-timeout",
        ),
        pytest.param(RuntimeError("boom"), "execution", id="plain-runtime-error"),
    ],
)
def test_classify_zerogpu_errors_as_spaces_raises_them(exc, category):
    assert backend._classify(exc) == category


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
