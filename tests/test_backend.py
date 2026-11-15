"""Backend tests — most are smoke / structural since the real work is GPU."""

import pathlib
import sys
import types

import gradio as gr
import pytest

import backend
import modes
import workflow

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

# spaces 0.51 wording. The first four are verbatim from the Space's runtime log on
# 2026-11-15 (spaces 0.51.3); the rest follow schedule() in spaces 0.51.1
# zero/client.py, which adds a suffix by account only for API callers.
_SIGNED_OUT_RUNS_LIMIT_API = (
    "You have exceeded your ZeroGPU runs limit. Authenticate with a Hugging Face token "
    "for more quota - https://huggingface.co/settings/tokens"
)
_FREE_ACCOUNT_QUOTA_API = (
    "You have exceeded your free ZeroGPU quota (279s requested vs. 21s left). Try again in "
    "23:56:38. Subscribe to Hugging Face PRO to get 40 min of ZeroGPU quota a day - "
    "https://huggingface.co/subscribe/pro?from=ZeroGPU"
)
_PRO_ACCOUNT_QUOTA_API = (
    "You have exceeded your Pro ZeroGPU quota (360s requested vs. 184s left). Try again in "
    "0:12:33. Add credits to keep using ZeroGPU. $1 for 10 minutes of ZeroGPU inference - "
    "https://huggingface.co/settings/billing?add-credits=true"
)
_LARGER_THAN_ALLOWED = (
    "The requested GPU duration (297s) is larger than the maximum allowedSubscribe to "
    "Hugging Face PRO to allow GPU tasks up to 40 min - "
    "https://huggingface.co/subscribe/pro?from=ZeroGPU"
)
_AUTHENTICATE_SUFFIX = (
    "Authenticate with a Hugging Face token for more quota - https://huggingface.co/settings/tokens"
)
_RUNS_LIMIT_PAGE = "You have exceeded your ZeroGPU runs limit. "
_NO_ACCOUNT_QUOTA_PAGE = f"You have exceeded your ZeroGPU quota {_WAIT}. "


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
            "unlogged",
            id="quota-runs-limit-plain",
        ),
        pytest.param(
            _HTMLError("You have exceeded your runs limit.", title=_QUOTA),
            "unlogged",
            id="quota-runs-limit-html",
        ),
        pytest.param(
            gr.Error(_SIGNED_OUT_RUNS_LIMIT_API, title=_QUOTA),
            "unlogged",
            id="runs-limit-signed-out-api",
        ),
        pytest.param(
            gr.Error(_RUNS_LIMIT_PAGE, title=_QUOTA),
            "unlogged",
            id="runs-limit-from-page",
        ),
        pytest.param(
            gr.Error(
                _RUNS_LIMIT_PAGE + "Subscribe to Hugging Face PRO to get 40 min of ZeroGPU "
                "quota a day - https://huggingface.co/subscribe/pro?from=ZeroGPU",
                title=_QUOTA,
            ),
            "quota_exceeded",
            id="runs-limit-free-account-api",
        ),
        pytest.param(
            gr.Error(
                _RUNS_LIMIT_PAGE + "Add credits to keep using ZeroGPU. $1 for 10 minutes of "
                "ZeroGPU inference - https://huggingface.co/settings/billing?add-credits=true",
                title=_QUOTA,
            ),
            "quota_exceeded",
            id="runs-limit-pro-account-api",
        ),
        pytest.param(
            gr.Error(_NO_ACCOUNT_QUOTA_PAGE, title=_QUOTA),
            "unlogged",
            id="no-account-quota-from-page",
        ),
        pytest.param(
            gr.Error(_NO_ACCOUNT_QUOTA_PAGE + _AUTHENTICATE_SUFFIX, title=_QUOTA),
            "unlogged",
            id="no-account-quota-api",
        ),
        pytest.param(
            gr.Error(_FREE_ACCOUNT_QUOTA_API, title=_QUOTA),
            "quota_exceeded",
            id="free-account-quota-api",
        ),
        pytest.param(
            gr.Error(_PRO_ACCOUNT_QUOTA_API, title=_QUOTA),
            "quota_exceeded",
            id="pro-account-quota-api",
        ),
        pytest.param(
            gr.Error(_LARGER_THAN_ALLOWED, title="ZeroGPU illegal duration"),
            "illegal_duration",
            id="larger-than-allowed-names-no-caller",
        ),
        pytest.param(
            gr.Error(
                "The requested GPU duration (297s) is larger than the maximum allowed"
                + _AUTHENTICATE_SUFFIX,
                title="ZeroGPU illegal duration",
            ),
            "unlogged",
            id="larger-than-allowed-signed-out-suffix",
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


# GPU seconds _duration_for asks ZeroGPU for, per mode and preset, with multiplier 1.0.
_GPU_ASK_S = {
    "t2v": {"fast": 186, "balanced": 198, "quality": 240},
    "i2v": {"fast": 186, "balanced": 198, "quality": 240},
    "a2v": {"fast": 216, "balanced": 240, "quality": 240},
    "lipsync": {"fast": 240, "balanced": 240, "quality": 240},
    "keyframe": {"fast": 240, "balanced": 240, "quality": 240},
    "style": {"fast": 240, "balanced": 240, "quality": 240},
}
# A free account's daily ZeroGPU allowance, and the factor ZeroGPU multiplies the
# ask by on Blackwell nodes before checking it (spaces/zero/configs.json).
_FREE_ACCOUNT_DAILY_S = 300
_BLACKWELL_FACTOR = 1.5


def _gpu_ask(mode, preset, inputs, multiplier=1.0):
    """Call the estimator the way ZeroGPU does: _execute_workflow's positional arguments."""
    wf = workflow.load_template(mode)
    for patch in modes.MODE_REGISTRY[mode].parameterize_fn(inputs[mode]):
        workflow.set_input(wf, *patch)
    return backend._duration_for(None, wf, [], mode, preset, multiplier, None)


@pytest.mark.parametrize(
    ("mode", "preset", "expected"),
    [(mode, preset, ask) for mode, row in _GPU_ASK_S.items() for preset, ask in row.items()],
)
def test_duration_for_asks_the_pinned_gpu_seconds(mode, preset, expected, canonical_inputs):
    assert _gpu_ask(mode, preset, canonical_inputs) == expected


@pytest.mark.parametrize("mode", ["t2v", "i2v"])
def test_balanced_text_and_image_to_video_fit_a_free_account_on_blackwell(mode, canonical_inputs):
    """A free account's request above 300 s is refused before any GPU work."""
    ask = _gpu_ask(mode, "balanced", canonical_inputs)
    assert round(ask * _BLACKWELL_FACTOR) <= _FREE_ACCOUNT_DAILY_S


@pytest.mark.parametrize("mode", list(_GPU_ASK_S))
def test_balanced_never_asks_less_than_fast(mode, canonical_inputs):
    assert _gpu_ask(mode, "balanced", canonical_inputs) >= _gpu_ask(mode, "fast", canonical_inputs)


@pytest.mark.parametrize("mode", list(_GPU_ASK_S))
def test_duration_for_preset_name_is_case_insensitive(mode, canonical_inputs):
    assert _gpu_ask(mode, "Balanced", canonical_inputs) == _gpu_ask(
        mode, "balanced", canonical_inputs
    )


def test_duration_for_keeps_the_retry_multiplier_and_unknown_inputs_as_they_were(
    canonical_inputs,
):
    assert _gpu_ask("t2v", "balanced", canonical_inputs, multiplier=2.0) == 240
    assert _gpu_ask("t2v", "turbo", canonical_inputs) == 231

    wf = workflow.load_template("t2v")
    assert backend._duration_for(None, wf, [], "unknown-mode", "balanced", 1.0, None) == 240
