"""ComfyUI library-mode backend.

Single-process, single-implementation. The @spaces.GPU decorator is the only
divergence between local and HF Spaces deployment.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import pathlib
import sys
import threading
import traceback as tb_mod
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

import models


@dataclass
class DownloadEvent:
    filename: str
    mb_done: float
    mb_total: float


@dataclass
class ProgressEvent:
    stage: int
    stage_label: str
    step: int
    total_steps: int


@dataclass
class OutputEvent:
    video_path: str
    audio_path: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class ErrorEvent:
    category: str  # "oom" | "zerogpu_timeout" | "execution" | "interrupt" | "download"
    message: str
    stage: int | None = None
    traceback: str = ""


def _on_spaces() -> bool:
    return bool(os.environ.get("SPACES_ZERO_GPU"))


try:
    import spaces  # type: ignore
except ImportError:
    spaces = None  # type: ignore[assignment]


def _identity(fn):
    return fn


# --- Per-call ZeroGPU duration estimator -----------------------------------
# `duration` is a per-call timeout. Shorter declared duration → faster queue
# priority on the shared ZeroGPU pool. Estimating from (mode, preset, frames)
# instead of using a one-size-fits-all 600s cap means light T2V calls jump
# the queue while heavy modes (lipsync, style) reserve real headroom.

_BASE_DURATION_S: dict[str, int] = {
    # Rough sampler+decode time at ~120 frames, balanced preset, warm cache.
    "t2v": 90,
    "i2v": 90,
    "a2v": 120,
    "lipsync": 240,   # extra: audio encoder + audio VAE + extra LoRAs
    "keyframe": 180,
    "style": 360,     # extra: preprocessor (canny/dwpose/depth) + IC-LoRAs
}
_PRESET_MULT: dict[str, float] = {"fast": 1.0, "balanced": 1.5, "quality": 3.0}


def _frames_from_workflow(workflow: dict) -> int:
    """Read the frame count from the workflow's EmptyLTXVLatentVideo node."""
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == "EmptyLTXVLatentVideo":
            try:
                return int((node.get("inputs") or {}).get("length", 121))
            except (TypeError, ValueError):
                return 121
    return 121


def _duration_for(
    executor: Any,
    workflow: dict,
    output_ids: list[str],
    mode: str,
    preset: str,
    multiplier: float = 1.0,
    progress: Any = None,
) -> int:
    """ZeroGPU duration estimator. Same signature as _execute_workflow.

    `progress` is a gr.Progress instance forwarded by the caller; we ignore it
    here (estimator doesn't emit progress) but must accept it positionally so
    ZeroGPU can call us with the same arg list it'll use for _execute_workflow.

    Estimate = (base × preset multiplier + cold-cache buffer + per-frame VAE
    decode time) × retry multiplier, clamped to [60s, 900s]. The 900s ceiling
    keeps a single failed call from torching the daily quota.
    """
    base = _BASE_DURATION_S.get(mode, 180)
    mult = _PRESET_MULT.get(preset.lower(), 1.5)
    frames = _frames_from_workflow(workflow)
    est = int((base * mult + 60 + frames * 0.3) * multiplier)
    return max(60, min(est, 900))


# Decorate at module load time so ZeroGPU's startup analyzer detects it.
_GPU = (
    spaces.GPU(duration=_duration_for)
    if (spaces is not None and _on_spaces())
    else _identity
)


@_GPU
def _execute_workflow(
    executor: Any,
    workflow: dict,
    output_ids: list[str],
    mode: str,
    preset: str,
    multiplier: float = 1.0,
    progress: Any = None,
) -> str:
    """Run the workflow on GPU and return the path of the first video output.

    Returns just the video path (a plain string, picklable across the
    @spaces.GPU subprocess boundary). The `mode`, `preset`, and `multiplier`
    args are consumed by `_duration_for` to estimate the GPU slot to reserve.

    `progress` is an optional `gr.Progress` instance. It's the only progress
    channel that crosses the @spaces.GPU subprocess boundary on HF Spaces —
    Gradio + the `spaces` library wrap it with cross-process IPC. When set,
    we mirror ComfyUI's step counter into it via the global progress hook,
    chaining to whatever hook was already installed (so the local event-based
    status banner keeps working alongside).
    """
    if progress is not None:
        import comfy.utils as _cu
        _saved_hook = getattr(_cu, "PROGRESS_BAR_HOOK", None)

        def _gp_hook(value, total, _preview=None, **_kw):
            try:
                v, t = int(value), int(total)
                progress(v / max(t, 1), desc=f"Sampling step {v}/{t}")
            except Exception:
                pass
            if _saved_hook is not None:
                try:
                    _saved_hook(value, total, _preview)
                except Exception:
                    pass

        _cu.set_progress_bar_global_hook(_gp_hook)

    executor.execute(
        workflow,
        prompt_id="ltx23-aio",
        extra_data={"client_id": "ltx23-aio"},
        execute_outputs=output_ids,
    )
    hist = getattr(executor, "history_result", {}) or {}
    outs = hist.get("outputs") or {}
    for output in outs.values():
        if not isinstance(output, dict):
            continue
        for value in output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict):
                    fn = item.get("filename") or ""
                    if fn.endswith((".mp4", ".webm", ".mov")):
                        return item.get("fullpath") or fn
    return ""


class _StubServer:
    """Minimal stub matching the surface ComfyUI's PromptExecutor expects."""

    client_id: str | None = "ltx23-aio"
    last_node_id: str | None = None

    def send_sync(self, event: str, data: dict, sid: str | None = None) -> None:
        pass

    def queue_updated(self) -> None:
        pass


class _StubPromptQueue:
    """Stub matching the surface VideoHelperSuite + others touch."""

    currently_running: dict = {}
    history: dict = {}
    flags: dict = {}

    def get_current_queue(self) -> tuple[list, list]:
        return ([], [])

    def get_tasks_remaining(self) -> int:
        return 0

    def set_flag(self, name: str, data) -> None:
        pass

    def get_flags(self, *a, **kw) -> dict:
        return {}

    def task_done(self, *a, **kw) -> None:
        pass

    def put(self, *a, **kw) -> None:
        pass

    def wipe_queue(self) -> None:
        pass

    def delete_queue_item(self, *a, **kw) -> None:
        pass


class _StubPromptServerInstance:
    """Surface that ComfyUI's `server.PromptServer.instance` exposes to custom nodes.

    VideoHelperSuite, KJNodes, and others read this at import time. They mostly
    use it to register HTTP routes or send WS events or peek at the prompt queue.
    No-ops here are fine — we have no real server.
    """

    client_id: str | None = "ltx23-aio"
    # KJNodes' preview thread reads `last_node_id.encode('ascii')` directly.
    # ComfyUI's real server keeps it as a string per executing node and resets
    # to None at end-of-prompt — which races the preview thread. Keep it a
    # safe non-empty string so .encode() never NPEs.
    last_node_id: str = "ltx23-aio"
    web_root: str = ""

    class _Routes:
        def get(self, *a, **kw):
            return lambda fn: fn

        def post(self, *a, **kw):
            return lambda fn: fn

        def static(self, *a, **kw):
            return None

    routes = _Routes()
    sockets: dict = {}
    prompt_queue = _StubPromptQueue()
    # Custom-Scripts checks PromptServer.instance.supports — claim the
    # "custom_nodes_from_web" capability so it skips its JS install path.
    supports: list[str] = ["custom_nodes_from_web"]
    web_root: str = ""

    def add_routes(self) -> None:
        pass

    def send_sync(self, event: str, data: dict, sid: str | None = None) -> None:
        pass

    def send_progress_text(self, text: str, node_id=None, sid=None) -> None:
        # Comfy_extras nodes call this; we just no-op since we don't have a UI
        # to surface intermediate text on.
        pass

    def queue_updated(self) -> None:
        pass

    def get_node_class_def(self, *a, **kw):
        return None

    def __getattr__(self, name):
        # Anything else our custom nodes might reach for — give them a no-op.
        # This is a deliberate liberal catch-all so the inference path doesn't
        # die on cosmetic UI hooks. Inspection-style access (hasattr) gets True.
        def _noop(*a, **kw):
            return None
        return _noop


def _comfy_dir() -> pathlib.Path:
    if _on_spaces():
        return pathlib.Path.home() / "comfyui"
    return pathlib.Path(__file__).parent / "comfyui"


class ComfyUILibraryBackend:
    """Wraps PromptExecutor for in-process workflow execution."""

    def __init__(self) -> None:
        self._comfy_dir = _comfy_dir()
        if not self._comfy_dir.exists():
            raise RuntimeError(
                f"ComfyUI not found at {self._comfy_dir}. "
                f"Local: run `bash setup.sh`. Spaces: see app.py:_bootstrap()."
            )
        if str(self._comfy_dir) not in sys.path:
            sys.path.insert(0, str(self._comfy_dir))

        # Defer comfy imports until the path is set up.
        # NOTE: ComfyUI ships PromptExecutor in the top-level `execution.py`
        # module, NOT under `comfy.execution`. Same for `nodes`. Both must be
        # imported AFTER the sys.path insert above.
        import asyncio
        import threading

        import comfy.cli_args  # noqa: F401 — side-effect: registers CLI flags
        import execution  # top-level module — provides PromptExecutor
        import nodes  # top-level module — provides init_extra_nodes (async)

        # CRITICAL ordering fix: ComfyUI's nodes.py:24 inserts `comfyui/comfy/`
        # at sys.path[0]. That dir contains a module-style `utils.py`, which
        # shadows `comfyui/utils/` (a package containing install_util.py).
        # Some custom nodes (KJNodes, VideoHelperSuite via app.frontend_management)
        # do `from utils.install_util import …` and get `comfy/utils.py` instead,
        # raising "'utils' is not a package". Rewrite sys.path so comfy_dir is
        # ahead of comfy_dir/comfy and force-clear any cached `utils` binding.
        comfy_subdir = str(self._comfy_dir / "comfy")
        sys.path = [p for p in sys.path if p not in (str(self._comfy_dir), comfy_subdir)]
        sys.path.insert(0, comfy_subdir)
        sys.path.insert(0, str(self._comfy_dir))
        if "utils" in sys.modules and not getattr(sys.modules["utils"], "__path__", None):
            del sys.modules["utils"]

        # Some custom nodes (e.g. VideoHelperSuite) read `server.PromptServer.instance`
        # at import time. We don't run a real ComfyUI server, so install a stub
        # that exposes the attributes those nodes touch (sockets, send, etc.).
        import server as comfy_server

        if getattr(comfy_server.PromptServer, "instance", None) is None:
            comfy_server.PromptServer.instance = _StubPromptServerInstance()

        # `nodes.init_extra_nodes` is async. We may be called from within a
        # running event loop (Gradio's handler) — running `asyncio.run()` there
        # raises. Run the coroutine in a fresh loop on a worker thread instead.
        def _init_in_thread() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(nodes.init_extra_nodes())
            finally:
                loop.close()

        thread = threading.Thread(target=_init_in_thread, daemon=False)
        thread.start()
        thread.join()
        # PromptExecutor expects a `server` with client_id, send_sync, last_node_id,
        # queue_updated. A minimal stub no-ops all of them — we don't run a real
        # websocket server, we surface progress via comfy.utils.PROGRESS_BAR_HOOK.
        # cache_args["ram"] is read unconditionally inside execute_async even when
        # cache_type is the default false — provide a sensible default so it doesn't
        # NoneType-subscript at line 727.
        self._executor = execution.PromptExecutor(
            server=_StubServer(),
            cache_args={"ram": 16.0, "lru": 0},
        )

    def __repr__(self) -> str:
        return f"ComfyUILibraryBackend(comfy_dir={self._comfy_dir!r})"

    async def submit(
        self,
        mode: str,
        workflow: dict,
        *,
        preset: str = "balanced",
        duration_multiplier: float = 1.0,
        gpu_duration: int = 0,  # legacy, ignored (now derived from preset+frames)
        progress: Any = None,
    ) -> AsyncIterator[Any]:
        """Run a workflow end-to-end. Yields Download/Progress/Output/Error events.

        `preset` and `duration_multiplier` flow through to the @spaces.GPU
        duration estimator. The handler can re-call submit() with
        duration_multiplier=2.0 if the first attempt aborts on timeout.
        """
        # Pre-flight: ensure all model files exist.
        try:
            needed = models.walk_workflow_for_models(workflow)
            for download_event in models.ensure_models(needed):
                yield download_event
        except Exception as e:
            yield ErrorEvent(
                category="download",
                message=str(e),
                traceback=tb_mod.format_exc(),
            )
            return

        # Run the inference in a worker thread; pass progress events through a queue.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _push(event: Any) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

        # Track stage progression. ComfyUI fires the progress hook from inside
        # samplers, so we advance the stage every time we observe a new sampler
        # starting (step==0 with a different total than before, or a "new run"
        # signal — value smaller than the running max for the same total).
        progress_state = {"stage": 0, "prev_total": -1, "max_step": -1}

        def _hook(value: int, total: int, _preview=None, **_kwargs: Any) -> None:
            v, t = int(value), int(total)
            # New sampler started (different total, or step rewound)
            if t != progress_state["prev_total"] or v < progress_state["max_step"]:
                progress_state["stage"] += 1
                progress_state["prev_total"] = t
                progress_state["max_step"] = v
            else:
                progress_state["max_step"] = max(progress_state["max_step"], v)
            _push(
                ProgressEvent(
                    stage=progress_state["stage"],
                    stage_label="diffusion",
                    step=v,
                    total_steps=t,
                )
            )

        def _worker() -> None:
            import comfy.utils

            saved_hook = getattr(comfy.utils, "PROGRESS_BAR_HOOK", None)
            try:
                # Workflow is already API-format (saved from ComfyUI editor's
                # "Save (API Format)"), so it can be handed to PromptExecutor
                # directly. The execute_outputs list pinpoints which output
                # nodes to evaluate — we let PromptExecutor walk the whole
                # graph by passing every output-class node id.
                output_ids = [
                    nid for nid, n in workflow.items()
                    if n.get("class_type", "").startswith(("SaveVideo", "VHS_VideoCombine", "PreviewAudio", "CreateVideo"))
                ]
                print(
                    f"[backend] submitting workflow: {len(workflow)} nodes, "
                    f"output_ids={output_ids}",
                    file=sys.stderr,
                    flush=True,
                )
                # Use the public setter; it writes the same global the
                # ProgressBar class reads, but is the documented API.
                comfy.utils.set_progress_bar_global_hook(_hook)
                # _execute_workflow is module-level and decorated with a
                # @spaces.GPU(duration=callable) on Spaces — the callable
                # estimates per-call timeout from (mode, preset, frames) so
                # light calls get fast queue priority while heavy ones reserve
                # real headroom. Off-Spaces it's a plain call.
                video_path = _execute_workflow(
                    self._executor, workflow, output_ids, mode, preset, duration_multiplier, progress,
                )
                # Fallback: if history_result didn't surface a path (rare on
                # Spaces — happens when ZeroGPU's subprocess boundary drops
                # mutated state), scan the output dir for the newest mp4
                # written within the last 60 s.
                if not video_path:
                    video_path = _newest_recent_video(self._comfy_dir / "output") or ""
                print(
                    f"[backend] workflow done; video_path={video_path!r}",
                    file=sys.stderr,
                    flush=True,
                )
                _push(OutputEvent(video_path=video_path))
            except Exception as exc:
                tb_text = tb_mod.format_exc()
                print(f"[backend] worker exception:\n{tb_text}", file=sys.stderr, flush=True)
                _push(
                    ErrorEvent(
                        category=_classify(exc),
                        message=str(exc),
                        traceback=tb_text,
                    )
                )
            finally:
                comfy.utils.set_progress_bar_global_hook(saved_hook)
                _free_memory()
                _push(None)  # sentinel: stop the consumer

        # ZeroGPU's @spaces.GPU wrapper reads the user's identity from the
        # current Gradio request via gradio.context.LocalContext.request,
        # which is a contextvar. Plain threads don't inherit contextvars, so
        # without this the worker sees request=None, X-IP-Token never gets
        # read, and `client.schedule` raises "Space app has reached its GPU
        # limit" (token-is-None branch in spaces/zero/client.py:138). Copy
        # the calling task's context so the request — and therefore the Pro
        # user's quota attribution — survives the thread boundary.
        ctx = contextvars.copy_context()
        thread = threading.Thread(target=ctx.run, args=(_worker,), daemon=True)
        thread.start()

        while True:
            event = await queue.get()
            if event is None:
                return
            yield event

    def interrupt(self) -> None:
        """Cancel the currently running workflow (if any)."""
        try:
            import comfy.model_management as mm

            mm.interrupt_current_processing()
        except Exception:
            pass


def _classify(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "outofmemory" in name or "cuda out of memory" in msg:
        return "oom"
    # ZeroGPU enforces the @spaces.GPU(duration=N) cap and re-raises as
    # gradio.exceptions.Error('GPU task aborted'). Surface a distinct
    # category so the handler can offer a retry with a bigger budget.
    if "gpu task aborted" in msg or ("gpu" in msg and "aborted" in msg):
        return "gpu_timeout"
    if "interrupt" in name:
        return "interrupt"
    return "execution"


def _free_memory() -> None:
    """Free VRAM after a workflow finishes (success or failure)."""
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
    except Exception:
        pass
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _newest_recent_video(output_root: pathlib.Path, within_seconds: float = 60.0) -> str | None:
    """Filesystem fallback: return the newest .mp4/.webm/.mov under *output_root*
    that was modified within the last *within_seconds* seconds.

    Used when the executor's history_result didn't surface a path — typically
    happens when ZeroGPU's subprocess boundary drops the mutation. The disk
    is shared, so the file is there even when the in-memory state isn't.
    """
    import time

    if not output_root.exists():
        return None
    cutoff = time.time() - within_seconds
    candidates: list[tuple[float, pathlib.Path]] = []
    for ext in (".mp4", ".webm", ".mov"):
        for p in output_root.rglob(f"*{ext}"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff:
                candidates.append((mtime, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return str(candidates[0][1])
