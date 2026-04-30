"""ComfyUI library-mode backend.

Single-process, single-implementation. The @spaces.GPU decorator is the only
divergence between local and HF Spaces deployment.
"""

from __future__ import annotations

import asyncio
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


class _StubServer:
    """Minimal stub matching the surface ComfyUI's PromptExecutor expects."""

    client_id: str | None = "ltx23-aio"
    last_node_id: str | None = None

    def send_sync(self, event: str, data: dict, sid: str | None = None) -> None:
        pass

    def queue_updated(self) -> None:
        pass


def _comfy_dir() -> pathlib.Path:
    if _on_spaces():
        return pathlib.Path("/data/comfyui")
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
        self, mode: str, workflow: dict, gpu_duration: int = 120
    ) -> AsyncIterator[Any]:
        """Run a workflow end-to-end. Yields Download/Progress/Output/Error events."""
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

        def _hook(value: int, total: int, _preview=None) -> None:
            _push(
                ProgressEvent(
                    stage=0,
                    stage_label="diffusion",
                    step=int(value),
                    total_steps=int(total),
                )
            )

        def _worker() -> None:
            import comfy.utils

            saved_hook = getattr(comfy.utils, "PROGRESS_BAR_HOOK", None)
            try:
                # Use the public setter; it writes the same global the
                # ProgressBar class reads, but is the documented API.
                comfy.utils.set_progress_bar_global_hook(_hook)
                self._executor.execute(
                    workflow,
                    prompt_id="ltx23-aio",
                    extra_data={"client_id": "ltx23-aio"},
                    execute_outputs=[],
                )
                # PromptExecutor writes output files via VHS_VideoCombine; we read its
                # history to find the most recent saved video.
                outputs = list(self._executor.outputs.values())
                video_path = _first_video_path(outputs) or ""
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

        if _on_spaces():
            import spaces

            execute = spaces.GPU(duration=gpu_duration)(_worker)
            thread = threading.Thread(target=execute, daemon=True)
        else:
            thread = threading.Thread(target=_worker, daemon=True)
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
    if "outofmemory" in name or "cuda out of memory" in str(exc).lower():
        return "oom"
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


def _first_video_path(outputs: Iterable) -> str | None:
    """Find the first .mp4 path emitted by VHS_VideoCombine in PromptExecutor outputs."""
    for output in outputs:
        if not isinstance(output, dict):
            continue
        for value in output.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "filename" in item:
                        fn = item["filename"]
                        if fn.endswith((".mp4", ".webm", ".mov")):
                            return item.get("fullpath", fn)
    return None
