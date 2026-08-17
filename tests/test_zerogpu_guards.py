"""Main-process guards against ZeroGPU's CUDA emulation.

On Spaces the GPU exists only inside the @spaces.GPU fork. ZeroGPU patches a
few torch.cuda entry points in the parent process (is_available, current_device,
get_device_properties) but leaves the real ones — empty_cache, mem_get_info — in
place, and those reach torch._C._cuda_init, which it replaces with a raising
stub. Parent-process code has to stay off those paths.
"""

import sys
import types

import app
import backend

_CUDA_INIT_ERROR = (
    "Low-level CUDA init (torch._C._cuda_init) reached. This means ZeroGPU's "
    "PyTorch CUDA emulation mode did not intercept a CUDA operation in your code."
)


def _install_zerogpu_stubs(monkeypatch, raising: bool = True) -> list[str]:
    """Stub torch + comfy.model_management the way ZeroGPU's parent process behaves.

    Returns the list that records every trapped call, so a test can assert the
    guard skipped them entirely rather than merely swallowing the exception.
    """
    reached: list[str] = []

    def _trap(name):
        def _call(*_args, **_kwargs):
            reached.append(name)
            if raising:
                raise RuntimeError(_CUDA_INIT_ERROR)
            return 0

        return _call

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=_trap("torch.cuda.empty_cache"),
        get_device_properties=lambda *_a, **_kw: types.SimpleNamespace(total_memory=0),
    )
    torch.mps = types.SimpleNamespace(empty_cache=_trap("torch.mps.empty_cache"))
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = _trap("mm.unload_all_models")
    mm.get_torch_device = lambda: "cuda:0"
    mm.get_free_memory = _trap("mm.get_free_memory")
    mm.current_loaded_models = []
    comfy.model_management = mm

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return reached


def test_unload_models_skips_cuda_on_spaces(monkeypatch):
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    reached = _install_zerogpu_stubs(monkeypatch)

    html = app._unload_models()

    assert reached == []
    assert "unload failed" not in html


def test_unload_models_empties_cache_off_spaces(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    reached = _install_zerogpu_stubs(monkeypatch, raising=False)
    monkeypatch.setattr(app, "_render_model_status", lambda: "<status>")

    assert app._unload_models() == "<status>"
    assert "mm.unload_all_models" in reached
    assert "torch.cuda.empty_cache" in reached


def test_render_model_status_skips_cuda_on_spaces(monkeypatch):
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    reached = _install_zerogpu_stubs(monkeypatch)
    monkeypatch.setattr(app, "_get_backend", lambda: object())

    html = app._render_model_status()

    assert reached == []
    assert "memory probe failed" not in html


def test_free_memory_skips_cuda_on_spaces(monkeypatch):
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    reached = _install_zerogpu_stubs(monkeypatch)

    backend._free_memory()

    assert reached == []


def test_free_memory_empties_cache_off_spaces(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    reached = _install_zerogpu_stubs(monkeypatch, raising=False)

    backend._free_memory()

    assert "mm.unload_all_models" in reached
    assert "torch.cuda.empty_cache" in reached
