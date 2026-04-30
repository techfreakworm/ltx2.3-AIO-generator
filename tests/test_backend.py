"""Backend tests — most are smoke / structural since the real work is GPU."""

import backend


def test_backend_class_exists():
    assert hasattr(backend, "ComfyUILibraryBackend")


def test_progress_event_dataclasses_exist():
    assert hasattr(backend, "DownloadEvent")
    assert hasattr(backend, "ProgressEvent")
    assert hasattr(backend, "OutputEvent")
    assert hasattr(backend, "ErrorEvent")
