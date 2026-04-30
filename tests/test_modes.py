"""Unit tests for modes.py — MODE_REGISTRY and parameterize_fn correctness."""
import pytest

import modes


def test_mode_dataclass_has_expected_fields():
    """Mode dataclass exposes the expected attribute set."""
    fields = {"name", "label", "icon", "parameterize_fn", "stage_map"}
    actual = set(modes.Mode.__dataclass_fields__.keys())
    assert fields == actual


def test_mode_registry_is_a_dict():
    """MODE_REGISTRY exists and is a dict (entries added in Tasks 11–12)."""
    assert isinstance(modes.MODE_REGISTRY, dict)
