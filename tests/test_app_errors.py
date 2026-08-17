"""User-facing translation of backend error categories."""

import app


def test_friendly_error_explains_a_workflow_that_produced_no_video():
    """'no_output' almost always means a model file failed to load mid-graph."""
    assert "no_output" in app._FRIENDLY_ERRORS
    title, body = app._friendly_error("no_output", "raw message")
    generic_title, generic_body = app._friendly_error("some_unmapped_category", "raw message")
    assert (title, body) != (generic_title, generic_body)
    assert "model" in body.lower()


def test_friendly_error_falls_back_for_unknown_categories():
    title, _body = app._friendly_error("totally_unknown", "raw message")
    assert title == "Generation failed"
