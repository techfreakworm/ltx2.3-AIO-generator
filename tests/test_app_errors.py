"""User-facing translation of backend error categories."""

import gradio as gr
import pytest

import app
import backend

# Verbatim from the Space's runtime log on 2026-11-15 (spaces 0.51.3): what a signed-out
# caller is told. The second names no caller; a free account gets the same sentence.
_SIGNED_OUT_REFUSALS = [
    gr.Error(
        "You have exceeded your ZeroGPU runs limit. Authenticate with a Hugging Face token "
        "for more quota - https://huggingface.co/settings/tokens",
        title="ZeroGPU quota exceeded",
    ),
    gr.Error(
        "The requested GPU duration (297s) is larger than the maximum allowedSubscribe to "
        "Hugging Face PRO to allow GPU tasks up to 40 min - "
        "https://huggingface.co/subscribe/pro?from=ZeroGPU",
        title="ZeroGPU illegal duration",
    ),
]


def _card_text(category):
    title, body = app._FRIENDLY_ERRORS[category]
    return f"{title} {body}".lower()


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


def test_unlogged_card_asks_the_visitor_to_sign_in_and_says_why():
    assert app._FRIENDLY_ERRORS["unlogged"][0] == "Sign in to Hugging Face"
    text = _card_text("unlogged")
    assert "sign in" in text
    assert "huggingface.co" in text
    assert "too little gpu time" in text
    for wrong_advice in ("shorter video", "fast preset", "25 min"):
        assert wrong_advice not in text


def test_illegal_duration_card_covers_signed_out_and_free_readers():
    """ZeroGPU's text does not say who the caller is, so the card has to address both."""
    assert app._FRIENDLY_ERRORS["illegal_duration"][0] == "More GPU time than your account allows"
    text = _card_text("illegal_duration")
    assert "not signed in" in text and "sign in at huggingface.co" in text
    assert "free account" in text and "fast or balanced preset" in text
    assert "shorter video" not in text


@pytest.mark.parametrize("exc", _SIGNED_OUT_REFUSALS, ids=["runs-limit", "larger-than-allowed"])
def test_signed_out_refusals_end_on_a_card_that_asks_to_sign_in(exc):
    title, body = app._friendly_error(backend._classify(exc), str(exc))
    assert "sign in" in f"{title} {body}".lower()


def test_quota_exceeded_card_is_unchanged():
    assert app._FRIENDLY_ERRORS["quota_exceeded"] == (
        "Daily GPU quota used up",
        "You've used today's GPU minutes. Wait for the rolling 24-hour "
        "reset, or upgrade Pro at huggingface.co/subscribe/pro for more.",
    )
