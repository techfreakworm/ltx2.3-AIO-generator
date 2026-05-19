"""Tests for the ZeroGPU per-call duration estimator + user-budget override."""

import backend


def _t2v_workflow(frames: int = 121) -> dict:
    return {
        "100": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"length": frames, "width": 512, "height": 512},
        }
    }


def test_duration_for_uses_user_budget_when_set() -> None:
    # 600s should pass through verbatim — the user knows what they're spending.
    assert (
        backend._duration_for(
            executor=None,
            workflow=_t2v_workflow(),
            output_ids=[],
            mode="t2v",
            preset="fast",
            user_budget=600,
        )
        == 600
    )


def test_duration_for_clamps_user_budget_to_floor() -> None:
    # 30s below the 60s ZeroGPU floor — clamp up, never below.
    assert (
        backend._duration_for(
            executor=None,
            workflow=_t2v_workflow(),
            output_ids=[],
            mode="t2v",
            preset="fast",
            user_budget=30,
        )
        == 60
    )


def test_duration_for_no_budget_returns_unclamped_estimate() -> None:
    # style/quality/121 frames: 360*3 + 60 + 121*0.3 = 1176.3 -> int 1176.
    # No upper ceiling — the whole point of the user-budget refactor.
    result = backend._duration_for(
        executor=None,
        workflow=_t2v_workflow(frames=121),
        output_ids=[],
        mode="style",
        preset="quality",
    )
    assert result == 1176


def test_duration_for_no_budget_honours_floor() -> None:
    # 1-frame t2v/fast: 90*1 + 60 + 0.3 = 150 -> int 150; well above floor, so
    # this is really testing that the floor doesn't accidentally fire on real
    # workloads. (See test_duration_for_clamps_user_budget_to_floor for the
    # actual floor case via user_budget.)
    result = backend._duration_for(
        executor=None,
        workflow=_t2v_workflow(frames=1),
        output_ids=[],
        mode="t2v",
        preset="fast",
    )
    assert result == 150


def test_estimate_duration_unclamped_matches_formula() -> None:
    # Surface the formula so the pre-flight gate in app.py can show the user
    # "needs X seconds" without re-implementing it.
    assert backend._estimate_duration_unclamped(mode="t2v", preset="fast", frames=121) == 90 + 60 + int(121 * 0.3)
    assert backend._estimate_duration_unclamped(mode="style", preset="quality", frames=121) == int(360 * 3.0 + 60 + 121 * 0.3)


def test_estimate_duration_unclamped_unknown_mode_uses_default() -> None:
    # Unknown mode -> default base 180. Preset still applies.
    assert backend._estimate_duration_unclamped(mode="nonsense", preset="balanced", frames=100) == int(180 * 1.5 + 60 + 100 * 0.3)
