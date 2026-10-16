"""Regression test for a live-demo crash found during the audit (Phase 1).

`orbit/demo/app.py` passed the raw `switch_after` widget value straight to
`service.sensor_trends`, which validates `1 <= switch_after < windows`. A
judge could set "Healthy windows before fault onset" (max 40) at or above
"Trend window count" (min 20) and the page raised ValueError mid-demo (and
would have hit IndexError on the fault-onset caption). The app now clamps
the effective onset index and states that it did so; this test locks both.

Scope: two AppTest renders (defaults, then the crashing combination). No
recovery cycle is executed (the run button is untouched), so no incident
is opened.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "orbit" / "demo" / "app.py"


def test_demo_survives_switch_after_at_or_past_trend_windows():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception, f"demo raised on default settings: {at.exception}"

    # The crashing combination a live judge could select.
    at.number_input(key="switch_after").set_value(40)
    at.number_input(key="trend_windows").set_value(20)
    at.run()
    assert not at.exception, f"demo raised on 40/20: {at.exception}"
    assert any("clamped" in str(c.value) for c in at.caption), (
        "expected the demo to state that the fault-onset index was clamped"
    )


def test_demo_exact_boundary_switch_after_equals_windows():
    # switch_after == windows is the minimal illegal combination (service
    # requires strict <); the clamp must resolve it to windows - 1.
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.number_input(key="switch_after").set_value(40)
    at.number_input(key="trend_windows").set_value(40)
    at.run()
    assert not at.exception, f"demo raised on 40/40: {at.exception}"


def test_demo_pressed_run_renders_result_escalation_and_resolution():
    """Press the run button through the UI for the two headline presets.

    Exercises the full redesigned rendering path (status overview, phase
    tracker, verdict banners, ledger sections) against real service
    results: preset 2 must end RESOLVED with passing verification, preset
    3 must end ESCALATED with a recorded escalation reason.
    """
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception, f"initial render failed: {at.exception}"

    # Preset 2 - successful, independently verified recovery.
    at.selectbox[0].select("2. Successful bearing recovery - independently verified")
    at.run()
    at.button[0].click()
    at.run()
    assert not at.exception, f"pressed run (preset 2) raised: {at.exception}"
    r2 = at.session_state["last_result"]["cycles"][-1]
    assert r2["outcome"] == "RESOLVED", r2["outcome"]
    assert r2["incident"]["verification"]["passed"] is True

    # Preset 3 - defective spare executes but verification fails; escalate.
    at.selectbox[0].select("3. Failed recovery - action runs, verification fails, escalation")
    at.run()
    at.button[0].click()
    at.run()
    assert not at.exception, f"pressed run (preset 3) raised: {at.exception}"
    r3 = at.session_state["last_result"]["cycles"][-1]
    assert r3["outcome"] == "ESCALATED", r3["outcome"]
    assert r3["incident"]["escalation_reason"]
    assert r3["incident"]["verification"]["passed"] is False

    assert len(at.session_state["history"]) == 2
