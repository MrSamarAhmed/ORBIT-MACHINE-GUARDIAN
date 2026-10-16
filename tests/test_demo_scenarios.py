"""Demonstration-scenario tests: every fault-injection preset must produce
its documented outcome deterministically, and the safety invariants must
hold on the real service path (no live API credentials, no AMD hardware).

Reuses the existing service and controller; no new framework or fixtures
beyond the deterministic simulator defaults.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orbit.api import service
from orbit.demo.scenarios import DEMO_SCENARIOS, incident_timeline
from orbit.model.inference import WEIGHTS_PATH


@pytest.mark.parametrize("preset", DEMO_SCENARIOS, ids=[s.key for s in DEMO_SCENARIOS])
def test_preset_produces_expected_outcome(preset):
    if preset.needs_classifier and not Path(WEIGHTS_PATH).exists():
        pytest.skip("trained weights unavailable; neural corroboration is optional")

    result = service.run_recovery_cycle(**preset.kwargs)
    assert result["cycles"][-1]["outcome"] == preset.expected_outcome

    # Before/after machine state is always available for the judge.
    assert result["state_before"] and result["state_after"]

    for cyc in result["cycles"]:
        inc = cyc.get("incident")
        if not inc:
            continue
        # Invariant B: RESOLVED only when the independent verifier passed.
        if inc["final_status"] == "RESOLVED":
            assert inc["verification"]["passed"] is True
        # Invariant C: escalation always carries a real reason.
        if inc["final_status"] == "ESCALATED":
            assert inc["escalation_reason"]
        # Invariant A + traceability: executed actions record the actual result.
        for phase in inc["phases"]:
            if phase["phase"] == "SIMULATING":
                ev = phase["evidence"]
                assert ev["executed"] in (True, False)
                assert "state_before" in ev and "state_after" in ev


def test_insufficient_evidence_preset_is_not_confident():
    preset = next(s for s in DEMO_SCENARIOS if s.key == "insufficient_evidence")
    result = service.run_recovery_cycle(**preset.kwargs)
    inc = result["cycles"][0]["incident"]
    # No invented certainty: diagnosis stays ambiguous and no action is taken.
    assert inc["diagnosis"]["confident"] is False
    assert inc["diagnosis"]["ambiguity_note"]
    assert not any(p["phase"] == "SIMULATING" for p in inc["phases"])


def test_conflicting_evidence_preset_records_disagreement():
    if not Path(WEIGHTS_PATH).exists():
        pytest.skip("trained weights unavailable; neural corroboration is optional")
    preset = next(s for s in DEMO_SCENARIOS if s.key == "conflicting_evidence")
    result = service.run_recovery_cycle(**preset.kwargs)
    inc = result["cycles"][0]["incident"]
    assert inc["corroboration"] is not None
    assert inc["corroboration"]["agrees_with_diagnosis"] is False
    # Invariant F: disagreement never bypasses deterministic verification.
    assert inc["verification"]["passed"] is True
    assert inc["final_status"] == "RESOLVED"


def test_timeline_marks_resolution_from_real_phases():
    result = service.run_recovery_cycle(scenario="bearing", use_classifier=False)
    inc = result["cycles"][0]["incident"]
    rows = incident_timeline(inc["phases"])
    reached = {r["stage"]: r["reached"] for r in rows}
    assert all(reached[s] for s in (
        "OBSERVED", "DIAGNOSED", "PLAN SELECTED",
        "ACTION EXECUTED", "INDEPENDENT VERIFICATION"))
    assert rows[-1]["recorded_phase"] == "RESOLVED"


def test_timeline_shows_executed_but_escalated():
    """The core differentiator: an executed action is not a resolved incident."""
    result = service.run_recovery_cycle(
        scenario="bearing", environment={"spare_bearing_quality": 0.0},
        use_classifier=False)
    inc = result["cycles"][0]["incident"]
    rows = incident_timeline(inc["phases"])
    reached = {r["stage"]: r["reached"] for r in rows}
    assert reached["ACTION EXECUTED"] is True
    assert reached["INDEPENDENT VERIFICATION"] is True
    assert rows[-1]["recorded_phase"] == "ESCALATED"
    assert inc["final_status"] == "ESCALATED"
