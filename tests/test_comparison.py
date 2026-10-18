"""Tests for the verification-vs-naive comparison harness.

The harness must prove, deterministically, that ORBIT never claims a
resolution the machine did not achieve, while the naive execution-trust
baseline does. All scenarios are synthetic and fully deterministic.
"""
from __future__ import annotations

import pytest

from orbit.actions.environment import MaintenanceEnvironment
from orbit.evaluation.comparison import (
    SCENARIOS,
    NaiveExecutionController,
    format_comparison_table,
    run_comparison,
)
from orbit.simulator.motor import MachineState


@pytest.fixture(scope="module")
def report():
    return run_comparison()


class TestComparisonHarness:
    def test_healthy_control_no_false_alarms(self, report):
        for arm in ("orbit", "naive"):
            row = next(r for r in report["rows"]
                       if r["scenario"] == "healthy" and r["arm"] == arm)
            assert row["incident_opened"] is False
            assert row["final_status"] == "NO_INCIDENT"
            assert row["behavior_correct"] is True

    def test_good_spare_both_arms_recover(self, report):
        for arm in ("orbit", "naive"):
            row = next(r for r in report["rows"]
                       if r["scenario"] == "bearing_good_spare" and r["arm"] == arm)
            assert row["final_status"] == "RESOLVED"
            assert row["ground_truth_recovered"] is True
            assert row["incorrect_resolution"] is False

    def test_defective_spare_naive_claims_false_resolution(self, report):
        row = next(r for r in report["rows"]
                   if r["scenario"] == "bearing_defective_spare"
                   and r["arm"] == "naive")
        assert row["final_status"] == "RESOLVED"
        assert row["ground_truth_recovered"] is False
        assert row["incorrect_resolution"] is True
        assert row["behavior_correct"] is False

    def test_defective_spare_orbit_escalates_instead(self, report):
        row = next(r for r in report["rows"]
                   if r["scenario"] == "bearing_defective_spare"
                   and r["arm"] == "orbit")
        assert row["final_status"] == "ESCALATED"
        assert row["ground_truth_recovered"] is False
        # Escalation of an unrecovered machine is the correct outcome:
        # ORBIT refuses to claim success it cannot prove.
        assert row["correct_escalation"] is True
        assert row["incorrect_resolution"] is False
        assert row["behavior_correct"] is True

    def test_summary_math_with_explicit_denominators(self, report):
        orbit = report["summary"]["orbit"]
        naive = report["summary"]["naive"]
        assert orbit["resolved_claims"] == 1
        assert orbit["incorrect_resolutions"] == 0
        assert orbit["false_resolution_rate"] == 0.0
        assert orbit["escalations"] == orbit["correct_escalations"] == 3
        assert naive["resolved_claims"] == 4
        assert naive["incorrect_resolutions"] == 3
        assert naive["false_resolution_rate"] == 0.75
        assert naive["false_resolution_denominator"] == 4
        assert naive["escalations"] == 0

    def test_table_renders_key_numbers(self, report):
        table = format_comparison_table(report)
        assert "false-resolution rate 0.00" in table
        assert "false-resolution rate 0.75" in table
        assert "Bearing fault, DEFECTIVE spare part" in table
        assert "escalated" in table

    def test_determinism_same_seed_same_report(self):
        subset = SCENARIOS[1:3]
        first = run_comparison(seed=91, scenarios=subset)
        second = run_comparison(seed=91, scenarios=subset)
        keys = ("scenario", "arm", "final_status", "ground_truth_recovered")
        assert [tuple(r[k] for k in keys) for r in first["rows"]] == \
               [tuple(r[k] for k in keys) for r in second["rows"]]

    def test_naive_controller_is_isolated_to_harness(self, report):
        """The naive subclass exists only for the comparison; production
        incident resolution must still require verification."""
        from orbit.incident.controller import IncidentController

        controller = NaiveExecutionController.__new__(NaiveExecutionController)
        outcome = controller._verify_recovery(None, [])
        assert outcome.passed is True
        assert outcome.windows_evaluated == 0
        # The production class keeps the deterministic verifier, not this override.
        assert (NaiveExecutionController._verify_recovery
                is not IncidentController._verify_recovery)

    def test_imbalance_and_overtemperature_naive_failures_recorded(self, report):
        for scenario in ("imbalance_defective_spare",
                         "overtemperature_persistent"):
            naive = next(r for r in report["rows"]
                         if r["scenario"] == scenario and r["arm"] == "naive")
            orbit = next(r for r in report["rows"]
                         if r["scenario"] == scenario and r["arm"] == "orbit")
            assert naive["incorrect_resolution"] is True
            assert orbit["correct_escalation"] is True
