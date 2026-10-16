"""Recovery Evidence Ledger tests.

The ledger is a read-only consolidation of the recorded incident data for
the demo UI. These tests verify that it reports only recorded evidence,
marks gaps explicitly, mirrors the controller's authoritative decisions,
and never alters the underlying incident record. No Streamlit internals
are exercised; only the pure builder is tested.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from orbit.api import service
from orbit.demo.scenarios import build_incident_ledger
from orbit.model.inference import WEIGHTS_PATH


def test_ledger_resolved_incident_reports_full_chain():
    result = service.run_recovery_cycle(scenario="bearing", use_classifier=False)
    inc = result["cycles"][0]["incident"]
    ledger = build_incident_ledger(inc, scenario=result["scenario"])
    assert ledger["incident_id"] == inc["incident_id"]
    assert ledger["opened_at_utc"]  # real timestamp recorded when the incident opened
    assert ledger["scenario"] == "bearing"
    assert ledger["observed_anomalies"] is not None
    assert ledger["diagnosis"] is not None
    assert ledger["planned_action"] is not None
    assert ledger["action_attempted"] is True
    assert ledger["action_executed"] is True
    assert ledger["verification_ran"] is True
    assert ledger["verification_passed"] is True
    assert ledger["final_status"] == "RESOLVED"
    assert ledger["reason"]
    for field in ("opened_at_utc", "verification", "observed_anomalies"):
        assert field not in ledger["unavailable_fields"]


def test_ledger_rows_match_verifier_results():
    result = service.run_recovery_cycle(scenario="bearing", use_classifier=False)
    inc = result["cycles"][0]["incident"]
    ledger = build_incident_ledger(inc, scenario="bearing")
    ref = inc["verification"]["results"]
    assert len(ledger["verification_rows"]) == len(ref)
    for row, r in zip(ledger["verification_rows"], ref):
        assert row["observed"] == r["value"]
        assert row["decision"] == ("PASS" if r["passed"] else "FAIL")
        assert str(r["threshold"]) in row["expected"]
        assert row["window"] == r["window"]


def test_ledger_executed_action_without_verified_recovery():
    result = service.run_recovery_cycle(
        scenario="bearing", environment={"spare_bearing_quality": 0.0},
        use_classifier=False)
    inc = result["cycles"][0]["incident"]
    ledger = build_incident_ledger(inc, scenario="bearing")
    # Executed does NOT imply resolved: verification failed -> escalated.
    assert ledger["action_attempted"] is True
    assert ledger["action_executed"] is True
    assert ledger["verification_ran"] is True
    assert ledger["verification_passed"] is False
    assert ledger["final_status"] == "ESCALATED"
    assert ledger["reason"] == inc["escalation_reason"]
    assert any(r["decision"] == "FAIL" for r in ledger["verification_rows"])


def test_ledger_insufficient_evidence_marks_absence_explicitly():
    result = service.run_recovery_cycle(
        scenario="bearing",
        machine_state={"bearing_severity": 0.15, "imbalance_severity": 0.15},
        use_classifier=False)
    inc = result["cycles"][0]["incident"]
    ledger = build_incident_ledger(inc, scenario="bearing")
    assert ledger["action_attempted"] is False
    assert ledger["action_executed"] is False
    assert ledger["verification_ran"] is False
    assert ledger["verification_passed"] is None
    assert "verification" in ledger["unavailable_fields"]
    assert "action_execution" in ledger["unavailable_fields"]
    assert ledger["final_status"] == "ESCALATED"


def test_ledger_missing_classifier_is_unavailable_not_invented():
    result = service.run_recovery_cycle(scenario="overtemperature", use_classifier=False)
    inc = result["cycles"][0]["incident"]
    ledger = build_incident_ledger(inc, scenario="overtemperature")
    assert ledger["corroboration"] is None
    assert "neural_corroboration" in ledger["unavailable_fields"]
    # The deterministic verification is unaffected by the missing classifier.
    assert ledger["verification_passed"] is True


def test_ledger_conflicting_evidence_records_disagreement():
    if not Path(WEIGHTS_PATH).exists():
        pytest.skip("trained weights unavailable; neural corroboration is optional")
    result = service.run_recovery_cycle(scenario="overtemperature", use_classifier=True)
    inc = result["cycles"][0]["incident"]
    ledger = build_incident_ledger(inc, scenario="overtemperature")
    assert ledger["corroboration"]["agrees_with_diagnosis"] is False
    assert "neural_corroboration" not in ledger["unavailable_fields"]
    # The controller's decision stands: verification governs.
    assert ledger["verification_passed"] is True
    assert ledger["final_status"] == "RESOLVED"


def test_ledger_does_not_modify_incident():
    result = service.run_recovery_cycle(scenario="bearing", use_classifier=False)
    inc = result["cycles"][0]["incident"]
    snapshot = copy.deepcopy(inc)
    build_incident_ledger(inc, scenario="bearing")
    assert inc == snapshot


def test_ledger_handles_sparse_incident_dict():
    ledger = build_incident_ledger({"incident_id": "INC-999", "phases": []})
    assert ledger["final_status"] is None
    assert ledger["verification_ran"] is False
    assert ledger["action_attempted"] is False
    for field in ("final_status", "verification", "observed_anomalies",
                  "diagnosis", "planned_action", "scenario", "opened_at_utc"):
        assert field in ledger["unavailable_fields"]
