"""Fail-closed boundary and transient-incident regressions (simulation only)."""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from orbit.actions.environment import MaintenanceEnvironment
from orbit.api import service
from orbit.api.app import app
from orbit.diagnosis.engine import DiagnosisEngine
from orbit.diagnosis.knowledge import ACTION_LIBRARY, VerificationCondition
from orbit.incident.controller import IncidentController
from orbit.incident.lifecycle import InvalidTransition, assert_transition
from orbit.recovery.verifier import verify_recovery
from orbit.simulator.motor import MachineState, MotorSimulator


ENV_FLAGS = (
    "machine_stoppable", "spare_bearing_available",
    "balancing_kit_available", "reduced_load_possible",
)
ENV_NUMBERS = (
    "spare_bearing_quality", "balancing_quality",
    "cooling_service_quality", "maintenance_window_minutes",
)
STATE_NUMBERS = (
    "bearing_severity", "imbalance_severity", "cooling_efficiency", "load",
)
INVALID_NUMBERS = (True, False, "0.5", None, float("nan"), float("inf"), -float("inf"))


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    # Device metadata is unrelated to these boundaries; avoid loading optional
    # weights or probing hardware when the service assembles its response.
    monkeypatch.setattr(service, "device_summary", lambda: {})
    return TestClient(app)


@pytest.mark.parametrize("field", ENV_FLAGS)
@pytest.mark.parametrize("value", ("false", "true", 0, 1, None))
def test_environment_requires_actual_booleans(field, value):
    with pytest.raises(ValueError, match=field):
        MaintenanceEnvironment(**{field: value})


@pytest.mark.parametrize("field", ENV_NUMBERS)
@pytest.mark.parametrize("value", INVALID_NUMBERS)
def test_environment_requires_finite_numbers(field, value):
    with pytest.raises(ValueError, match=field):
        MaintenanceEnvironment(**{field: value})


@pytest.mark.parametrize("field", STATE_NUMBERS)
@pytest.mark.parametrize("value", INVALID_NUMBERS)
def test_machine_state_requires_finite_numbers(field, value):
    with pytest.raises(ValueError, match=field):
        MachineState(**{field: value})


def test_valid_environment_and_state_boundaries_are_preserved():
    env = MaintenanceEnvironment(
        **dict.fromkeys(ENV_FLAGS, False),
        spare_bearing_quality=0, balancing_quality=1,
        cooling_service_quality=0.5, maintenance_window_minutes=0,
    )
    assert all(getattr(env, field) is False for field in ENV_FLAGS)
    state = MachineState(
        bearing_severity=0, imbalance_severity=1, cooling_efficiency=1, load=1.5,
    )
    assert state.copy() == state


@pytest.mark.parametrize("constructor,field,value", [
    (MaintenanceEnvironment, "spare_bearing_quality", -0.1),
    (MaintenanceEnvironment, "balancing_quality", 1.1),
    (MaintenanceEnvironment, "cooling_service_quality", 1.1),
    (MaintenanceEnvironment, "maintenance_window_minutes", -1),
    (MachineState, "bearing_severity", -0.1),
    (MachineState, "imbalance_severity", 1.1),
    (MachineState, "cooling_efficiency", 0),
    (MachineState, "load", 0),
    (MachineState, "load", 1.6),
])
def test_numeric_ranges_remain_enforced(constructor, field, value):
    with pytest.raises(ValueError, match=field):
        constructor(**{field: value})


@pytest.mark.parametrize("field,scenario", [
    ("machine_stoppable", "bearing"),
    ("spare_bearing_available", "bearing"),
    ("balancing_kit_available", "imbalance"),
    ("reduced_load_possible", "overtemperature"),
])
def test_api_false_string_is_rejected_but_real_false_escalates(client, field, scenario):
    payload = {"scenario": scenario, "use_classifier": False, "use_amd": False}
    malformed = client.post("/cycle", json={
        **payload, "environment": {field: "false"},
    })
    assert malformed.status_code == 422
    assert field in malformed.json()["detail"]

    response = client.post("/cycle", json={
        **payload, "environment": {field: False},
    })
    assert response.status_code == 200
    cycle = response.json()["cycles"][0]
    incident = cycle["incident"]
    assert cycle["outcome"] == incident["final_status"] == "ESCALATED"
    assert "precondition" in incident["escalation_reason"]
    assert incident["action_id"] is None
    assert incident["verification"] is None
    assert incident["attempts_used"] == 0
    assert "SIMULATING" not in [phase["phase"] for phase in incident["phases"]]


@pytest.mark.parametrize("section,field", [
    ("environment", "spare_bearing_quality"),
    ("environment", "maintenance_window_minutes"),
    ("machine_state", "bearing_severity"),
    ("machine_state", "imbalance_severity"),
    ("machine_state", "cooling_efficiency"),
    ("machine_state", "load"),
])
@pytest.mark.parametrize("value", (True, "0.5", None))
def test_api_malformed_numeric_payload_returns_422(client, section, field, value):
    response = client.post("/cycle", json={
        "scenario": "bearing", "use_classifier": False, "use_amd": False,
        section: {field: value},
    })
    assert response.status_code == 422
    assert field in response.json()["detail"]


@pytest.mark.parametrize("value", (*INVALID_NUMBERS, -1.0))
def test_invalid_monitor_interval_raises_value_error(config, detector, value):
    with pytest.raises(ValueError, match="monitor_interval_s"):
        IncidentController(
            MotorSimulator(config), detector, DiagnosisEngine(), monitor_interval_s=value,
        )


@pytest.mark.parametrize("value", (0, 0.5, 60))
def test_valid_monitor_interval_is_accepted(config, detector, value):
    controller = IncidentController(
        MotorSimulator(config), detector, DiagnosisEngine(), monitor_interval_s=value,
    )
    assert controller.monitor_interval_s == value


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
@pytest.mark.parametrize("target", ("observation", "threshold"))
@pytest.mark.parametrize("operator", ("<", "<="))
def test_verifier_rejects_nonfinite_inputs_before_evaluation(monkeypatch, value, target, operator):
    action = ACTION_LIBRARY["replace_bearing"]
    conditions = tuple(replace(c, operator=operator) for c in action.verification)
    rows = [
        {"env_bpfo_ratio": 1.0, "vib_kurtosis": 2.0},
        {"env_bpfo_ratio": 1.0, "vib_kurtosis": 2.0},
    ]
    if target == "threshold":
        conditions = (conditions[0], replace(conditions[1], threshold=value))
    else:
        rows[-1]["vib_kurtosis"] = value
    action = replace(action, verification=conditions)
    evaluate = Mock(return_value=True)
    monkeypatch.setattr(VerificationCondition, "evaluate", evaluate)
    with pytest.raises(ValueError, match="finite"):
        verify_recovery(action, rows)
    evaluate.assert_not_called()


def test_transient_diagnosis_escalates_without_unverified_resolution(config, detector, monkeypatch):
    class AmbiguousEngine(DiagnosisEngine):
        def diagnose(self, anomaly, features, detector):
            base = super().diagnose(anomaly, features, detector)
            return replace(
                base, is_confident=False, primary=None, recommended_action=None,
                ambiguity_note="stub: hypotheses remain tied",
            )

    sim = MotorSimulator(config, seed=53)
    sim.advance(600.0)
    controller = IncidentController(sim, detector, AmbiguousEngine(), evidence_rounds=2)
    healthy = controller._sample_features()
    assert not healthy[1].is_anomaly
    sim.set_state(MachineState(bearing_severity=0.6))
    faulty = controller._sample_features()
    assert faulty[1].is_anomaly
    samples = Mock(side_effect=[faulty, healthy])
    monkeypatch.setattr(controller, "_sample_features", samples)
    execute = Mock(side_effect=AssertionError("transient must not trigger intervention"))
    monkeypatch.setattr(controller.executor, "execute", execute)

    result = controller.run_cycle()
    incident = result.incident
    assert result.anomaly_detected
    assert result.outcome == incident.final_status == "ESCALATED"
    assert "unverified" in incident.escalation_reason
    assert incident.diagnosis is not None and not incident.diagnosis.is_confident
    assert incident.action_id is None
    assert incident.verification is None
    assert incident.attempts_used == 0
    assert [phase.phase for phase in incident.phases] == [
        "DETECTED", "DIAGNOSING", "DIAGNOSING", "ESCALATED", "MONITORING",
    ]
    assert controller.state == "MONITORING"
    assert samples.call_count == 2
    execute.assert_not_called()
    with pytest.raises(InvalidTransition):
        assert_transition("DIAGNOSING", "RESOLVED")
