"""Milestone 5 tests: FastAPI endpoints + service orchestration.

Covers API request validation, the stateless service surface, and the
full observe -> verify loop returning an honest audit trail. No GPU or
trained weights are required (corroboration is optional)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orbit.api.app import app
from orbit.api import service


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health_and_scenarios(client):
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/scenarios").json()
    assert set(body["scenarios"]) == {"normal", "bearing", "imbalance", "overtemperature"}
    assert "spare_bearing_available" in body["environment_fields"]


def test_device_endpoint_reports_platform(client):
    body = client.get("/device").json()
    assert body["environment"]["platform"] in {"cpu", "rocm", "cuda", "mps"}
    assert "classifier" in body


def test_trends_valid_and_invalid(client):
    ok = client.get("/trends", params={"scenario": "bearing", "windows": 30, "switch_after": 10})
    assert ok.status_code == 200
    rows = ok.json()["windows"]
    assert len(rows) == 30
    assert rows[0]["state"] == "healthy"
    assert rows[10]["state"] == "fault"

    assert client.get("/trends", params={"scenario": "nope"}).status_code == 422
    # switch_after must be < windows
    bad = client.get("/trends", params={"scenario": "bearing", "windows": 10, "switch_after": 10})
    assert bad.status_code == 422


def test_cycle_resolves_bearing(client):
    resp = client.post("/cycle", json={"scenario": "bearing", "cycles": 1,
                                        "use_classifier": False})
    assert resp.status_code == 200
    result = resp.json()
    incident = result["cycles"][0]["incident"]
    assert result["cycles"][0]["outcome"] == "RESOLVED"
    assert incident["final_status"] == "RESOLVED"
    assert result["still_anomalous"] is False
    phases = [p["phase"] for p in incident["phases"]]
    # RESOLVED is the last business-state transition; MONITORING follows
    # when the controller hands control back to the monitor loop.
    assert "VERIFYING" in phases
    business = [p for p in phases if p != "MONITORING"]
    assert business[-1] == "RESOLVED"


def test_cycle_healthy_no_action(client):
    resp = client.post("/cycle", json={"scenario": "normal", "use_classifier": False})
    assert resp.status_code == 200
    result = resp.json()
    assert result["cycles"][0]["outcome"] == "HEALTHY_NO_ACTION"
    assert result["cycles"][0]["incident"] is None


def test_cycle_defective_spare_escalates(client):
    resp = client.post("/cycle", json={
        "scenario": "bearing", "use_classifier": False,
        "environment": {"spare_bearing_quality": 0.0},
    })
    assert resp.status_code == 200
    assert resp.json()["cycles"][0]["outcome"] == "ESCALATED"


def test_cycle_rejects_unknown_environment_field(client):
    resp = client.post("/cycle", json={
        "scenario": "bearing", "environment": {"bogus_field": 1}})
    assert resp.status_code == 422


def test_cycle_rejects_out_of_range(client):
    # pydantic bounds: cycles <= 5
    resp = client.post("/cycle", json={"scenario": "bearing", "cycles": 9})
    assert resp.status_code == 422


def test_service_state_is_deterministic():
    a = service.run_recovery_cycle("imbalance", use_classifier=False)
    b = service.run_recovery_cycle("imbalance", use_classifier=False)
    assert a["features_before"] == b["features_before"]
    assert a["cycles"][0]["outcome"] == b["cycles"][0]["outcome"]


def test_custom_machine_state_resolves(client):
    resp = client.post("/cycle", json={
        "scenario": "bearing", "use_classifier": False,
        "machine_state": {"bearing_severity": 0.6},
    })
    assert resp.status_code == 200
    result = resp.json()
    assert result["custom_state"] is True
    assert result["state_before"]["bearing_severity"] == 0.6
    assert result["cycles"][0]["outcome"] == "RESOLVED"


def test_custom_concurrent_state_rediagnoses(client):
    # Concurrent bearing + imbalance: the dominant fault is repaired first,
    # the residual is re-diagnosed across a second cycle.
    resp = client.post("/cycle", json={
        "scenario": "bearing", "cycles": 2, "use_classifier": False,
        "machine_state": {"bearing_severity": 0.25, "imbalance_severity": 0.55},
    })
    assert resp.status_code == 200
    result = resp.json()
    assert result["custom_state"] is True
    assert result["cycles"][-1]["outcome"] == "RESOLVED"


def test_custom_machine_state_rejects_unknown_field(client):
    resp = client.post("/cycle", json={
        "scenario": "bearing", "machine_state": {"warp_factor": 9}})
    assert resp.status_code == 422


def test_custom_machine_state_rejects_out_of_range(client):
    resp = client.post("/cycle", json={
        "scenario": "bearing", "machine_state": {"bearing_severity": 1.7}})
    assert resp.status_code == 422


def test_custom_machine_state_rejects_empty(client):
    resp = client.post("/cycle", json={"scenario": "bearing", "machine_state": {}})
    assert resp.status_code == 422
