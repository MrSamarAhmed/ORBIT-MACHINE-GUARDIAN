"""AMD Token Factory advisory tests — fully mocked, never a live API call.

Covers the advisor client (schema validation, failure handling, redaction)
and the controller integration (the advisory is recorded but never
selects the action, never executes, and can never declare recovery).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from orbit.actions.environment import MaintenanceEnvironment
from orbit.amd.advisor import AmdAdvisor, build_evidence_payload, validate_assessment
from orbit.amd.config import AmdConfig
from orbit.demo.scenarios import build_incident_ledger
from orbit.diagnosis.engine import DiagnosisEngine
from orbit.diagnosis.knowledge import ACTION_LIBRARY
from orbit.incident.controller import IncidentController
from orbit.signalproc.features import FEATURE_NAMES
from orbit.simulator.motor import MachineState, MotorSimulator
from tests.conftest import SCENARIOS_STATE

CONFIG = AmdConfig(api_key="sk-radeon-test-key-000")


# ---------------------------------------------------------------------------
# Fakes (no network, no real OpenAI client)
# ---------------------------------------------------------------------------
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(_Msg(content))]


class _FakeCompletions:
    def __init__(self, outcome):
        self._outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return _Response(self._outcome)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, outcome):
        self.chat = _FakeChat(_FakeCompletions(outcome))


def _advisor_with(outcome, monkeypatch) -> tuple[AmdAdvisor, _FakeCompletions]:
    advisor = AmdAdvisor(CONFIG)
    fake = _FakeCompletions(outcome)
    monkeypatch.setattr(
        advisor, "_build_client", lambda: SimpleNamespace(chat=_FakeChat(fake))
    )
    return advisor, fake


def _payload() -> dict:
    return {
        "incident_id": "INC-001",
        "sensor_features": {"vib_rms": 0.42, "env_bpfo_ratio": 6.1, "temp_c": 55.0},
        "anomaly_detector": {"is_anomaly": True, "max_abs_z": 8.1,
                             "temp_limit_exceeded": False, "breaches": []},
        "neural_classifier": None,
        "evidence_trend": [],
        "orbit_diagnosis": {"confident": True, "primary": "Bearing outer-race defect"},
        "permitted_actions": sorted(ACTION_LIBRARY),
    }


def _successful_content(**overrides) -> str:
    body = {
        "assessment": "Bearing outer-race defect signature is dominant.",
        "supporting_evidence": ["env_bpfo_ratio breach at z=8.1",
                                "vib_kurtosis breach indicates impacts"],
        "recommended_action": "replace_bearing",
        "confidence": "high",
        "needs_escalation": False,
    }
    body.update(overrides)
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestAmdConfig:
    def test_missing_key_returns_none(self, tmp_path):
        assert AmdConfig.from_env(env={}, dotenv_path=tmp_path / "no.env") is None

    def test_env_key_builds_default_config(self, tmp_path):
        cfg = AmdConfig.from_env(env={"RADEON_API_KEY": "sk-abc"},
                                 dotenv_path=tmp_path / "no.env")
        assert cfg is not None
        assert cfg.model == "DeepSeek-V4-Flash"
        assert cfg.base_url == "https://developer.amd.com.cn/radeon/api/v1"

    def test_env_takes_precedence_over_dotenv(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('RADEON_API_KEY="sk-from-file"\n'
                            'RADEON_MODEL=Other-Model\n', encoding="utf-8")
        cfg = AmdConfig.from_env(env={"RADEON_API_KEY": "sk-from-env"},
                                 dotenv_path=env_file)
        assert cfg is not None
        assert cfg.api_key == "sk-from-env"
        assert cfg.model == "Other-Model"  # file values still apply for non-secret keys

    def test_redacted_never_exposes_key(self):
        identity = CONFIG.redacted()
        assert identity["api_key"] == "[REDACTED]"
        assert "sk-radeon-test-key-000" not in json.dumps(identity)

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            AmdConfig(api_key="")
        with pytest.raises(ValueError):
            AmdConfig(api_key="sk-x", timeout_s=0)
        with pytest.raises(ValueError):
            AmdConfig(api_key="sk-x", base_url="ftp://nope")


# ---------------------------------------------------------------------------
# Response schema validation
# ---------------------------------------------------------------------------
class TestValidateAssessment:
    def test_valid_string_confidence(self):
        parsed, error = validate_assessment(json.loads(_successful_content()))
        assert error is None
        assert parsed["confidence"] == "high"
        assert parsed["action_supported"] is True

    def test_recommended_action_none_is_supported(self):
        parsed, error = validate_assessment(
            json.loads(_successful_content(recommended_action="none")))
        assert error is None
        assert parsed["action_supported"] is True

    def test_unsupported_action_flagged_not_rewritten(self):
        parsed, error = validate_assessment(
            json.loads(_successful_content(recommended_action="swap_the_rotor")))
        assert error is None
        assert parsed["action_supported"] is False
        assert parsed["recommended_action"] == "swap_the_rotor"

    @pytest.mark.parametrize("value,expected", [
        (0.93, "high"), (0.5, "medium"), (0.1, "low"),
    ])
    def test_numeric_confidence_is_bucketed(self, value, expected):
        parsed, error = validate_assessment(
            json.loads(_successful_content(confidence=value)))
        assert error is None
        assert parsed["confidence"] == expected

    def test_unknown_confidence_label_rejected(self):
        _, error = validate_assessment(
            json.loads(_successful_content(confidence="certain")))
        assert error and "not in" in error

    def test_out_of_range_numeric_confidence_rejected(self):
        _, error = validate_assessment(
            json.loads(_successful_content(confidence=1.5)))
        assert error and "outside [0, 1]" in error

    def test_needs_escalation_must_be_boolean(self):
        _, error = validate_assessment(
            json.loads(_successful_content(needs_escalation="yes")))
        assert error and "boolean" in error

    def test_missing_and_unexpected_fields_rejected(self):
        _, error = validate_assessment({"assessment": "only this"})
        assert error and "missing required field" in error
        complete = json.loads(_successful_content())
        complete["surprise"] = 1
        _, error = validate_assessment(complete)
        assert error and "unexpected field" in error

    def test_non_object_and_bad_evidence_rejected(self):
        _, error = validate_assessment("a string")
        assert error and "JSON object" in error
        _, error = validate_assessment(
            json.loads(_successful_content(supporting_evidence=["ok", 42])))
        assert error and "non-empty string" in error
        _, error = validate_assessment(
            json.loads(_successful_content(supporting_evidence=[])))
        assert error and "non-empty array" in error

    def test_oversized_assessment_rejected(self):
        _, error = validate_assessment(
            json.loads(_successful_content(assessment="x" * 2001)))
        assert error and "2000" in error


# ---------------------------------------------------------------------------
# Advisor client (mocked transport)
# ---------------------------------------------------------------------------
class TestAmdAdvisorAssess:
    def test_successful_assessment(self, monkeypatch):
        advisor, fake = _advisor_with(_successful_content(), monkeypatch)
        result = advisor.assess(_payload())
        assert result["available"] is True
        assert result["recommended_action"] == "replace_bearing"
        assert result["needs_escalation"] is False
        assert result["recovery_declared"] is False
        assert result["confidence_basis"].startswith("model-reported")
        # Exactly one call, correct model, retries disabled upstream.
        assert len(fake.calls) == 1
        assert fake.calls[0]["model"] == "DeepSeek-V4-Flash"
        # The evidence payload reached the transport unchanged.
        sent = json.loads(fake.calls[0]["messages"][1]["content"])
        assert sent["incident_id"] == "INC-001"

    def test_code_fenced_json_is_parsed(self, monkeypatch):
        advisor, _ = _advisor_with("```json\n" + _successful_content() + "\n```",
                                   monkeypatch)
        result = advisor.assess(_payload())
        assert result["available"] is True

    def test_json_with_surrounding_prose_is_parsed(self, monkeypatch):
        advisor, _ = _advisor_with(
            "Here is my analysis:\n" + _successful_content() + "\nHope that helps.",
            monkeypatch)
        result = advisor.assess(_payload())
        assert result["available"] is True

    def test_timeout_is_reported_not_raised(self, monkeypatch):
        advisor, _ = _advisor_with(TimeoutError("timed out"), monkeypatch)
        result = advisor.assess(_payload())
        assert result["available"] is False
        assert result["reason"] == "timeout"
        assert result["recovery_declared"] is False

    def test_http_error_is_reported_not_raised(self, monkeypatch):
        class ApiError(Exception):
            pass
        advisor, _ = _advisor_with(ApiError("429 quota exceeded"), monkeypatch)
        result = advisor.assess(_payload())
        assert result["available"] is False
        assert result["reason"] in ("http_error", "api_error")

    def test_invalid_json_rejected(self, monkeypatch):
        advisor, _ = _advisor_with("not json at all", monkeypatch)
        result = advisor.assess(_payload())
        assert result["available"] is False
        assert result["reason"] == "invalid_json"

    def test_invalid_schema_rejected(self, monkeypatch):
        advisor, _ = _advisor_with('{"assessment": "partial answer only"}', monkeypatch)
        result = advisor.assess(_payload())
        assert result["available"] is False
        assert result["reason"] == "invalid_schema"

    def test_unexpected_response_shape_rejected(self, monkeypatch):
        advisor = AmdAdvisor(CONFIG)
        # Transport response lacking any choices.
        fake = _FakeCompletions("ignored")
        fake.create = lambda **kw: SimpleNamespace(choices=[])
        monkeypatch.setattr(
            advisor, "_build_client",
            lambda: SimpleNamespace(chat=_FakeChat(fake)))
        result = advisor.assess(_payload())
        assert result["available"] is False
        assert result["reason"] == "unexpected_response"

    def test_api_key_never_appears_in_failure_details(self, monkeypatch):
        class Boom(Exception):
            pass
        advisor, _ = _advisor_with(
            Boom("request failed with key sk-radeon-test-key-000 embedded"),
            monkeypatch)
        result = advisor.assess(_payload())
        assert "sk-radeon-test-key-000" not in json.dumps(result)


# ---------------------------------------------------------------------------
# Evidence payload (pure function, real ORBIT schemas)
# ---------------------------------------------------------------------------
class TestEvidencePayload:
    def test_payload_uses_real_recorded_values(self, config, detector):
        sim = MotorSimulator(config, seed=7)
        sim.advance(600.0)
        sim.set_state(SCENARIOS_STATE["bearing"])
        from orbit.signalproc.features import extract_features
        features = extract_features(sim.sample_window(), config)
        anomaly = detector.predict(features)
        diagnosis = DiagnosisEngine().diagnose(anomaly, features, detector)

        payload = build_evidence_payload(
            incident_id="INC-009",
            features=features,
            anomaly=anomaly,
            diagnosis=diagnosis,
            corroboration=None,
            evidence_trend=[],
        )
        assert set(payload["sensor_features"]) == set(FEATURE_NAMES)
        assert payload["anomaly_detector"]["is_anomaly"] is True
        assert payload["anomaly_detector"]["breaches"]
        assert payload["orbit_diagnosis"]["confident"] is True
        assert payload["permitted_actions"] == sorted(ACTION_LIBRARY)
        assert payload["neural_classifier"] is None
        assert payload["incident_id"] == "INC-009"


# ---------------------------------------------------------------------------
# Controller integration — the advisory never decides
# ---------------------------------------------------------------------------
class FakeAdvisor:
    """Scripted advisor: returns a canned result or raises. Records payloads."""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.payloads = []
        self.config = SimpleNamespace(model="DeepSeek-V4-Flash")

    def assess(self, payload):
        self.payloads.append(payload)
        if self._error is not None:
            raise self._error
        return dict(self._result)


def _advisory(**overrides):
    base = {
        "available": True,
        "provider": "amd_token_factory",
        "model": "DeepSeek-V4-Flash",
        "assessment": "Bearing outer-race defect signature is dominant.",
        "supporting_evidence": ["env_bpfo_ratio breach"],
        "recommended_action": "replace_bearing",
        "action_supported": True,
        "confidence": "high",
        "confidence_reported": "high",
        "confidence_basis": "model-reported; advisory only",
        "needs_escalation": False,
        "diagnosis_produced": False,
        "recovery_declared": False,
    }
    base.update(overrides)
    return base


class AdvisoryControllerFactory:
    def __init__(self, config, detector):
        self._config, self._detector = config, detector

    def make(self, advisor=None, state="bearing", env=None):
        sim = MotorSimulator(self._config, seed=53)
        sim.advance(600.0)
        sim.set_state(SCENARIOS_STATE[state])
        return IncidentController(
            simulator=sim,
            detector=self._detector,
            engine=DiagnosisEngine(),
            environment=env if env is not None else MaintenanceEnvironment(),
            amd_advisor=advisor,
            config=self._config,
        )


@pytest.fixture()
def advisory_factory(config, detector):
    return AdvisoryControllerFactory(config, detector)


class TestControllerAdvisory:
    def test_successful_advisory_recorded_but_verifier_decides(
            self, advisory_factory):
        advisor = FakeAdvisor(_advisory())
        controller = advisory_factory.make(advisor=advisor)
        result = controller.run_cycle()
        incident = result.incident
        assert incident.final_status == "RESOLVED"
        # Exactly one API call, carrying real evidence.
        assert len(advisor.payloads) == 1
        payload = advisor.payloads[0]
        assert payload["incident_id"] == incident.incident_id
        assert payload["orbit_diagnosis"]["confident"] is True
        assert payload["permitted_actions"] == sorted(ACTION_LIBRARY)
        # Advisory agrees; recorded in the incident and the plan evidence.
        assert incident.amd_advisory["agrees_with_plan"] is True
        plan = next(p for p in incident.phases if p.phase == "ACTION_PLANNED")
        assert plan.evidence["amd_advisory"]["available"] is True
        # No new lifecycle states: phase sequence is unchanged.
        assert [p.phase for p in incident.phases] == [
            "DETECTED", "DIAGNOSING", "ACTION_PLANNED",
            "SIMULATING", "VERIFYING", "RESOLVED", "MONITORING",
        ]

    def test_advisory_escalation_request_is_honoured_conservatively(
            self, advisory_factory):
        advisor = FakeAdvisor(_advisory(needs_escalation=True,
                                        assessment="evidence conflicts"))
        controller = advisory_factory.make(advisor=advisor)
        result = controller.run_cycle()
        incident = result.incident
        assert incident.final_status == "ESCALATED"
        assert "AMD advisory" in incident.escalation_reason
        # Conservative outcome: no simulated intervention was executed.
        assert not any(p.phase == "SIMULATING" for p in incident.phases)
        assert incident.action_id is None

    def test_unsupported_recommended_action_is_recorded_not_executed(
            self, advisory_factory):
        advisor = FakeAdvisor(_advisory(recommended_action="rebuild_the_rotor",
                                        action_supported=False))
        controller = advisory_factory.make(advisor=advisor)
        result = controller.run_cycle()
        incident = result.incident
        # ORBIT's deterministic action governed; the model's was not executed.
        assert incident.action_id == "replace_bearing"
        assert incident.final_status == "RESOLVED"
        assert incident.amd_advisory["action_supported"] is False
        assert incident.amd_advisory["agrees_with_plan"] is False

    def test_disagreement_with_supported_action_leaves_plan_governing(
            self, advisory_factory):
        advisor = FakeAdvisor(_advisory(recommended_action="field_balancing"))
        controller = advisory_factory.make(advisor=advisor)
        result = controller.run_cycle()
        incident = result.incident
        assert incident.action_id == "replace_bearing"
        assert incident.final_status == "RESOLVED"
        assert incident.amd_advisory["agrees_with_plan"] is False

    def test_advisor_exception_falls_back_to_deterministic_path(
            self, advisory_factory):
        advisor = FakeAdvisor(error=RuntimeError("advisor blew up"))
        controller = advisory_factory.make(advisor=advisor)
        result = controller.run_cycle()
        incident = result.incident
        assert incident.final_status == "RESOLVED"
        assert incident.amd_advisory["available"] is False
        assert incident.amd_advisory["reason"] == "advisor_unexpected_error"

    def test_api_failure_is_never_a_diagnosis_or_recovery(
            self, advisory_factory):
        advisor = FakeAdvisor(_advisory(available=False, reason="timeout",
                                        model="DeepSeek-V4-Flash"))
        controller = advisory_factory.make(advisor=advisor)
        result = controller.run_cycle()
        incident = result.incident
        assert incident.final_status == "RESOLVED"
        assert incident.amd_advisory["available"] is False
        assert incident.amd_advisory["recovery_declared"] is False
        # RESOLVED came from the verifier, not the advisory.
        assert incident.verification.passed is True

    def test_amd_cannot_declare_recovery_when_verification_fails(
            self, advisory_factory):
        """THE safety gate: defective spare + optimistic advisory must still
        escalate because only the deterministic verifier may say RESOLVED."""
        advisor = FakeAdvisor(_advisory(
            assessment="signature should clear after replacement",
            confidence="high", needs_escalation=False,
            recovery_declared=True,  # even a bogus claim is ignored
        ))
        env = MaintenanceEnvironment(spare_bearing_quality=0.0)
        controller = advisory_factory.make(advisor=advisor, env=env)
        result = controller.run_cycle()
        incident = result.incident
        assert incident.final_status == "ESCALATED"
        assert incident.verification.passed is False
        assert "verification failed" in incident.escalation_reason
        # The advisory was executed (action ran) but its optimism decided nothing.
        assert incident.amd_advisory["recovery_declared"] is True
        assert any(p.phase == "RESOLVED" for p in incident.phases) is False

    def test_verification_failure_reason_not_masked_by_advisory(
            self, advisory_factory):
        advisor = FakeAdvisor(_advisory())
        env = MaintenanceEnvironment(spare_bearing_quality=0.0)
        controller = advisory_factory.make(advisor=advisor, env=env)
        result = controller.run_cycle()
        assert result.incident.final_status == "ESCALATED"
        assert "AMD advisory" not in result.incident.escalation_reason
        assert "verification failed" in result.incident.escalation_reason

    def test_no_advisor_leaves_incident_untouched(self, advisory_factory):
        controller = advisory_factory.make(advisor=None)
        result = controller.run_cycle()
        assert result.incident.final_status == "RESOLVED"
        assert result.incident.amd_advisory is None


# ---------------------------------------------------------------------------
# Ledger + service + API wiring
# ---------------------------------------------------------------------------
class TestLedgerAndWiring:
    def test_ledger_records_advisory(self, advisory_factory):
        advisor = FakeAdvisor(_advisory())
        controller = advisory_factory.make(advisor=advisor)
        result = controller.run_cycle()
        ledger = build_incident_ledger(result.incident.as_dict(), scenario="bearing")
        assert ledger["amd_advisory"]["available"] is True
        assert "AMD advisory" not in ledger["unavailable_fields"]
        row = next(r for r in ledger["summary_rows"] if r["field"] == "AMD advisory")
        assert "replace_bearing" in row["value"]
        assert "not the executed action" in row["value"]

    def test_ledger_marks_advisory_unavailable_when_absent(self, advisory_factory):
        controller = advisory_factory.make(advisor=None)
        result = controller.run_cycle()
        ledger = build_incident_ledger(result.incident.as_dict(), scenario="bearing")
        assert ledger["amd_advisory"] is None
        assert "amd_advisory" in ledger["unavailable_fields"]
        row = next(r for r in ledger["summary_rows"] if r["field"] == "AMD advisory")
        assert row["value"] == "unavailable"

    def test_ledger_records_unavailable_advisory_reason(self, advisory_factory):
        advisor = FakeAdvisor(_advisory(available=False, reason="timeout",
                                        model="DeepSeek-V4-Flash"))
        controller = advisory_factory.make(advisor=advisor)
        result = controller.run_cycle()
        ledger = build_incident_ledger(result.incident.as_dict(), scenario="bearing")
        row = next(r for r in ledger["summary_rows"] if r["field"] == "AMD advisory")
        assert "timeout" in row["value"]
        assert "deterministic pipeline governed" in row["value"]

    def test_service_loads_no_advisor_without_credentials(self, monkeypatch,
                                                          tmp_path):
        from orbit.api import service
        monkeypatch.delenv("RADEON_API_KEY", raising=False)
        monkeypatch.setattr(
            "orbit.amd.config.DEFAULT_DOTENV_PATH", tmp_path / "absent.env")
        assert service._load_amd_advisor() is None
        status = service.amd_advisor_status()
        assert status["configured"] is False

    def test_service_use_amd_without_credentials_is_noop(self, monkeypatch,
                                                         tmp_path):
        """use_amd=True with no key must not call any API and must not change
        the deterministic outcome."""
        from orbit.api import service
        monkeypatch.delenv("RADEON_API_KEY", raising=False)
        monkeypatch.setattr(
            "orbit.amd.config.DEFAULT_DOTENV_PATH", tmp_path / "absent.env")
        result = service.run_recovery_cycle("bearing", use_amd=True)
        incident = result["cycles"][0]["incident"]
        assert incident["final_status"] == "RESOLVED"
        assert incident["amd_advisory"] is None
        assert result["device"]["amd_advisor"]["configured"] is False

    def test_cycle_request_accepts_use_amd_flag(self):
        from orbit.api.app import CycleRequest
        request = CycleRequest(scenario="bearing", use_amd=True)
        assert request.use_amd is True
        assert CycleRequest(scenario="bearing").use_amd is False
