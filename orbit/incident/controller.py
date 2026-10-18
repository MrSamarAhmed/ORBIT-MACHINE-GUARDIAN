"""Autonomous incident controller.

Drives the full OBSERVE -> DETECT -> DIAGNOSE -> PLAN -> ACT -> VERIFY
-> RECOVER-OR-ESCALATE loop against the motor simulator.

Design rules (inherited from the MaintAI trust model):

* Only the deterministic verifier may move an incident to RESOLVED.
  The diagnoser, planner, or executor can never declare recovery.
* Ambiguous diagnoses never trigger an intervention; the controller
  collects more evidence, then escalates.
* Precondition failures block the intervention and escalate.
* Every phase transition is recorded with its evidence, producing a
  complete audit trail of what the agent knew and did at each step.
* The optional AMD Token Factory advisory reads recorded evidence and
  recommends, but never decides: it cannot select the action, cannot
  execute, and cannot declare recovery. Its only permitted effect is the
  conservative one — requesting escalation for human review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from orbit.actions.environment import MaintenanceEnvironment
from orbit.actions.executor import ActionExecutor, ActionResult
from orbit.amd.advisor import build_evidence_payload
from orbit.config import MotorConfig
from orbit.detection.baseline import AnomalyResult, BaselineDetector
from orbit.diagnosis.engine import DiagnosisEngine, DiagnosisResult
from orbit.incident.lifecycle import InvalidTransition, assert_transition
from orbit.recovery.verifier import VerificationOutcome, verify_recovery
from orbit.signalproc.features import extract_features
from orbit.simulator.motor import MachineState, MotorSimulator

if TYPE_CHECKING:  # torch-backed, imported only for type checking
    from orbit.model.inference import FaultClassifier as CorroboratingClassifier
    from orbit.amd.advisor import AmdAdvisor

DEFAULT_MONITOR_WINDOWS = 3
DEFAULT_MONITOR_INTERVAL_S = 60.0
DEFAULT_EVIDENCE_ROUNDS = 3
DEFAULT_VERIFICATION_WINDOWS = 3

# Corroborating neural classifier over vibration windows. It never
# decides actions: its probabilities are recorded alongside the
# deterministic diagnosis as supporting evidence (AMD/ROCm path).
CLASSIFIER_MODE_TO_HYPOTHESIS = {
    "bearing": "bearing_outer_defect",
    "imbalance": "shaft_imbalance",
    "normal": None,
}


@dataclass
class PhaseRecord:
    phase: str
    summary: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentRecord:
    incident_id: str
    opened_by_breach: List[str]
    opened_at_utc: Optional[str] = None
    phases: List[PhaseRecord] = field(default_factory=list)
    final_status: Optional[str] = None
    diagnosis: Optional[DiagnosisResult] = None
    corroboration: Optional[Dict[str, Any]] = None
    amd_advisory: Optional[Dict[str, Any]] = None
    action_id: Optional[str] = None
    verification: Optional[VerificationOutcome] = None
    escalation_reason: Optional[str] = None
    attempts_used: int = 0

    def phase_summaries(self) -> List[str]:
        return [f"{p.phase}: {p.summary}" for p in self.phases]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "opened_at_utc": self.opened_at_utc,
            "final_status": self.final_status,
            "opened_by_breach": self.opened_by_breach,
            "diagnosis": self.diagnosis.as_summary() if self.diagnosis else None,
            "corroboration": self.corroboration,
            "amd_advisory": self.amd_advisory,
            "action_id": self.action_id,
            "attempts_used": self.attempts_used,
            "verification": self.verification.as_dict() if self.verification else None,
            "escalation_reason": self.escalation_reason,
            "phases": [
                {"phase": p.phase, "summary": p.summary, "evidence": p.evidence}
                for p in self.phases
            ],
        }


@dataclass
class CycleResult:
    """Outcome of one autonomous cycle (monitoring + incident handling)."""

    anomaly_detected: bool
    incident: Optional[IncidentRecord] = None
    monitoring_windows: int = 0

    @property
    def outcome(self) -> str:
        if self.incident is None:
            return "HEALTHY_NO_ACTION"
        return self.incident.final_status or "OPEN"


class IncidentController:
    def __init__(
        self,
        simulator: MotorSimulator,
        detector: BaselineDetector,
        engine: DiagnosisEngine,
        environment: Optional[MaintenanceEnvironment] = None,
        executor: Optional[ActionExecutor] = None,
        classifier: Optional["CorroboratingClassifier"] = None,
        amd_advisor: Optional["AmdAdvisor"] = None,
        config: Optional[MotorConfig] = None,
        monitor_windows: int = DEFAULT_MONITOR_WINDOWS,
        monitor_interval_s: float = DEFAULT_MONITOR_INTERVAL_S,
        evidence_rounds: int = DEFAULT_EVIDENCE_ROUNDS,
        verification_windows: int = DEFAULT_VERIFICATION_WINDOWS,
    ):
        if monitor_windows < 1:
            raise ValueError("monitor_windows must be >= 1")
        if monitor_interval_s < 0:
            raise ValueError("monitor_interval_s must be >= 0")
        if evidence_rounds < 1:
            raise ValueError("evidence_rounds must be >= 1")
        if verification_windows < 1:
            raise ValueError("verification_windows must be >= 1")
        self.simulator = simulator
        self.detector = detector
        self.engine = engine
        self.environment = environment if environment is not None else MaintenanceEnvironment()
        self.executor = executor if executor is not None else ActionExecutor()
        self.classifier = classifier
        self.amd_advisor = amd_advisor
        self.config = config if config is not None else simulator.config
        self.monitor_windows = monitor_windows
        self.monitor_interval_s = float(monitor_interval_s)
        self.evidence_rounds = evidence_rounds
        self.verification_windows = verification_windows
        self.state = "MONITORING"
        self.incidents: List[IncidentRecord] = []
        self._incident_counter = 0

    # ------------------------------------------------------------------
    def _transition(self, target: str, incident: IncidentRecord,
                    summary: str, evidence: Optional[Dict[str, Any]] = None) -> None:
        assert_transition(self.state, target)
        self.state = target
        incident.phases.append(
            PhaseRecord(phase=target, summary=summary, evidence=evidence or {})
        )

    def _sample_features(self):
        window = self.simulator.sample_window()
        features = extract_features(window, self.config)
        return features, self.detector.predict(features), window

    def _classify_corroboration(self, window) -> Optional[Dict[str, object]]:
        """Run the (optionally GPU-accelerated) vibration classifier on a
        raw window. Corroborating evidence only — never authoritative."""
        if self.classifier is None:
            return None
        result = self.classifier.classify(window.vibration[None, :])
        predicted_mode = CLASSIFIER_MODE_TO_HYPOTHESIS.get(result["predicted_class"])
        return {
            "predicted_class": result["predicted_class"],
            "class_probabilities": result["class_probabilities"],
            "mapped_hypothesis": predicted_mode,
            "device": result["device"],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _diagnosis_trend(incident: IncidentRecord) -> List[Dict[str, Any]]:
        """Top hypothesis scores across the recorded diagnosis rounds.

        Real trend data: every ambiguous-evidence DIAGNOSING phase stores
        its ranked hypotheses, so the advisor sees how the ranking evolved
        instead of only the final verdict. Phases that recorded nothing
        contribute nothing.
        """
        trend: List[Dict[str, Any]] = []
        for phase in incident.phases:
            if phase.phase != "DIAGNOSING":
                continue
            hypotheses = phase.evidence.get("hypotheses")
            if not hypotheses:
                continue
            runner = hypotheses[1] if len(hypotheses) > 1 else {}
            trend.append({
                "top": hypotheses[0].get("mode"),
                "top_score": hypotheses[0].get("score"),
                "runner_up": runner.get("mode"),
                "runner_up_score": runner.get("score"),
            })
        return trend

    def _consult_amd_advisor(
        self,
        incident: IncidentRecord,
        features: Dict[str, float],
        anomaly: AnomalyResult,
        diagnosis: DiagnosisResult,
        corroboration: Optional[Dict[str, object]],
    ) -> Optional[Dict[str, Any]]:
        """Consult the optional AMD advisory model. Advisory only.

        The advisor receives evidence ORBIT has already recorded and
        returns a structured recommendation. It can never select the
        executed action, never execute anything, and never declare
        recovery: a valid recommendation is recorded with an
        ``agrees_with_plan`` verdict while ORBIT's deterministic plan
        governs; an API or schema failure yields ``available: False`` and
        changes nothing.
        """
        if self.amd_advisor is None:
            return None
        try:
            payload = build_evidence_payload(
                incident_id=incident.incident_id,
                features=features,
                anomaly=anomaly,
                diagnosis=diagnosis,
                corroboration=corroboration,
                evidence_trend=self._diagnosis_trend(incident),
            )
            result = self.amd_advisor.assess(payload)
        except Exception as exc:  # defensive: an advisor bug must not stop the loop
            result = {
                "available": False,
                "provider": "amd_token_factory",
                "reason": "advisor_unexpected_error",
                "detail": f"{type(exc).__name__}: {exc}",
                "diagnosis_produced": False,
                "recovery_declared": False,
            }
        if result.get("available") and result.get("recommended_action") is not None:
            result["agrees_with_plan"] = (
                result["recommended_action"]
                == (diagnosis.recommended_action.action_id
                    if diagnosis.recommended_action else None)
            )
        return result

    # ------------------------------------------------------------------
    def run_cycle(self) -> CycleResult:
        """Observe the machine and, on anomaly, run the full recovery loop."""
        if self.state != "MONITORING":
            raise InvalidTransition(
                f"cannot start a cycle from {self.state!r}; incident still open"
            )
        windows = 0
        first_anomaly: Optional[Tuple[Dict[str, float], AnomalyResult, Any]] = None
        for check in range(self.monitor_windows):
            if check > 0:
                self.simulator.advance(self.monitor_interval_s)
            features, anomaly, window = self._sample_features()
            windows += 1
            if anomaly.is_anomaly:
                first_anomaly = (features, anomaly, window)
                break
        if first_anomaly is None:
            return CycleResult(
                anomaly_detected=False, monitoring_windows=windows
            )
        incident = self._open_incident(first_anomaly[1])
        return CycleResult(
            anomaly_detected=True,
            incident=self._handle_incident(incident, first_anomaly),
            monitoring_windows=windows,
        )

    def _open_incident(self, anomaly: AnomalyResult) -> IncidentRecord:
        self._incident_counter += 1
        record = IncidentRecord(
            incident_id=f"INC-{self._incident_counter:03d}",
            opened_by_breach=[b.as_evidence() for b in anomaly.breaches],
            opened_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.incidents.append(record)
        self._transition(
            "DETECTED",
            record,
            f"anomaly detected: {len(anomaly.breaches)} baseline breach(es)",
            evidence={
                "breaches": [b.as_evidence() for b in anomaly.breaches],
                "max_abs_z": round(anomaly.max_abs_z, 2),
                "temp_limit_exceeded": anomaly.temp_limit_exceeded,
            },
        )
        return record

    # ------------------------------------------------------------------
    def _handle_incident(
        self, incident: IncidentRecord,
        initial: Tuple[Dict[str, float], AnomalyResult, Any],
    ) -> IncidentRecord:
        features, anomaly, window = initial

        # --- DIAGNOSE (with extra evidence rounds when ambiguous) ---
        self._transition("DIAGNOSING", incident, "ranking fault hypotheses")
        diagnosis = self.engine.diagnose(anomaly, features, self.detector)
        for round_index in range(self.evidence_rounds):
            if diagnosis.is_confident:
                break
            self._transition(
                "DIAGNOSING",
                incident,
                f"diagnosis ambiguous; collecting additional evidence "
                f"(round {round_index + 1}/{self.evidence_rounds})",
                evidence={"hypotheses": diagnosis.as_summary()["hypotheses"]},
            )
            features, anomaly, window = self._sample_features()
            if not anomaly.is_anomaly:
                self._resolve_incident(
                    incident,
                    diagnosis,
                    reason="anomaly did not persist across evidence windows; "
                           "treating as transient and closing",
                )
                return incident
            diagnosis = self.engine.diagnose(anomaly, features, self.detector)
        incident.diagnosis = diagnosis
        if not diagnosis.is_confident:
            self._escalate(
                incident,
                diagnosis,
                reason=f"diagnosis remained ambiguous after "
                       f"{self.evidence_rounds} evidence rounds: "
                       f"{diagnosis.ambiguity_note}",
            )
            return incident

        action = diagnosis.recommended_action
        assert action is not None  # confident diagnoses always carry an action

        # Corroborating neural verdict (optional, AMD/ROCm-accelerated).
        corroboration = self._classify_corroboration(window)
        incident.corroboration = corroboration
        if corroboration is not None:
            corroboration["agrees_with_diagnosis"] = (
                corroboration["mapped_hypothesis"] == diagnosis.primary.mode_id
            )

        # AMD Token Factory advisory (optional): evidence analysis only.
        # Recorded verbatim; the deterministic plan below still governs.
        amd_advisory = self._consult_amd_advisor(
            incident=incident,
            features=features,
            anomaly=anomaly,
            diagnosis=diagnosis,
            corroboration=corroboration,
        )
        incident.amd_advisory = amd_advisory

        # --- PLAN ---
        ok, failures = self.executor.check_preconditions(action, self.environment)
        self._transition(
            "ACTION_PLANNED",
            incident,
            f"planned action: {action.title}",
            evidence={
                "target_hypothesis": diagnosis.primary.mode_id,
                "hypothesis_score": round(diagnosis.primary.score, 2),
                "supporting_evidence": [e.as_dict() for e in diagnosis.primary.supporting],
                "contradicting_evidence": [
                    e.as_dict() for e in diagnosis.primary.contradicting
                ],
                "neural_corroboration": corroboration,
                "amd_advisory": amd_advisory,
                "action": action.action_id,
                "preconditions": [
                    {"id": p.requirement_id, "description": p.description}
                    for p in action.preconditions
                ],
                "preconditions_met": ok,
                "expected_changes": list(action.expected_changes),
                "verification_conditions": [
                    {"name": v.name, "feature": v.feature,
                     "operator": v.operator, "threshold": v.threshold}
                    for v in action.verification
                ],
            },
        )
        if not ok:
            self._escalate(
                incident,
                diagnosis,
                reason="intervention preconditions not met: " + "; ".join(failures),
            )
            return incident

        # The advisory's only permitted control-flow effect is the
        # conservative one: requesting human review before any simulated
        # intervention. It cannot select the action and cannot declare
        # recovery; only verify_recovery below may reach RESOLVED. An
        # unavailable or failed advisory never escalates or blocks.
        if amd_advisory is not None and amd_advisory.get("available") \
                and amd_advisory.get("needs_escalation"):
            self._escalate(
                incident,
                diagnosis,
                reason=(
                    f"AMD advisory ({amd_advisory.get('model')}) requested "
                    f"human review before intervention: "
                    f"{amd_advisory.get('assessment')}"
                ),
            )
            return incident

        # --- ACT + VERIFY (retry loop) ---
        while True:
            incident.attempts_used += 1
            self._transition(
                "SIMULATING",
                incident,
                f"simulating intervention: {action.title} "
                f"(attempt {incident.attempts_used}/{action.max_attempts})",
                evidence={"action": action.action_id},
            )
            result: ActionResult = self.executor.execute(
                action, self.simulator, self.environment
            )
            incident.phases[-1].evidence.update(
                {
                    "executed": result.executed,
                    "effectiveness": round(result.effectiveness, 4),
                    "settle_time_s": result.settle_time_s,
                    "state_before": {
                        k: round(v, 4) for k, v in asdict(result.state_before).items()
                    },
                    "state_after": {
                        k: round(v, 4) for k, v in asdict(result.state_after).items()
                    },
                    "notes": list(result.notes),
                }
            )
            self.simulator.advance(result.settle_time_s)
            post_rows = []
            for _ in range(self.verification_windows):
                row, _, _ = self._sample_features()
                post_rows.append(row)

            self._transition("VERIFYING", incident, "verifying recovery conditions")
            outcome = self._verify_recovery(action, post_rows)
            incident.verification = outcome
            incident.action_id = action.action_id
            if outcome.passed:
                self._transition(
                    "RESOLVED",
                    incident,
                    "recovery verified: all conditions passed on "
                    f"{outcome.windows_evaluated} post-intervention windows",
                    evidence=outcome.as_dict(),
                )
                incident.final_status = "RESOLVED"
                self._return_to_monitoring(incident)
                return incident
            if incident.attempts_used >= action.max_attempts:
                self._escalate(
                    incident,
                    diagnosis,
                    reason=f"verification failed after {incident.attempts_used} "
                           f"attempt(s): "
                           + "; ".join(
                               f"{f.condition.name}={f.value:.4f} "
                               f"(need {f.condition.operator} {f.condition.threshold})"
                               for f in outcome.failures
                           ),
                )
                return incident
            self._transition(
                "ACTION_PLANNED",
                incident,
                "verification failed; replanning intervention",
                evidence=outcome.as_dict(),
            )

    # ------------------------------------------------------------------
    def _verify_recovery(self, action, post_rows):
        """Deterministic verification — the only path to RESOLVED.

        Hook method so the evaluation comparison harness can instantiate
        a naive no-verification baseline by overriding it. Production
        code always takes the deterministic verifier.
        """
        return verify_recovery(action, post_rows)

    def _escalate(self, incident: IncidentRecord, diagnosis: DiagnosisResult,
                  reason: str) -> None:
        action = diagnosis.recommended_action
        note = action.escalation_note if action else ""
        self._transition(
            "ESCALATED",
            incident,
            f"escalated to human operator: {reason}",
            evidence={"reason": reason, "knowledge_base_note": note},
        )
        incident.final_status = "ESCALATED"
        incident.escalation_reason = reason
        self._return_to_monitoring(incident)

    def _resolve_incident(self, incident: IncidentRecord, diagnosis: DiagnosisResult,
                          reason: str) -> None:
        self._transition("RESOLVED", incident, reason)
        incident.final_status = "RESOLVED"
        self._return_to_monitoring(incident)

    def _return_to_monitoring(self, incident: IncidentRecord) -> None:
        self._transition("MONITORING", incident, "incident closed; resuming monitoring")
