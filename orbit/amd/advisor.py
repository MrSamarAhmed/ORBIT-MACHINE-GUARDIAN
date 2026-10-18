"""AMD Token Factory advisory client (OpenAI-compatible chat API).

The model is an **advisor**, not a controller. It reads incident evidence
ORBIT has already collected and returns a structured recommendation. It
never executes anything, never selects the action ORBIT performs, and can
never move an incident to ``RESOLVED`` — only
``orbit.recovery.verifier.verify_recovery`` may do that.

Every failure mode (missing credentials, missing SDK, timeout, HTTP
error, malformed JSON, schema violation, unsupported action) is returned
as a structured ``available: False`` result so the deterministic pipeline
continues unchanged. An API failure is never interpreted as a diagnosis
or as a recovery.

``assess()`` never raises: callers can treat an unavailable advisory the
same way they treat a disabled classifier.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from orbit.amd.config import AmdConfig
from orbit.diagnosis.knowledge import ACTION_LIBRARY

PROVIDER = "amd_token_factory"

REQUIRED_FIELDS = (
    "assessment",
    "supporting_evidence",
    "recommended_action",
    "confidence",
    "needs_escalation",
)
ALLOWED_CONFIDENCE = ("high", "medium", "low")
MAX_SUPPORTING_EVIDENCE = 12
MAX_ASSESSMENT_CHARS = 2000

SYSTEM_PROMPT = (
    "You are a maintenance advisory analyst for an industrial rotating-machine "
    "monitoring system. You are given evidence a deterministic system already "
    "collected: sensor features, anomaly-detector breaches, an optional neural "
    "classifier result, and the deterministic diagnosis with ranked hypotheses.\n\n"
    "Respond with ONLY a single JSON object and no prose, using exactly these keys:\n"
    '  "assessment": short string analysing the evidence\n'
    '  "supporting_evidence": array of strings citing the specific features or '
    "detector findings you relied on\n"
    '  "recommended_action": string; MUST be one of the permitted action ids '
    "listed in the payload, or \"none\" if no intervention is justified\n"
    '  "confidence": one of "high", "medium", "low"\n'
    '  "needs_escalation": boolean; true if a human should review before acting\n\n'
    "Rules: do not invent measurements that are absent from the payload. If the "
    "evidence is insufficient or conflicting, say so and set needs_escalation to "
    "true. Your recommendation is advisory only - an independent deterministic "
    "verifier, not you, decides whether the machine recovered."
)


def _redact(text: str, api_key: str) -> str:
    """Strip the credential from any message before it is recorded."""
    return text.replace(api_key, "[REDACTED]") if api_key else text


def build_evidence_payload(
    *,
    incident_id: Optional[str],
    features: Dict[str, float],
    anomaly: Any,
    diagnosis: Any,
    corroboration: Optional[Dict[str, Any]] = None,
    evidence_trend: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble the compact evidence payload from real recorded ORBIT data.

    Pure function: no API call, no invented values. Anything ORBIT did not
    record is passed through as ``None`` rather than guessed.
    """
    return {
        "incident_id": incident_id,
        "sensor_features": {k: round(float(v), 4) for k, v in features.items()},
        "anomaly_detector": {
            "is_anomaly": bool(anomaly.is_anomaly),
            "max_abs_z": round(float(anomaly.max_abs_z), 2),
            "temp_limit_exceeded": bool(anomaly.temp_limit_exceeded),
            "breaches": [b.as_evidence() for b in anomaly.breaches],
        },
        "neural_classifier": corroboration,
        "evidence_trend": evidence_trend,
        "orbit_diagnosis": diagnosis.as_summary() if diagnosis is not None else None,
        "permitted_actions": sorted(ACTION_LIBRARY),
    }


def _normalize_confidence(raw: Any) -> Tuple[Optional[str], Optional[str]]:
    """Map the model's confidence onto an advisory label.

    Returns ``(label, error)``. A numeric confidence is bucketed, but it is
    recorded as model-reported and never treated as a calibrated probability.
    """
    if isinstance(raw, bool):
        return None, "confidence must be a string label or number, got bool"
    if isinstance(raw, str):
        label = raw.strip().lower()
        if label in ALLOWED_CONFIDENCE:
            return label, None
        return None, (
            f"confidence {raw!r} not in {list(ALLOWED_CONFIDENCE)}"
        )
    if isinstance(raw, (int, float)):
        value = float(raw)
        if not 0.0 <= value <= 1.0:
            return None, f"numeric confidence {value} outside [0, 1]"
        if value >= 0.75:
            return "high", None
        if value >= 0.4:
            return "medium", None
        return "low", None
    return None, f"confidence has unsupported type {type(raw).__name__}"


def validate_assessment(raw: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate the complete response schema.

    Returns ``(assessment, error)``. Malformed or unsupported responses are
    rejected with an explicit reason instead of being coerced, so a partial
    answer can never masquerade as a usable recommendation.
    """
    if not isinstance(raw, dict):
        return None, f"expected a JSON object, got {type(raw).__name__}"

    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        return None, f"missing required field(s): {missing}"
    unexpected = sorted(set(raw) - set(REQUIRED_FIELDS))
    if unexpected:
        return None, f"unexpected field(s): {unexpected}"

    assessment = raw["assessment"]
    if not isinstance(assessment, str) or not assessment.strip():
        return None, "assessment must be a non-empty string"
    if len(assessment) > MAX_ASSESSMENT_CHARS:
        return None, f"assessment exceeds {MAX_ASSESSMENT_CHARS} characters"

    evidence = raw["supporting_evidence"]
    if not isinstance(evidence, list) or not evidence:
        return None, "supporting_evidence must be a non-empty array"
    if len(evidence) > MAX_SUPPORTING_EVIDENCE:
        return None, (
            f"supporting_evidence has {len(evidence)} items "
            f"(max {MAX_SUPPORTING_EVIDENCE})"
        )
    if not all(isinstance(item, str) and item.strip() for item in evidence):
        return None, "every supporting_evidence item must be a non-empty string"

    action = raw["recommended_action"]
    if not isinstance(action, str) or not action.strip():
        return None, "recommended_action must be a non-empty string"
    action = action.strip()

    confidence, confidence_error = _normalize_confidence(raw["confidence"])
    if confidence_error:
        return None, confidence_error

    needs_escalation = raw["needs_escalation"]
    if not isinstance(needs_escalation, bool):
        return None, "needs_escalation must be a boolean"

    return {
        "assessment": assessment.strip(),
        "supporting_evidence": [item.strip() for item in evidence],
        "recommended_action": action,
        "action_supported": action in ACTION_LIBRARY or action == "none",
        "confidence": confidence,
        "confidence_reported": raw["confidence"],
        "needs_escalation": needs_escalation,
    }, None


class AmdAdvisor:
    """Advisory client for the AMD-hosted model. Optional by design."""

    def __init__(self, config: AmdConfig):
        self.config = config

    @property
    def identity(self) -> Dict[str, object]:
        return self.config.redacted()

    # ------------------------------------------------------------------
    def _build_client(self) -> Any:
        """Construct the OpenAI-compatible client lazily.

        Importing ``openai`` here (not at module scope) keeps the
        deterministic ORBIT pipeline importable without the SDK installed.
        Retries are disabled so a limited API quota is never burned by
        automatic re-requests.
        """
        from openai import OpenAI

        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            max_retries=0,
        )

    def _unavailable(self, reason: str, detail: str = "") -> Dict[str, Any]:
        detail = _redact(detail, self.config.api_key)
        return {
            "available": False,
            "provider": PROVIDER,
            "model": self.config.model,
            "reason": reason,
            "detail": detail or None,
            # Explicit: an API failure produced no diagnosis and no recovery.
            "diagnosis_produced": False,
            "recovery_declared": False,
        }

    @staticmethod
    def _classify_exception(exc: BaseException) -> str:
        name = type(exc).__name__.lower()
        module = type(exc).__module__ or ""
        if "timeout" in name or "timeout" in str(exc).lower():
            return "timeout"
        if "apiconnection" in name or "connection" in name:
            return "connection_error"
        if "apierror" in name or "apistatus" in name or module.startswith("openai"):
            return "http_error"
        return "api_error"

    def assess(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the model for a structured advisory on recorded evidence.

        Makes exactly one API call. Never raises; every failure returns a
        structured ``available: False`` result naming the reason. Missing
        credentials never reach here: ``AmdConfig.from_env`` returns None
        when no key is configured, so no advisor is constructed.
        """
        try:
            client = self._build_client()
        except ImportError as exc:
            return self._unavailable("sdk_not_installed", f"{exc}")
        except Exception as exc:  # pragma: no cover - defensive
            return self._unavailable(
                "client_init_failed", f"{type(exc).__name__}: {exc}"
            )

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(evidence, sort_keys=True, default=str),
                    },
                ],
                temperature=0.0,
            )
        except Exception as exc:
            return self._unavailable(
                self._classify_exception(exc), f"{type(exc).__name__}: {exc}"
            )

        content = self._extract_content(response)
        if content is None:
            return self._unavailable(
                "unexpected_response", "model returned no message content"
            )

        parsed, parse_error = self._parse_json(content)
        if parse_error:
            return self._unavailable("invalid_json", parse_error)

        assessment, schema_error = validate_assessment(parsed)
        if schema_error:
            return self._unavailable("invalid_schema", schema_error)

        assert assessment is not None  # validate_assessment guarantees this
        return {
            "available": True,
            "provider": PROVIDER,
            "model": self.config.model,
            "assessment": assessment["assessment"],
            "supporting_evidence": assessment["supporting_evidence"],
            "recommended_action": assessment["recommended_action"],
            "action_supported": assessment["action_supported"],
            "confidence": assessment["confidence"],
            "confidence_reported": assessment["confidence_reported"],
            "confidence_basis": (
                "model-reported; advisory only, not a calibrated probability"
            ),
            "needs_escalation": assessment["needs_escalation"],
            # The model never decides these; recorded so the ledger can show it.
            "diagnosis_produced": False,
            "recovery_declared": False,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_content(response: Any) -> Optional[str]:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        return content

    @staticmethod
    def _parse_json(content: str) -> Tuple[Any, Optional[str]]:
        """Parse the model's JSON, tolerating code fences and stray prose."""
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), None
            except json.JSONDecodeError as exc:
                return None, f"embedded JSON object unparseable: {exc}"
        return None, "response contained no JSON object"
