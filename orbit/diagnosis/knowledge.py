"""Maintenance knowledge base: fault modes, signatures, corrective actions.

This is an explicit, inspectable knowledge base in the style of FMEA
(failure mode and effects analysis) tables used in industrial
maintenance. Each fault mode declares:

* which features support it and in which direction,
* which features contradict it when they stay inside baseline,
* a corrective action with preconditions (safety/system requirements
  before the intervention may run) and deterministic verification
  conditions (measurable postconditions proving recovery).

The knowledge base is deliberately small and human-reviewable: it is a
demo of the workflow, not a plant-specific FMEA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SignatureRule:
    """A feature rule supporting a fault hypothesis."""

    feature: str
    direction: str  # "high" or "low"
    weight: float   # relative contribution to hypothesis score
    rationale: str


@dataclass(frozen=True)
class VerificationCondition:
    """A measurable postcondition an intervention must achieve."""

    name: str
    feature: str
    operator: str          # "<" or "<="
    threshold: float
    units: str = ""

    def evaluate(self, features: Dict[str, float]) -> bool:
        value = float(features[self.feature])
        if self.operator == "<":
            return value < self.threshold
        if self.operator == "<=":
            return value <= self.threshold
        raise ValueError(f"unsupported operator {self.operator!r}")


@dataclass(frozen=True)
class ActionPlan:
    """A structured corrective intervention plan."""

    action_id: str
    title: str
    description: str
    preconditions: Tuple[str, ...]
    expected_changes: Tuple[str, ...]
    verification: Tuple[VerificationCondition, ...]
    max_attempts: int = 1
    escalation_note: str = ""


@dataclass(frozen=True)
class FaultMode:
    """A known fault mode with its diagnostic signature and remedy."""

    mode_id: str
    display_name: str
    description: str
    signature: Tuple[SignatureRule, ...]
    actions: Tuple[ActionPlan, ...]
    contradicting_note: str = ""


def _sig(feature, direction, weight, rationale) -> SignatureRule:
    return SignatureRule(feature=feature, direction=direction, weight=weight, rationale=rationale)


def _ver(name, feature, operator, threshold, units="") -> VerificationCondition:
    return VerificationCondition(
        name=name, feature=feature, operator=operator,
        threshold=threshold, units=units,
    )


def _act(action_id, title, description, preconditions, expected, verification,
         max_attempts=1, escalation_note="") -> ActionPlan:
    return ActionPlan(
        action_id=action_id, title=title, description=description,
        preconditions=tuple(preconditions), expected_changes=tuple(expected),
        verification=tuple(verification), max_attempts=max_attempts,
        escalation_note=escalation_note,
    )


BEARING_DEFECT = FaultMode(
    mode_id="bearing_outer_defect",
    display_name="Bearing outer-race defect",
    description=(
        "Localized spall on the outer race generates periodic impacts at the "
        "ball-pass frequency (BPFO). Impacts ring the structural resonance, "
        "raising high-frequency energy, envelope BPFO amplitude, and kurtosis."
    ),
    signature=(
        _sig("env_bpfo_ratio", "high", 3.0,
             "Envelope spectrum peak at BPFO is the primary outer-race signature."),
        _sig("vib_kurtosis", "high", 2.0,
             "Periodic impacts make the vibration amplitude distribution heavy-tailed."),
        _sig("hf_energy_frac", "high", 1.5,
             "Resonance ringing concentrates energy in the 2-5 kHz band."),
        _sig("spec_bpfo_band_amp", "high", 1.5,
             "Raw spectrum shows elevated energy near BPFO."),
    ),
    actions=(
        _act(
            "replace_bearing",
            "Replace drive-end bearing",
            "Swap the 6205 drive-end bearing with a new unit (simulated).",
            preconditions=(
                "Machine can be brought to a controlled stop",
                "Replacement bearing available in stores",
            ),
            expected=(
                "Envelope BPFO ratio returns to baseline",
                "Kurtosis returns to near-Gaussian values",
            ),
            verification=(
                _ver("envelope_bpfo_normal", "env_bpfo_ratio", "<", 3.0),
                _ver("kurtosis_normal", "vib_kurtosis", "<", 4.5),
            ),
            max_attempts=1,
            escalation_note="If a new bearing does not clear the signature, "
                            "escalate for shaft-alignment inspection.",
        ),
    ),
    contradicting_note=(
        "Normal envelope BPFO and kurtosis contradict a bearing defect even "
        "when overall vibration RMS is elevated."
    ),
)

SHAFT_IMBALANCE = FaultMode(
    mode_id="shaft_imbalance",
    display_name="Shaft imbalance",
    description=(
        "Uneven rotor mass distribution raises the once-per-revolution "
        "centrifugal force, producing a dominant 1x spectral line with "
        "little impact noise."
    ),
    signature=(
        _sig("spec_1x_amp", "high", 3.0,
             "Dominant 1x component is the primary imbalance signature."),
        _sig("vib_rms", "high", 1.0,
             "Overall vibration level rises with the 1x component."),
        _sig("spec_2x_amp", "high", 0.5,
             "Small 2x side energy can accompany severe imbalance."),
    ),
    actions=(
        _act(
            "field_balancing",
            "In-place rotor balancing",
            "Add correction mass at the computed angle on the rotor (simulated).",
            preconditions=(
                "Machine can be brought to a controlled stop",
                "Balancing kit with calibrated weights available",
            ),
            expected=(
                "1x amplitude returns to baseline tolerance",
                "Overall RMS decreases accordingly",
            ),
            verification=(
                _ver("spec_1x_normal", "spec_1x_amp", "<", 0.5),
            ),
            max_attempts=1,
            escalation_note="If balancing does not reduce 1x, escalate for "
                            "coupling/bent-shaft inspection.",
        ),
    ),
    contradicting_note=(
        "Elevated kurtosis or envelope BPFO with a normal 1x line points away "
        "from imbalance toward a bearing fault."
    ),
)

COOLING_DEGRADATION = FaultMode(
    mode_id="cooling_degradation",
    display_name="Cooling system degradation",
    description=(
        "Blocked filters or a failing fan reduce heat rejection. Temperature "
        "settles above the alarm limit while vibration stays healthy."
    ),
    signature=(
        _sig("temp_c", "high", 3.0,
             "Absolute temperature above the engineering alarm limit."),
    ),
    actions=(
        _act(
            "cooling_overhaul",
            "Clean/replace cooling filters and fan",
            "Restore cooling airflow by cleaning filters and servicing the fan (simulated).",
            preconditions=(
                "Machine can be run at reduced load during service",
                "Maintenance window at least 15 minutes",
            ),
            expected=(
                "Temperature falls toward the healthy equilibrium",
            ),
            verification=(
                _ver("temp_below_limit", "temp_c", "<", 75.0, "degC"),
            ),
            max_attempts=1,
            escalation_note="If temperature stays high after cooling service, "
                            "escalate for motor rewinding inspection.",
        ),
    ),
    contradicting_note=(
        "Elevated temperature with strong vibration signatures may indicate a "
        "mechanical fault generating friction heat instead of a cooling problem."
    ),
)

KNOWLEDGE_BASE: Dict[str, FaultMode] = {
    mode.mode_id: mode
    for mode in (BEARING_DEFECT, SHAFT_IMBALANCE, COOLING_DEGRADATION)
}

ACTION_LIBRARY: Dict[str, ActionPlan] = {
    action.action_id: action
    for mode in KNOWLEDGE_BASE.values()
    for action in mode.actions
}
