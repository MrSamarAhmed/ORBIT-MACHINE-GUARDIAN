"""Transparent business-value projection calculator.

This module exists because plant managers think in downtime hours and
money, not in z-scores. Every formula is stated, every input is
configurable, and the output is a RANGE with the assumptions attached.

What the projection can honestly claim to model:
1. Repeat-downtime avoidance. When an intervention does not actually fix
   the machine but the workflow declares success, the fault resurfaces
   as a second outage. A verification-aware workflow escalates instead,
   so the repair is done once, with evidence. The fraction of
   interventions that fail silently is an INPUT (``failed_recovery_fraction``);
   ORBIT's synthetic-scenario measurement (0.75 for a naive workflow) is
   offered as a preset, clearly labelled as simulator-only.
2. Diagnosis-time reduction. An escalated incident arrives with a
   structured evidence ledger; the technician does not start from a
   blank page. The assumed reduction fraction is an INPUT and an
   assumption, not a measured fact.

These are PROJECTIONS under stated assumptions, not measured savings.
No customer deployments, testimonials, or real factory results exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

#: Measured in ORBIT's synthetic comparison (orbit/evaluation/comparison.py):
#: share of naive workflow "resolved" claims where the machine had NOT
#: actually recovered. Simulator-only measurement; not an industrial figure.
MEASURED_NAIVE_FALSE_RESOLUTION_RATE = 0.75

DISCLAIMER = (
    "Projection under stated assumptions - NOT measured savings. "
    "All fault scenarios are simulated; no customer deployments exist."
)


@dataclass(frozen=True)
class ValueInputs:
    """All monetary and operational assumptions are explicit inputs."""

    incidents_per_year: float = 24.0
    downtime_hours_per_incident: float = 4.0
    cost_per_downtime_hour: float = 10_000.0
    technician_hours_per_incident: float = 3.0
    technician_cost_per_hour: float = 90.0
    # Share of corrective interventions that do NOT actually fix the
    # machine yet are declared successful by a naive workflow.
    failed_recovery_fraction: float = MEASURED_NAIVE_FALSE_RESOLUTION_RATE
    # Share of a failed intervention's downtime that recurs as a second
    # outage before the fault is truly fixed.
    repeat_downtime_share: float = 0.5
    # How much of the diagnostic work the evidence ledger removes for
    # escalated incidents. Pure assumption.
    diagnosis_time_reduction: float = 0.3
    # Sensitivity: low/high scenarios vary every fraction by this amount.
    sensitivity: float = 0.1

    def __post_init__(self) -> None:
        for name in ("incidents_per_year", "downtime_hours_per_incident",
                     "cost_per_downtime_hour", "technician_hours_per_incident",
                     "technician_cost_per_hour"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("failed_recovery_fraction", "repeat_downtime_share",
                     "diagnosis_time_reduction", "sensitivity"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")


@dataclass(frozen=True)
class ValueEstimate:
    baseline_downtime_cost: float
    avoided_repeat_downtime_cost: float
    diagnosis_time_savings: float
    total_estimate: float
    low: float
    high: float
    lines: List[Dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "baseline_downtime_cost": self.baseline_downtime_cost,
            "avoided_repeat_downtime_cost": self.avoided_repeat_downtime_cost,
            "diagnosis_time_savings": self.diagnosis_time_savings,
            "total_estimate": self.total_estimate,
            "range": [self.low, self.high],
            "lines": self.lines,
            "disclaimer": DISCLAIMER,
        }


def _delta_repeat_downtime(p: ValueInputs) -> float:
    """Annual cost of repeat outages a naive workflow accepts."""
    return (p.incidents_per_year * p.failed_recovery_fraction
            * p.repeat_downtime_share * p.downtime_hours_per_incident
            * p.cost_per_downtime_hour)


def _delta_diagnosis_time(p: ValueInputs) -> float:
    """Annual technician-time savings from evidence-led escalations."""
    return (p.incidents_per_year * p.diagnosis_time_reduction
            * p.technician_hours_per_incident * p.technician_cost_per_hour)


def _variant(p: ValueInputs, shift: float) -> ValueEstimate:
    """Compute the estimate with every fraction shifted by ``shift``."""
    varied = ValueInputs(
        incidents_per_year=p.incidents_per_year,
        downtime_hours_per_incident=p.downtime_hours_per_incident,
        cost_per_downtime_hour=p.cost_per_downtime_hour,
        technician_hours_per_incident=p.technician_hours_per_incident,
        technician_cost_per_hour=p.technician_cost_per_hour,
        failed_recovery_fraction=_clamp(
            p.failed_recovery_fraction + shift),
        repeat_downtime_share=_clamp(p.repeat_downtime_share + shift),
        diagnosis_time_reduction=_clamp(
            p.diagnosis_time_reduction + shift),
        sensitivity=p.sensitivity,
    )
    repeat = _delta_repeat_downtime(varied)
    diagnosis = _delta_diagnosis_time(varied)
    baseline = (p.incidents_per_year * p.downtime_hours_per_incident
                * p.cost_per_downtime_hour)
    return ValueEstimate(
        baseline_downtime_cost=baseline,
        avoided_repeat_downtime_cost=repeat,
        diagnosis_time_savings=diagnosis,
        total_estimate=repeat + diagnosis,
        low=repeat + diagnosis,
        high=repeat + diagnosis,
    )


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def estimate_value(p: ValueInputs) -> ValueEstimate:
    """Annual projection with a low/high sensitivity range."""
    base = _variant(p, 0.0)
    low = _variant(p, -p.sensitivity)
    high = _variant(p, +p.sensitivity)
    lines = [
        {
            "term": "Baseline annual downtime cost",
            "formula": "incidents × downtime_hours × cost_per_hour",
            "value": base.baseline_downtime_cost,
        },
        {
            "term": "Avoided repeat downtime (verification + escalation)",
            "formula": "incidents × failed_recovery_fraction × "
                       "repeat_downtime_share × downtime_hours × cost_per_hour",
            "value": base.avoided_repeat_downtime_cost,
        },
        {
            "term": "Technician diagnosis-time savings (evidence ledger)",
            "formula": "incidents × diagnosis_time_reduction × "
                       "technician_hours × technician_cost_per_hour",
            "value": base.diagnosis_time_savings,
        },
    ]
    return ValueEstimate(
        baseline_downtime_cost=base.baseline_downtime_cost,
        avoided_repeat_downtime_cost=base.avoided_repeat_downtime_cost,
        diagnosis_time_savings=base.diagnosis_time_savings,
        total_estimate=base.total_estimate,
        low=low.total_estimate,
        high=high.total_estimate,
        lines=lines,
    )
