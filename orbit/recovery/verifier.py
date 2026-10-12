"""Deterministic recovery verification.

The verifier is the ONLY authority allowed to close an incident. It
evaluates an action's declared verification conditions against
post-intervention feature rows and requires every condition to pass on
every evaluated window — a single failing window means the machine has
not demonstrably recovered.

This module contains no randomness and no heuristics: given the same
feature rows it always returns the same verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from orbit.diagnosis.knowledge import ActionPlan, VerificationCondition


@dataclass(frozen=True)
class ConditionResult:
    condition: VerificationCondition
    window_index: int
    value: float
    passed: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "condition": self.condition.name,
            "feature": self.condition.feature,
            "value": round(self.value, 4),
            "operator": self.condition.operator,
            "threshold": self.condition.threshold,
            "window": self.window_index,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class VerificationOutcome:
    passed: bool
    results: Tuple[ConditionResult, ...]
    windows_evaluated: int

    @property
    def failures(self) -> Tuple[ConditionResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    def as_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "windows_evaluated": self.windows_evaluated,
            "results": [r.as_dict() for r in self.results],
        }


def verify_recovery(
    action: ActionPlan, feature_rows: Sequence[Dict[str, float]]
) -> VerificationOutcome:
    if not feature_rows:
        raise ValueError("at least one post-intervention window is required")
    if not action.verification:
        raise ValueError(f"action {action.action_id} declares no verification conditions")
    results: List[ConditionResult] = []
    for position, row in enumerate(feature_rows):
        for condition in action.verification:
            if condition.feature not in row:
                raise ValueError(
                    f"feature {condition.feature!r} missing from verification window"
                )
            results.append(
                ConditionResult(
                    condition=condition,
                    window_index=position,
                    value=float(row[condition.feature]),
                    passed=condition.evaluate(row),
                )
            )
    passed = all(r.passed for r in results)
    return VerificationOutcome(
        passed=passed,
        results=tuple(results),
        windows_evaluated=len(feature_rows),
    )
