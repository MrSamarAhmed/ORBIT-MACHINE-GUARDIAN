"""Incident lifecycle state machine.

Deterministic, explicitly enumerated transitions. Invalid transitions
raise — an incident can never silently jump states.

    MONITORING -> DETECTED -> DIAGNOSING -> ACTION_PLANNED -> SIMULATING
                -> VERIFYING -> RESOLVED | ESCALATED

DIAGNOSING may self-loop while collecting additional evidence; VERIFYING
may return to ACTION_PLANNED for a retry while attempts remain; both
DIAGNOSING and ACTION_PLANNED can escalate (ambiguous evidence, unmet
preconditions). Terminal states return to MONITORING once the incident is
closed (by resolution or by a human handling the escalation).
"""
from __future__ import annotations


class InvalidTransition(ValueError):
    """Raised when a lifecycle transition is not allowed."""


STATES = (
    "MONITORING",
    "DETECTED",
    "DIAGNOSING",
    "ACTION_PLANNED",
    "SIMULATING",
    "VERIFYING",
    "RESOLVED",
    "ESCALATED",
)

TERMINAL_STATES = ("RESOLVED", "ESCALATED")

ALLOWED_TRANSITIONS = {
    "MONITORING": frozenset({"DETECTED"}),
    "DETECTED": frozenset({"DIAGNOSING"}),
    "DIAGNOSING": frozenset({"ACTION_PLANNED", "ESCALATED", "DIAGNOSING"}),
    "ACTION_PLANNED": frozenset({"SIMULATING", "ESCALATED"}),
    "SIMULATING": frozenset({"VERIFYING", "ESCALATED"}),
    "VERIFYING": frozenset({"RESOLVED", "ACTION_PLANNED", "ESCALATED"}),
    "RESOLVED": frozenset({"MONITORING"}),
    "ESCALATED": frozenset({"MONITORING"}),
}


def assert_transition(current: str, target: str) -> None:
    if current not in ALLOWED_TRANSITIONS:
        raise InvalidTransition(f"unknown state {current!r}")
    if target not in ALLOWED_TRANSITIONS:
        raise InvalidTransition(f"unknown state {target!r}")
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransition(
            f"illegal transition {current!r} -> {target!r}; allowed: "
            f"{sorted(ALLOWED_TRANSITIONS[current])}"
        )
