"""Lifecycle state machine tests."""
from __future__ import annotations

import pytest

from orbit.incident.lifecycle import (
    ALLOWED_TRANSITIONS,
    InvalidTransition,
    STATES,
    assert_transition,
)


class TestStateMachine:
    def test_all_states_have_transitions(self):
        for state in STATES:
            assert state in ALLOWED_TRANSITIONS

    def test_happy_path(self):
        path = [
            "MONITORING", "DETECTED", "DIAGNOSING", "ACTION_PLANNED",
            "SIMULATING", "VERIFYING", "RESOLVED", "MONITORING",
        ]
        for current, target in zip(path, path[1:]):
            assert_transition(current, target)

    def test_escalation_paths(self):
        assert_transition("DIAGNOSING", "ESCALATED")
        assert_transition("ACTION_PLANNED", "ESCALATED")
        assert_transition("SIMULATING", "ESCALATED")
        assert_transition("VERIFYING", "ESCALATED")
        assert_transition("ESCALATED", "MONITORING")

    def test_diagnosing_self_loop_for_evidence(self):
        assert_transition("DIAGNOSING", "DIAGNOSING")

    def test_verification_retry(self):
        assert_transition("VERIFYING", "ACTION_PLANNED")

    def test_invalid_transitions_raise(self):
        for current, target in [
            ("MONITORING", "RESOLVED"),
            ("MONITORING", "SIMULATING"),
            ("DETECTED", "SIMULATING"),
            ("DETECTED", "RESOLVED"),
            ("RESOLVED", "DETECTED"),
            ("ESCALATED", "ACTION_PLANNED"),
            ("VERIFYING", "DETECTED"),
            ("SIMULATING", "RESOLVED"),  # must pass through VERIFYING
        ]:
            with pytest.raises(InvalidTransition):
                assert_transition(current, target)

    def test_unknown_states_raise(self):
        with pytest.raises(InvalidTransition):
            assert_transition("MONITORING", "HAPPY")
        with pytest.raises(InvalidTransition):
            assert_transition("HAPPY", "MONITORING")

    def test_only_verifying_reaches_resolved(self):
        """RESOLVED is reachable only from VERIFYING: no path bypasses the
        deterministic verifier."""
        for source, targets in ALLOWED_TRANSITIONS.items():
            if source != "VERIFYING":
                assert "RESOLVED" not in targets
