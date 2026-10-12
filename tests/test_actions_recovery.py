"""Action executor and recovery verifier tests."""
from __future__ import annotations

import pytest

from orbit.actions.environment import MaintenanceEnvironment
from orbit.actions.executor import ActionExecutor, REQUIREMENT_CHECKS
from orbit.diagnosis.knowledge import ACTION_LIBRARY, VerificationCondition
from orbit.recovery.verifier import verify_recovery
from orbit.simulator.motor import MachineState, MotorSimulator


@pytest.fixture(scope="module")
def executor() -> ActionExecutor:
    return ActionExecutor()


class TestPreconditions:
    def test_all_preconditions_have_checks(self, executor):
        for action in ACTION_LIBRARY.values():
            for pre in action.preconditions:
                assert pre.requirement_id in REQUIREMENT_CHECKS
            ok, failures = executor.check_preconditions(action, MaintenanceEnvironment())
            assert ok and not failures

    def test_missing_spare_blocks_bearing_replacement(self, executor):
        env = MaintenanceEnvironment(spare_bearing_available=False)
        action = ACTION_LIBRARY["replace_bearing"]
        ok, failures = executor.check_preconditions(action, env)
        assert not ok
        assert any("bearing available" in f.lower() for f in failures)

    def test_short_window_blocks_cooling_overhaul(self, executor):
        env = MaintenanceEnvironment(maintenance_window_minutes=10)
        action = ACTION_LIBRARY["cooling_overhaul"]
        ok, failures = executor.check_preconditions(action, env)
        assert not ok and failures

    def test_unstoppable_machine_blocks_mechanical_actions(self, executor):
        env = MaintenanceEnvironment(machine_stoppable=False)
        for action_id in ("replace_bearing", "field_balancing"):
            ok, _ = executor.check_preconditions(ACTION_LIBRARY[action_id], env)
            assert not ok

    def test_unknown_requirement_fails_closed(self):
        from orbit.diagnosis.knowledge import ActionPlan, Precondition

        bogus = ActionPlan(
            action_id="x", title="x", description="",
            preconditions=(Precondition("nonexistent_requirement", "desc"),),
            expected_changes=(), verification=(),
        )
        with pytest.raises(ValueError, match="unknown requirement_id"):
            ActionExecutor().check_preconditions(bogus, MaintenanceEnvironment())


class TestExecutorEffects:
    def test_good_spare_cures_bearing(self, executor, config):
        sim = MotorSimulator(config, seed=1)
        sim.set_state(MachineState(bearing_severity=0.6))
        result = executor.execute(
            ACTION_LIBRARY["replace_bearing"], sim, MaintenanceEnvironment()
        )
        assert sim.state.bearing_severity == pytest.approx(0.0)
        assert result.effectiveness == 1.0
        assert result.state_after.bearing_severity == 0.0
        assert result.settle_time_s > 0

    def test_defective_spares_are_ineffective(self, executor, config):
        sim = MotorSimulator(config, seed=1)
        sim.set_state(MachineState(bearing_severity=0.6))
        env = MaintenanceEnvironment(spare_bearing_quality=0.0)
        result = executor.execute(ACTION_LIBRARY["replace_bearing"], sim, env)
        assert sim.state.bearing_severity == pytest.approx(0.6)
        assert result.effectiveness == 0.0
        assert result.notes  # the reduced effectiveness is recorded

    def test_partial_quality_partial_effect(self, executor, config):
        sim = MotorSimulator(config, seed=1)
        sim.set_state(MachineState(imbalance_severity=0.8))
        env = MaintenanceEnvironment(balancing_quality=0.5)
        executor.execute(ACTION_LIBRARY["field_balancing"], sim, env)
        assert sim.state.imbalance_severity == pytest.approx(0.4)

    def test_cooling_overhaul_restores_efficiency(self, executor, config):
        sim = MotorSimulator(config, seed=1)
        sim.set_state(MachineState(cooling_efficiency=0.5))
        executor.execute(ACTION_LIBRARY["cooling_overhaul"], sim, MaintenanceEnvironment())
        assert sim.state.cooling_efficiency == pytest.approx(1.0)

    def test_cooling_settle_time_is_thermal(self, executor, config):
        sim = MotorSimulator(config, seed=1)
        result = executor.execute(
            ACTION_LIBRARY["cooling_overhaul"], sim, MaintenanceEnvironment()
        )
        assert result.settle_time_s >= 100.0  # minutes-scale thermal settling

    def test_unknown_action_raises(self, executor, config):
        from orbit.diagnosis.knowledge import ActionPlan

        ghost = ActionPlan(
            action_id="ghost", title="g", description="",
            preconditions=(), expected_changes=(), verification=(),
        )
        with pytest.raises(ValueError, match="unknown action"):
            executor.execute(ghost, MotorSimulator(config), MaintenanceEnvironment())

    def test_other_state_fields_untouched(self, executor, config):
        sim = MotorSimulator(config, seed=1)
        sim.set_state(MachineState(bearing_severity=0.4, imbalance_severity=0.7,
                                   cooling_efficiency=0.6, load=1.2))
        executor.execute(ACTION_LIBRARY["replace_bearing"], sim, MaintenanceEnvironment())
        assert sim.state.imbalance_severity == 0.7
        assert sim.state.cooling_efficiency == 0.6
        assert sim.state.load == 1.2


class TestVerifier:
    def _bearing_action(self):
        return ACTION_LIBRARY["replace_bearing"]

    def test_healthy_rows_pass(self, executor):
        action = self._bearing_action()
        rows = [
            {"env_bpfo_ratio": 1.4, "vib_kurtosis": 2.5},
            {"env_bpfo_ratio": 1.8, "vib_kurtosis": 3.0},
        ]
        outcome = verify_recovery(action, rows)
        assert outcome.passed
        assert outcome.windows_evaluated == 2
        assert not outcome.failures

    def test_single_failing_window_fails(self):
        action = self._bearing_action()
        rows = [
            {"env_bpfo_ratio": 1.4, "vib_kurtosis": 2.5},
            {"env_bpfo_ratio": 9.9, "vib_kurtosis": 7.0},  # fault persists
            {"env_bpfo_ratio": 1.4, "vib_kurtosis": 2.5},
        ]
        outcome = verify_recovery(action, rows)
        assert not outcome.passed
        assert len(outcome.failures) == 2
        failed_features = {f.condition.feature for f in outcome.failures}
        assert failed_features == {"env_bpfo_ratio", "vib_kurtosis"}

    def test_empty_rows_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            verify_recovery(self._bearing_action(), [])

    def test_missing_feature_raises(self):
        with pytest.raises(ValueError, match="missing"):
            verify_recovery(self._bearing_action(), [{"vib_kurtosis": 2.0}])

    def test_action_without_conditions_raises(self):
        from orbit.diagnosis.knowledge import ActionPlan

        action = ActionPlan(
            action_id="x", title="x", description="",
            preconditions=(), expected_changes=(), verification=(),
        )
        with pytest.raises(ValueError, match="no verification conditions"):
            verify_recovery(action, [{"env_bpfo_ratio": 1.0}])

    def test_deterministic(self):
        rows = [{"env_bpfo_ratio": 2.0, "vib_kurtosis": 4.0}]
        a = verify_recovery(self._bearing_action(), rows)
        b = verify_recovery(self._bearing_action(), rows)
        assert a.passed == b.passed
        assert [r.as_dict() for r in a.results] == [r.as_dict() for r in b.results]

    def test_condition_operators(self):
        cond = VerificationCondition(name="t", feature="temp_c", operator="<", threshold=75.0)
        assert cond.evaluate({"temp_c": 70.0})
        assert not cond.evaluate({"temp_c": 80.0})
        with pytest.raises(ValueError):
            VerificationCondition(name="t", feature="temp_c", operator=">=", threshold=1.0).evaluate({"temp_c": 1.0})
