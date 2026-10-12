"""Simulated intervention executor.

Applies the documented effect of a knowledge-base action to the machine
simulator state. Effects are explicit and conservative:

* ``replace_bearing`` removes ``spare_bearing_quality`` of the current
  bearing severity. A defective spare (quality 0) changes nothing — the
  intervention is executed but ineffective.
* ``field_balancing`` removes ``balancing_quality`` of the imbalance.
* ``cooling_overhaul`` restores cooling efficiency toward 1.0 by
  ``cooling_service_quality``.

Each action declares a settle time: how long the machine must run after
the intervention before verification measurements are meaningful
(mechanical effects are near-immediate; thermal settling takes minutes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from orbit.actions.environment import MaintenanceEnvironment
from orbit.diagnosis.knowledge import ActionPlan, ACTION_LIBRARY
from orbit.simulator.motor import MachineState, MotorSimulator


@dataclass(frozen=True)
class ActionResult:
    action_id: str
    executed: bool
    settle_time_s: float
    state_before: MachineState
    state_after: MachineState
    effectiveness: float
    notes: Tuple[str, ...]


def _replace_bearing(state: MachineState, env: MaintenanceEnvironment) -> MachineState:
    quality = env.spare_bearing_quality
    new_severity = state.bearing_severity * (1.0 - quality)
    return MachineState(
        bearing_severity=new_severity,
        imbalance_severity=state.imbalance_severity,
        cooling_efficiency=state.cooling_efficiency,
        load=state.load,
    )


def _field_balancing(state: MachineState, env: MaintenanceEnvironment) -> MachineState:
    quality = env.balancing_quality
    return MachineState(
        bearing_severity=state.bearing_severity,
        imbalance_severity=state.imbalance_severity * (1.0 - quality),
        cooling_efficiency=state.cooling_efficiency,
        load=state.load,
    )


def _cooling_overhaul(state: MachineState, env: MaintenanceEnvironment) -> MachineState:
    quality = env.cooling_service_quality
    return MachineState(
        bearing_severity=state.bearing_severity,
        imbalance_severity=state.imbalance_severity,
        cooling_efficiency=state.cooling_efficiency
        + quality * (1.0 - state.cooling_efficiency),
        load=state.load,
    )


EFFECTS: Dict[str, Callable[[MachineState, MaintenanceEnvironment], MachineState]] = {
    "replace_bearing": _replace_bearing,
    "field_balancing": _field_balancing,
    "cooling_overhaul": _cooling_overhaul,
}

SETTLE_TIME_S: Dict[str, float] = {
    "replace_bearing": 2.0,      # mechanical: run-in only
    "field_balancing": 2.0,      # mechanical
    "cooling_overhaul": 150.0,   # thermal: ~5 time constants
}


class ActionExecutor:
    """Executes simulated interventions; never touches real equipment."""

    def check_preconditions(
        self, action: ActionPlan, env: MaintenanceEnvironment
    ) -> Tuple[bool, Tuple[str, ...]]:
        """Evaluate every declared precondition. Fail-closed on unknown ids."""
        failures = []
        for pre in action.preconditions:
            check = REQUIREMENT_CHECKS.get(pre.requirement_id)
            if check is None:
                raise ValueError(
                    f"unknown requirement_id {pre.requirement_id!r} for action "
                    f"{action.action_id!r}; refusing to proceed (fail-closed)"
                )
            if not check(env):
                failures.append(f"precondition not met: {pre.description}")
        return (not failures, tuple(failures))

    def execute(
        self,
        action: ActionPlan,
        simulator: MotorSimulator,
        env: MaintenanceEnvironment,
    ) -> ActionResult:
        if action.action_id not in ACTION_LIBRARY:
            raise ValueError(f"unknown action {action.action_id!r}")
        effect = EFFECTS.get(action.action_id)
        if effect is None:
            raise ValueError(
                f"no simulated effect defined for action {action.action_id!r}"
            )
        before = simulator.state.copy()
        after = effect(before, env)
        simulator.set_state(after)
        if action.action_id == "replace_bearing":
            effectiveness = env.spare_bearing_quality
        elif action.action_id == "field_balancing":
            effectiveness = env.balancing_quality
        else:
            effectiveness = env.cooling_service_quality
        notes = []
        if effectiveness < 1.0:
            notes.append(
                f"intervention effectiveness reduced to {effectiveness:.0%} "
                "(resource quality)"
            )
        return ActionResult(
            action_id=action.action_id,
            executed=True,
            settle_time_s=SETTLE_TIME_S[action.action_id],
            state_before=before,
            state_after=after,
            effectiveness=effectiveness,
            notes=tuple(notes),
        )


def _require_machine_stoppable(env: MaintenanceEnvironment) -> bool:
    return env.machine_stoppable


def _require_spare_bearing(env: MaintenanceEnvironment) -> bool:
    return env.spare_bearing_available


def _require_balancing_kit(env: MaintenanceEnvironment) -> bool:
    return env.balancing_kit_available


def _require_reduced_load(env: MaintenanceEnvironment) -> bool:
    return env.reduced_load_possible


def _require_maintenance_window(env: MaintenanceEnvironment) -> bool:
    return env.maintenance_window_minutes >= 15.0


REQUIREMENT_CHECKS: Dict[str, Callable[[MaintenanceEnvironment], bool]] = {
    "machine_stoppable": _require_machine_stoppable,
    "spare_bearing_in_store": _require_spare_bearing,
    "balancing_kit_available": _require_balancing_kit,
    "reduced_load_possible": _require_reduced_load,
    "maintenance_window_15min": _require_maintenance_window,
}
