"""Simulated maintenance environment.

Models the resources an intervention depends on (spare parts, tools,
maintenance windows) and their *quality*. Quality in [0, 1] determines
intervention effectiveness in the simulator: a defective spare bearing
(quality 0) leaves the fault untouched, modelling the real-world case
where a corrective action is executed but does not cure the machine.

Everything here is simulated; nothing connects to real equipment.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenanceEnvironment:
    machine_stoppable: bool = True
    spare_bearing_available: bool = True
    spare_bearing_quality: float = 1.0
    balancing_kit_available: bool = True
    balancing_quality: float = 1.0
    reduced_load_possible: bool = True
    cooling_service_quality: float = 1.0
    maintenance_window_minutes: float = 30.0

    def __post_init__(self) -> None:
        for name in ("spare_bearing_quality", "balancing_quality", "cooling_service_quality"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.maintenance_window_minutes < 0:
            raise ValueError("maintenance_window_minutes must be >= 0")
