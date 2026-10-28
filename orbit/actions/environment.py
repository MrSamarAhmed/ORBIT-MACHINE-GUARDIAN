"""Simulated maintenance environment.

Models the resources an intervention depends on (spare parts, tools,
maintenance windows) and their *quality*. Quality in [0, 1] determines
intervention effectiveness in the simulator: a defective spare bearing
(quality 0) leaves the fault untouched, modelling the real-world case
where a corrective action is executed but does not cure the machine.

Everything here is simulated; nothing connects to real equipment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


if __name__ == "__main__":
    # Streamlit Cloud may still point at this legacy entry point. Delegate
    # only when executed as a script; normal imports keep this module pure.
    import runpy
    from pathlib import Path

    _demo_app = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    runpy.run_path(str(_demo_app), run_name="__main__")


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
        for name in ("machine_stoppable", "spare_bearing_available",
                     "balancing_kit_available", "reduced_load_possible"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("spare_bearing_quality", "balancing_quality",
                     "cooling_service_quality", "maintenance_window_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name == "maintenance_window_minutes":
                if value < 0:
                    raise ValueError("maintenance_window_minutes must be >= 0")
            elif not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
