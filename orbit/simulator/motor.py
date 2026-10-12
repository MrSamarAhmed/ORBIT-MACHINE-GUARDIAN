"""Deterministic motor simulator for ORBIT Machine Guardian.

Simulation assumptions (all documented, none claimed to be measured):

1. Vibration is a linear superposition of a healthy baseline (small
   residual 1x component + broadband sensor noise) and fault components.
2. An outer-race bearing defect produces an impulse train at BPFO with
   +/-2% timing jitter; each impulse rings a structural resonance at
   ``resonance_hz`` with exponential decay. This is the standard
   "bearing defect impulse response" model used in condition monitoring.
3. Shaft imbalance raises the 1x (once-per-revolution) component.
4. Temperature follows a first-order lag toward a load- and
   cooling-dependent equilibrium; bearing degradation adds friction heat.
5. All stochastic elements use a numpy Generator seeded deterministically
   from (seed, window_index, machine state), so a given state and index
   always produce bit-identical output.

The simulator models interventions as state changes; it never claims that
a real machine was repaired.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np

from orbit.config import MotorConfig


@dataclass
class MachineState:
    """Time-varying health state of the simulated motor."""

    bearing_severity: float = 0.0     # outer-race defect, 0..1
    imbalance_severity: float = 0.0   # rotor imbalance, 0..1
    cooling_efficiency: float = 1.0   # 1.0 = healthy, <1 = degraded
    load: float = 1.0                 # fraction of nominal load

    def __post_init__(self) -> None:
        for name in ("bearing_severity", "imbalance_severity"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0 < self.cooling_efficiency <= 1:
            raise ValueError(f"cooling_efficiency must be in (0, 1], got {self.cooling_efficiency}")
        if not 0 < self.load <= 1.5:
            raise ValueError(f"load must be in (0, 1.5], got {self.load}")

    def copy(self) -> "MachineState":
        return replace(self)

    def severity_key(self) -> tuple:
        """Discretized state used for deterministic seeding."""
        return (
            int(round(self.bearing_severity * 1000)),
            int(round(self.imbalance_severity * 1000)),
            int(round(self.cooling_efficiency * 1000)),
            int(round(self.load * 1000)),
        )


@dataclass(frozen=True)
class Scenario:
    """A named fault scenario used for tests and demos."""

    name: str
    state: MachineState
    description: str = ""


def _scenario(name: str, description: str, **state_kwargs) -> Scenario:
    return Scenario(name=name, state=MachineState(**state_kwargs), description=description)


SCENARIOS: Dict[str, Scenario] = {
    "normal": _scenario(
        "normal", "Healthy operation under nominal load.",
    ),
    "bearing": _scenario(
        "bearing",
        "Outer-race bearing degradation (severity 0.6).",
        bearing_severity=0.6,
    ),
    "imbalance": _scenario(
        "imbalance",
        "Shaft imbalance (severity 0.6).",
        imbalance_severity=0.6,
    ),
    "overtemperature": _scenario(
        "overtemperature",
        "Degraded cooling (efficiency 0.5) driving temperature above the alarm limit.",
        cooling_efficiency=0.5,
    ),
}


@dataclass
class SensorWindow:
    """One sampling window of simulated sensor data."""

    window_index: int
    t_start_s: float
    vibration: np.ndarray        # accelerometer [m/s^2], length window_samples
    temperature_c: float
    shaft_hz: float
    state: MachineState          # state snapshot while the window was produced

    def __post_init__(self) -> None:
        if self.vibration.ndim != 1:
            raise ValueError("vibration must be a 1-D array")


class MotorSimulator:
    """Seeded, deterministic motor simulator.

    Vibration windows depend only on (config, seed, machine state, window
    index); thermal state additionally depends on elapsed time, which the
    simulator tracks internally.
    """

    def __init__(self, config: Optional[MotorConfig] = None, seed: Optional[int] = None):
        self.config = config if config is not None else MotorConfig()
        if seed is None:
            seed = self.config.seed
        self.seed = int(seed)
        self.state = MachineState()
        self.temperature_c = (
            self.config.temp_ambient_c + self.config.temp_healthy_rise_c
        )
        self._clock_s = 0.0
        self._window_index = 0

    # ------------------------------------------------------------------
    # State / scenario management
    # ------------------------------------------------------------------
    def set_state(self, state: MachineState) -> None:
        if not isinstance(state, MachineState):
            raise ValueError("state must be a MachineState instance")
        self.state = state.copy()

    def set_scenario(self, name: str) -> Scenario:
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}; options: {sorted(SCENARIOS)}")
        scenario = SCENARIOS[name]
        self.set_state(scenario.state)
        return scenario

    # ------------------------------------------------------------------
    # Thermal model
    # ------------------------------------------------------------------
    def equilibrium_temperature_c(self, state: Optional[MachineState] = None) -> float:
        state = state if state is not None else self.state
        healthy_rise = self.config.temp_healthy_rise_c * state.load
        friction_rise = 30.0 * state.bearing_severity
        return self.config.temp_ambient_c + (healthy_rise + friction_rise) / state.cooling_efficiency

    def advance(self, seconds: float) -> None:
        """Fast-forward the thermal state without generating vibration."""
        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        if seconds == 0:
            return
        eq = self.equilibrium_temperature_c()
        tau = self.config.temp_tau_s
        self.temperature_c = eq + (self.temperature_c - eq) * math.exp(-seconds / tau)
        self._clock_s += seconds

    # ------------------------------------------------------------------
    # Sensor sampling
    # ------------------------------------------------------------------
    def _window_rng(self, state: MachineState, index: int) -> np.random.Generator:
        return np.random.default_rng(
            [self.seed, index, *state.severity_key()]
        )

    def _vibration(self, state: MachineState, index: int) -> np.ndarray:
        cfg = self.config
        n = cfg.window_samples
        rng = self._window_rng(state, index)
        t = np.arange(n) / cfg.fs

        # Healthy baseline: residual 1x + small harmonics + sensor noise
        x = np.zeros(n)
        phase = rng.uniform(0, 2 * np.pi)
        x += cfg.healthy_1x_amp * np.sin(2 * np.pi * cfg.shaft_hz * t + phase)
        x += 0.4 * cfg.healthy_1x_amp * np.sin(2 * np.pi * 2 * cfg.shaft_hz * t + phase * 1.7)
        x += 0.2 * cfg.healthy_1x_amp * np.sin(2 * np.pi * 3 * cfg.shaft_hz * t + phase * 2.3)
        x += rng.normal(0.0, cfg.healthy_noise_std, n)

        # Shaft imbalance: grows the 1x component
        if state.imbalance_severity > 0:
            x += (
                state.imbalance_severity
                * cfg.imbalance_1x_amp
                * np.sin(2 * np.pi * cfg.shaft_hz * t + phase + 0.9)
            )
            x += (
                0.15
                * state.imbalance_severity
                * cfg.imbalance_1x_amp
                * np.sin(2 * np.pi * 2 * cfg.shaft_hz * t + phase + 1.4)
            )

        # Outer-race bearing defect: jittered impulse train at BPFO,
        # each impulse ringing the structural resonance.
        if state.bearing_severity > 0:
            bpfo = cfg.bpfo_hz
            period = 1.0 / bpfo
            duration = t[-1] + 1.0 / cfg.fs
            impulse_amp = state.bearing_severity * cfg.bearing_impulse_amp * (0.5 + 0.5 * state.load)
            k_max = int(math.ceil(duration / period)) + 1
            offsets = (rng.uniform(0.0, 1.0, k_max) - 0.5) * 2 * cfg.defect_jitter * period
            for k in range(k_max):
                t_k = k * period + offsets[k] - period  # start one period early
                if t_k > t[-1]:
                    break
                dt = t - t_k
                active = dt >= 0
                response = np.zeros(n)
                response[active] = (
                    np.exp(-dt[active] / cfg.resonance_decay_s)
                    * np.sin(2 * np.pi * cfg.resonance_hz * dt[active])
                )
                x += impulse_amp * response
        return x

    def sample_window(self) -> SensorWindow:
        """Advance the clock by one window and return the sensor readings."""
        cfg = self.config
        state = self.state.copy()
        index = self._window_index
        t_start = self._clock_s

        self.advance(cfg.window_s)
        vibration = self._vibration(state, index)

        window = SensorWindow(
            window_index=index,
            t_start_s=t_start,
            vibration=vibration,
            temperature_c=self.temperature_c,
            shaft_hz=cfg.shaft_hz,
            state=state,
        )
        self._window_index += 1
        return window

    def sample_windows(self, count: int) -> "list[SensorWindow]":
        if count <= 0:
            raise ValueError("count must be >= 1")
        return [self.sample_window() for _ in range(count)]
