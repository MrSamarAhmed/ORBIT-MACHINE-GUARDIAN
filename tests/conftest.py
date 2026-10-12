"""Shared test fixtures: simulator, config, and a fitted baseline."""
from __future__ import annotations

import pytest

from orbit.config import MotorConfig
from orbit.detection.baseline import BaselineDetector
from orbit.signalproc.features import extract_features
from orbit.simulator.motor import MachineState, MotorSimulator

CALIBRATION_WINDOWS = 40

SCENARIOS_STATE = {
    "normal": MachineState(),
    "bearing": MachineState(bearing_severity=0.6),
    "imbalance": MachineState(imbalance_severity=0.6),
    "overtemperature": MachineState(cooling_efficiency=0.5),
}


@pytest.fixture(scope="session")
def config() -> MotorConfig:
    return MotorConfig()


@pytest.fixture(scope="session")
def healthy_rows(config) -> list:
    sim = MotorSimulator(config, seed=7)
    sim.set_state(SCENARIOS_STATE["normal"])
    sim.advance(600.0)  # reach thermal steady state
    return [extract_features(w, config) for w in sim.sample_windows(CALIBRATION_WINDOWS)]


@pytest.fixture(scope="session")
def detector(healthy_rows, config) -> BaselineDetector:
    return BaselineDetector(z_threshold=3.5, temp_limit_c=config.temp_limit_c).fit(healthy_rows)
