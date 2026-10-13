"""In-process orchestration shared by the FastAPI server and Streamlit demo.

Wraps the deterministic ORBIT components behind a small, JSON-friendly
surface. Every endpoint returns real measurements taken from the
simulator + detector + controller — nothing is asserted without being
computed on this machine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from orbit.actions.environment import MaintenanceEnvironment
from orbit.config import MotorConfig
from orbit.detection.baseline import BaselineDetector
from orbit.diagnosis.engine import DiagnosisEngine
from orbit.evaluation.metrics import CALIBRATION_SEED, CALIBRATION_WINDOWS, _fit_detector
from orbit.incident.controller import IncidentController
from orbit.model.device import describe_environment
from orbit.simulator.motor import SCENARIOS, MachineState, MotorSimulator
from orbit.signalproc.features import FEATURE_NAMES, extract_features

EVALUATION_PATH = Path("runtime/evaluation/results.json")
BENCHMARK_DIR = Path("runtime/benchmark")
MODEL_WEIGHTS_PATH = Path("runtime/model/fault_cnn.pt")

# Precedence of the demo device: prefer GPU/ROCm when available,
# fall back to CPU silently for the deterministic pipeline (which does
# not need torch at all).
STATE_FIELDS = (
    "bearing_severity",
    "imbalance_severity",
    "cooling_efficiency",
    "load",
)


@lru_cache(maxsize=1)
def get_config() -> MotorConfig:
    return MotorConfig()


@lru_cache(maxsize=1)
def get_detector() -> BaselineDetector:
    return _fit_detector(get_config())


def scenario_names() -> List[str]:
    return sorted(SCENARIOS)


def scenario_state(name: str) -> MachineState:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; options: {scenario_names()}")
    return SCENARIOS[name].state.copy()


def _load_classifier():
    """Load the corroboration classifier if weights exist; else return None.

    The neural classifier is optional corroboration only: the deterministic
    pipeline must run without torch weights so the CPU dev path is never
    blocked by the model layer.
    """
    if not MODEL_WEIGHTS_PATH.exists():
        return None
    try:
        from orbit.model.inference import FaultClassifier

        return FaultClassifier.load()
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"load_error": str(exc)}


def device_summary() -> Dict[str, Any]:
    env = describe_environment()
    classifier = _load_classifier()
    if isinstance(classifier, dict):
        corroboration = {"available": False, "error": classifier["load_error"]}
    elif classifier is None:
        corroboration = {
            "available": False,
            "reason": "trained weights missing; run python -m orbit.model.train",
        }
    else:
        corroboration = {"available": True, "device": classifier.device_info.as_dict()}
    return {"environment": env, "classifier": corroboration}


def _make_simulator(config: MotorConfig, seed: int) -> MotorSimulator:
    sim = MotorSimulator(config, seed=seed)
    sim.advance(600.0)
    return sim


def sensor_trends(
    scenario: str,
    windows: int = 60,
    switch_after: int = 15,
    seed: int = 2024,
) -> Dict[str, Any]:
    """Return per-window feature trends: healthy for `switch_after`
    windows, then the requested fault scenario is injected. Down-samples
    one vibration waveform per switch-boundary window for plotting."""
    if windows < 2:
        raise ValueError("windows must be >= 2")
    if not 1 <= switch_after < windows:
        raise ValueError("switch_after must be in [1, windows)")

    config = get_config()
    sim = _make_simulator(config, seed)
    fault_state = scenario_state(scenario)

    rows: List[Dict[str, Any]] = []
    waveform: Optional[List[float]] = None
    waveform_at: Optional[float] = None

    for index in range(windows):
        if index == switch_after:
            sim.set_state(fault_state)
        win = sim.sample_window()
        feats = extract_features(win, config)
        rows.append({
            "window_index": index,
            "t_start_s": round(win.t_start_s, 3),
            "state": ("fault" if index >= switch_after else "healthy"),
            "temperature_c": round(win.temperature_c, 3),
            **{k: round(float(v), 4) for k, v in feats.items()},
        })
        if waveform is None or index == switch_after:
            step = max(1, win.vibration.size // 512)
            waveform = [float(v) for v in win.vibration[::step]]
            waveform_at = float(win.t_start_s)

    return {
        "scenario": scenario,
        "windows": rows,
        "waveform": {
            "t_start_s": waveform_at,
            "fs": config.fs,
            "downsample": waveform,
        },
        "feature_names": list(FEATURE_NAMES),
        "switch_after_index": switch_after,
    }


def _environment_from_payload(payload: Dict[str, Any]) -> MaintenanceEnvironment:
    """Build a MaintenanceEnvironment from an optional JSON payload.

    Unknown keys raise rather than being silently ignored, so the demo
    can never claim a scenario it did not configure.
    """
    valid = {f for f in MaintenanceEnvironment.__dataclass_fields__}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(f"unknown environment fields: {sorted(unknown)}")
    return MaintenanceEnvironment(**payload)


def _machine_state_from_payload(payload: Dict[str, Any]) -> MachineState:
    """Build a MachineState from an optional JSON payload.

    Fail-closed: unknown keys raise, and out-of-range severities raise via
    MachineState validation, so a demo can never claim a state it did not
    actually configure. An empty payload is invalid — callers only invoke
    this when a custom state was explicitly requested.
    """
    if not isinstance(payload, dict) or not payload:
        raise ValueError("machine_state must be a non-empty object of severity fields")
    valid = set(STATE_FIELDS)
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(f"unknown machine_state fields: {sorted(unknown)}")
    return MachineState(**payload)


def _feature_snapshot(sim: MotorSimulator, config: MotorConfig) -> Dict[str, float]:
    feats = extract_features(sim.sample_window(), config)
    return {k: round(float(v), 4) for k, v in feats.items()}


def run_recovery_cycle(
    scenario: str,
    cycles: int = 1,
    environment: Optional[Dict[str, Any]] = None,
    machine_state: Optional[Dict[str, Any]] = None,
    seed: int = 6006,
    monitor_interval_s: float = 60.0,
    use_classifier: bool = True,
) -> Dict[str, Any]:
    """Drive the full observe -> detect -> diagnose -> plan -> act ->
    verify loop for a scenario and return the incident audit trail.

    ``machine_state`` optionally overrides the named scenario with an
    explicit MachineState (custom severities / concurrent faults) for
    stress testing and ambiguity demonstrations. When omitted, the named
    scenario's state is used unchanged."""
    if cycles < 1 or cycles > 5:
        raise ValueError("cycles must be in [1, 5]")

    config = get_config()
    detector = get_detector()
    env = _environment_from_payload(environment or {})
    state = (_machine_state_from_payload(machine_state)
             if machine_state is not None else scenario_state(scenario))
    sim = _make_simulator(config, seed)
    sim.set_state(state)

    classifier = _load_classifier() if use_classifier else None
    if isinstance(classifier, dict):
        classifier = None  # load error: fall back to deterministic-only

    controller = IncidentController(
        simulator=sim,
        detector=detector,
        engine=DiagnosisEngine(),
        environment=env,
        classifier=classifier,
        config=config,
        monitor_interval_s=monitor_interval_s,
    )

    features_before = _feature_snapshot(sim, config)
    state_before = _state_dict(sim.state)
    cycle_results = []
    for _ in range(cycles):
        result = controller.run_cycle()
        cycle_results.append({
            "anomaly_detected": result.anomaly_detected,
            "monitoring_windows": result.monitoring_windows,
            "outcome": result.outcome,
            "incident": result.incident.as_dict() if result.incident else None,
        })

    features_after = _feature_snapshot(sim, config)
    detector_after = detector.predict(features_after)

    return {
        "scenario": scenario,
        "custom_state": machine_state is not None,
        "environment": {k: getattr(env, k) for k in valid_env_fields()},
        "state_before": state_before,
        "features_before": features_before,
        "cycles": cycle_results,
        "features_after": features_after,
        "state_after": _state_dict(sim.state),
        "still_anomalous": bool(detector_after.is_anomaly),
        "device": device_summary(),
    }


def valid_env_fields() -> Tuple[str, ...]:
    return tuple(MaintenanceEnvironment.__dataclass_fields__)


def _state_dict(state: MachineState) -> Dict[str, float]:
    return {name: round(float(getattr(state, name)), 4) for name in STATE_FIELDS}


def evaluation_results() -> Dict[str, Any]:
    if not EVALUATION_PATH.exists():
        return {
            "available": False,
            "reason": f"{EVALUATION_PATH} missing; run: python -m orbit.evaluation",
        }
    return json.loads(EVALUATION_PATH.read_text())


def latest_benchmark() -> Dict[str, Any]:
    if not BENCHMARK_DIR.exists():
        return {"available": False, "reason": "no benchmark files; run: python -m orbit.model.benchmark"}
    files = sorted(BENCHMARK_DIR.glob("benchmark_*.json"))
    if not files:
        return {"available": False, "reason": "no benchmark files; run: python -m orbit.model.benchmark"}
    return json.loads(files[-1].read_text())
