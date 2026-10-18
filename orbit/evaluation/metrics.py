"""Measured evaluation harness (Phase 6).

Produces honest, reproducible metrics on SYNTHETIC simulator data with
disjoint seeds for calibration vs evaluation. Every number written here
is actually computed by running the components; nothing is asserted
without measurement.

Run: ``python -m orbit.evaluation`` -> runtime/evaluation/results.json
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from orbit.actions.environment import MaintenanceEnvironment
from orbit.config import MotorConfig
from orbit.detection.baseline import BaselineDetector
from orbit.diagnosis.engine import DiagnosisEngine
from orbit.incident.controller import IncidentController
from orbit.model.dataset import CLASSES, CLASS_TO_INDEX, generate_split
from orbit.simulator.motor import MachineState, MotorSimulator
from orbit.signalproc.features import extract_features

EVAL_OUT = Path("runtime/evaluation/results.json")

# Calibration uses a distinct seed from every evaluation split (no leakage).
CALIBRATION_SEED = 7
CALIBRATION_WINDOWS = 60

# Evaluation fault grid: (class, bearing_sev, imbalance_sev, cooling_eff).
FAULT_GRID = [
    ("bearing", bs, 0.0, 1.0)
    for bs in (0.3, 0.5, 0.7, 0.9)
] + [
    ("imbalance", 0.0, ims, 1.0)
    for ims in (0.3, 0.5, 0.7, 0.9)
] + [
    ("overtemperature", 0.0, 0.0, eff)
    for eff in (0.4, 0.5)
]


def _fit_detector(config: MotorConfig) -> BaselineDetector:
    sim = MotorSimulator(config, seed=CALIBRATION_SEED)
    sim.advance(600.0)
    rows = [extract_features(w, config) for w in sim.sample_windows(CALIBRATION_WINDOWS)]
    return BaselineDetector(z_threshold=3.5, temp_limit_c=config.temp_limit_c).fit(rows)


def _normal_windows(config: MotorConfig, seed: int, count: int) -> List[Dict[str, float]]:
    sim = MotorSimulator(config, seed=seed)
    sim.advance(600.0)
    return [extract_features(w, config) for w in sim.sample_windows(count)]


def _fault_windows(
    config: MotorConfig, seed: int, class_name: str, bs: float, ims: float, eff: float, count: int
) -> List[Dict[str, float]]:
    sim = MotorSimulator(config, seed=seed)
    # Reach the fault's thermal steady state so temperature is representative.
    sim.set_state(MachineState(bearing_severity=bs, imbalance_severity=ims, cooling_efficiency=eff))
    sim.advance(300.0)
    return [extract_features(w, config) for w in sim.sample_windows(count)]


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# Deterministic per-class seed offsets (hash() is process-randomized and
# must never be used for reproducible evaluation).
_CLASS_SEED_OFFSET = {"bearing": 11, "imbalance": 23, "overtemperature": 37}


def evaluate_detection(config: MotorConfig, detector: BaselineDetector,
                       windows_per_group: int = 25) -> Dict[str, object]:
    """Fault-vs-normal detection metrics. Normal windows are true negatives;
    every faulted window is a positive. Seeds are disjoint from calibration."""
    eval_seed = 5001
    false_pos = 0
    true_neg = 0
    for row in _normal_windows(config, eval_seed, windows_per_group):
        true_neg += 1
        if detector.predict(row).is_anomaly:
            false_pos += 1
    fpr = false_pos / true_neg if true_neg else 0.0

    per_class = {}
    for class_name, bs, ims, eff in FAULT_GRID:
        seed = eval_seed + _CLASS_SEED_OFFSET[class_name]
        rows = _fault_windows(config, seed, class_name, bs, ims, eff, windows_per_group)
        detected = sum(detector.predict(r).is_anomaly for r in rows)
        entry = per_class.setdefault(class_name, {"tp": 0, "fn": 0})
        entry["tp"] += detected
        entry["fn"] += len(rows) - detected

    detection = {
        name: {**_prf(v["tp"], 0, v["fn"]), "positives": v["tp"] + v["fn"]}
        for name, v in per_class.items()
    }
    # Overall (all faulted windows pooled as positives).
    tot_tp = sum(v["tp"] for v in per_class.values())
    tot_fn = sum(v["fn"] for v in per_class.values())
    return {
        "normal_false_positive_rate": round(fpr, 4),
        "normal_windows": true_neg,
        "normal_false_alarms": false_pos,
        "per_fault_class_detection": detection,
        "overall_detection": {**_prf(tot_tp, false_pos, tot_fn)},
        "calibration_seed": CALIBRATION_SEED,
        "eval_seed": eval_seed,
        "note": "synthetic simulator data; disjoint calibration/eval seeds",
    }


def evaluate_classifier(config: MotorConfig) -> Dict[str, object]:
    from orbit.model.inference import WEIGHTS_PATH, FaultClassifier

    if not WEIGHTS_PATH.exists():
        return {"available": False, "reason": "run python -m orbit.model.train first"}
    clf = FaultClassifier.load("cpu")
    split = generate_split(name="eval", windows_per_class=80, seed=9009)
    probs = clf.predict_proba(split.X)
    preds = probs.argmax(axis=1)
    per_class = {}
    for index, name in enumerate(CLASSES):
        tp = int(((preds == index) & (split.y == index)).sum())
        fp = int(((preds == index) & (split.y != index)).sum())
        fn = int(((preds != index) & (split.y == index)).sum())
        per_class[name] = _prf(tp, fp, fn)
    acc = float((preds == split.y).mean())
    return {
        "available": True,
        "accuracy": round(acc, 4),
        "per_class": per_class,
        "eval_seed": 9009,
        "windows": int(split.X.shape[0]),
        "note": "held-out synthetic split; classifier trained on disjoint seed 1001",
    }


def _controller(config, detector, env) -> IncidentController:
    sim = MotorSimulator(config, seed=6006)
    sim.advance(600.0)
    return IncidentController(
        simulator=sim, detector=detector, engine=DiagnosisEngine(),
        environment=env, config=config, monitor_interval_s=60.0,
    )


def evaluate_recovery(config: MotorConfig, detector: BaselineDetector) -> Dict[str, object]:
    """Run the recovery scenarios and measure, per designed bucket, how often
    the deterministic verifier reached the intended terminal outcome.

    Each scenario declares the outcome it is DESIGNED to produce:
      - "resolve": a real, verified intervention should close the incident
        (concurrent faults are allowed to close over two re-diagnosis cycles
        as the dominant fault is repaired first).
      - "escalate": the pipeline must refuse to claim RESOLVED and escalate
        (failed intervention, or an unmet precondition).
      - "no_action": a healthy machine must not trigger any intervention.
    Ambiguity-induced escalation is exercised separately in the controller
    unit tests with a controlled ambiguous engine; here escalation is covered
    by the failed-intervention and precondition paths against the real engine.
    Deterministic seeds throughout.
    """
    scenarios = [
        ("A_bearing_repair", MachineState(bearing_severity=0.6),
         MaintenanceEnvironment(), "resolve", "RESOLVED"),
        ("A_imbalance_balance", MachineState(imbalance_severity=0.6),
         MaintenanceEnvironment(), "resolve", "RESOLVED"),
        ("A_overtemp_cooling", MachineState(cooling_efficiency=0.5),
         MaintenanceEnvironment(), "resolve", "RESOLVED"),
        ("concurrent_faults", MachineState(bearing_severity=0.3, imbalance_severity=0.55),
         MaintenanceEnvironment(), "resolve", "RESOLVED"),
        ("B_defective_spare", MachineState(bearing_severity=0.6),
         MaintenanceEnvironment(spare_bearing_quality=0.0), "escalate", "ESCALATED"),
        ("precond_no_spare", MachineState(bearing_severity=0.6),
         MaintenanceEnvironment(spare_bearing_available=False), "escalate", "ESCALATED"),
        ("D_healthy", MachineState(), MaintenanceEnvironment(), "no_action",
         "HEALTHY_NO_ACTION"),
    ]
    outcomes = {}
    counts = {"resolve": [0, 0], "escalate": [0, 0], "no_action": [0, 0]}
    latencies = []
    for name, state, env, bucket, expected in scenarios:
        controller = _controller(config, detector, env)
        controller.simulator.set_state(state)
        start = time.perf_counter()
        result = controller.run_cycle()
        actual = result.outcome
        # Allow a concurrent-fault incident to close over a second re-diagnosis
        # cycle: the dominant fault is repaired first, then the residual.
        if bucket == "resolve" and actual == "RESOLVED":
            second = controller.run_cycle()
            if second.outcome in ("RESOLVED", "ESCALATED") and second.incident:
                actual = second.outcome
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        match = actual == expected
        outcomes[name] = {"designed": bucket, "expected": expected,
                          "actual": actual, "match": match}
        counts[bucket][0] += 1
        if match:
            counts[bucket][1] += 1

    def _rate(bucket):
        total, hit = counts[bucket]
        return round(hit / total, 4) if total else None

    return {
        "scenarios": outcomes,
        "scenarios_total": len(scenarios),
        "resolve_designed": counts["resolve"][0],
        "resolve_correct": counts["resolve"][1],
        "recovery_success_rate": _rate("resolve"),
        "escalate_designed": counts["escalate"][0],
        "escalate_correct": counts["escalate"][1],
        "correct_escalation_rate": _rate("escalate"),
        "no_action_designed": counts["no_action"][0],
        "no_action_correct": counts["no_action"][1],
        "no_action_rate": _rate("no_action"),
        "mean_end_to_end_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "note": "deterministic seeds; recovery counted as success only when the "
                "deterministic verifier passed; rates are per designed bucket, "
                "not over all scenarios",
    }


def run_evaluation() -> Dict[str, object]:
    config = MotorConfig()
    detector = _fit_detector(config)
    results = {
        "synthetic_only": True,
        "data_source": "ORBIT motor simulator (synthetic). No real industrial data in this file.",
        "detection": evaluate_detection(config, detector),
        "classifier": evaluate_classifier(config),
        "recovery": evaluate_recovery(config, detector),
    }
    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    results = run_evaluation()
    print(json.dumps(results, indent=2))
    print(f"\nsaved {EVAL_OUT}")


if __name__ == "__main__":
    main()
