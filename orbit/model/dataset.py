"""Synthetic labelled dataset generation for the fault classifier.

Splits are disjoint by simulator seed: the model is never trained and
evaluated on windows from the same seeded realization, so there is no
data leakage between splits.

Classes: normal, bearing (outer-race), imbalance. The classifier sees
only vibration; cooling degradation is vibration-invisible and remains
the responsibility of the baseline detector + temperature limit.

Every window draws a random operating load in [0.6, 1.2] so the model
must learn load-robust signatures rather than absolute amplitudes.
Fault severity is drawn from [0.2, 1.0]; milder faults are left to the
statistical detector because near-threshold labels would be noisy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from orbit.config import MotorConfig
from orbit.simulator.motor import MachineState, MotorSimulator

CLASSES: Tuple[str, ...] = ("normal", "bearing", "imbalance")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}

_MIN_SEVERITY = 0.2
_MAX_SEVERITY = 1.0
_LOAD_MIN = 0.6
_LOAD_MAX = 1.2


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    X: np.ndarray  # (n, window_samples) float32
    y: np.ndarray  # (n,) int64 class indices
    seeds_used: Tuple[int, ...]

    @property
    def class_counts(self) -> Dict[str, int]:
        return {name: int((self.y == index).sum()) for index, name in enumerate(CLASSES)}

    def __len__(self) -> int:
        return int(self.X.shape[0])


def _make_windows(
    config: MotorConfig,
    seed: int,
    count: int,
    class_name: str,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    sim = MotorSimulator(config, seed=seed)
    windows = []
    for _ in range(count):
        if class_name == "normal":
            state = MachineState(load=float(rng.uniform(_LOAD_MIN, _LOAD_MAX)))
        elif class_name == "bearing":
            state = MachineState(
                bearing_severity=float(rng.uniform(_MIN_SEVERITY, _MAX_SEVERITY)),
                load=float(rng.uniform(_LOAD_MIN, _LOAD_MAX)),
            )
        else:
            state = MachineState(
                imbalance_severity=float(rng.uniform(_MIN_SEVERITY, _MAX_SEVERITY)),
                load=float(rng.uniform(_LOAD_MIN, _LOAD_MAX)),
            )
        sim.set_state(state)
        windows.append(sim.sample_window().vibration.astype(np.float32))
    return windows


def generate_split(
    config: Optional[MotorConfig] = None,
    name: str = "train",
    windows_per_class: int = 500,
    seed: int = 1000,
) -> DatasetSplit:
    """Generate one split. Distinct ``seed`` values produce disjoint
    simulator realizations; callers must use disjoint seeds per split."""
    if windows_per_class < 1:
        raise ValueError("windows_per_class must be >= 1")
    config = config if config is not None else MotorConfig()
    rng = np.random.default_rng(seed)

    X: List[np.ndarray] = []
    y: List[int] = []
    for class_index, class_name in enumerate(CLASSES):
        for window in _make_windows(config, seed, windows_per_class, class_name, rng):
            X.append(window)
            y.append(class_index)
    X_array = np.stack(X)
    y_array = np.array(y, dtype=np.int64)
    return DatasetSplit(
        name=name, X=X_array, y=y_array, seeds_used=(seed,)
    )


def generate_all_splits(
    config: Optional[MotorConfig] = None,
    train_windows_per_class: int = 500,
    val_windows_per_class: int = 150,
    test_windows_per_class: int = 300,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Train/val/test splits with disjoint seeds (no shared realizations)."""
    return (
        generate_split(config, "train", train_windows_per_class, seed=1001),
        generate_split(config, "val", val_windows_per_class, seed=2002),
        generate_split(config, "test", test_windows_per_class, seed=3003),
    )
