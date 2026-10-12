"""Device-abstracted inference for the fault classifier.

The same trained weights are evaluated on any resolved device; CPU and
GPU paths are logically identical (same model, same pre/postprocessing).
``parity_check`` validates this numerically between two devices.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from orbit.model.cnn import FaultCNN
from orbit.model.dataset import CLASSES
from orbit.model.device import DeviceInfo, resolve_device

WEIGHTS_PATH = Path("runtime/model/fault_cnn.pt")


class FaultClassifier:
    def __init__(self, model: FaultCNN, device_info: DeviceInfo):
        self.model = model
        self.device_info = device_info
        self.model.to(device_info.torch_device)
        self.model.eval()

    @classmethod
    def load(cls, requested: Optional[str] = None,
             weights_path: Path = WEIGHTS_PATH) -> "FaultClassifier":
        if not weights_path.exists():
            raise FileNotFoundError(
                f"trained weights not found at {weights_path}; "
                f"run: python -m orbit.model.train"
            )
        device_info = resolve_device(requested)
        model = FaultCNN()
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return cls(model, device_info)

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Class probabilities for a batch of windows (n, samples)."""
        if windows.ndim != 2:
            raise ValueError(f"expected 2-D (n, samples) array, got shape {windows.shape}")
        X = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
        X = X.to(self.device_info.torch_device)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def classify(self, windows: np.ndarray) -> Dict[str, object]:
        probs = self.predict_proba(windows)
        mean_probs = probs.mean(axis=0)
        best = int(mean_probs.argmax())
        return {
            "predicted_class": CLASSES[best],
            "class_probabilities": {
                name: round(float(mean_probs[i]), 4) for i, name in enumerate(CLASSES)
            },
            "windows": int(probs.shape[0]),
            "device": self.device_info.as_dict(),
        }


def parity_check(
    windows: np.ndarray,
    device_a: Optional[str] = None,
    device_b: str = "cpu",
    weights_path: Path = WEIGHTS_PATH,
    atol: float = 1e-4,
) -> Dict[str, object]:
    """Verify two devices produce identical outputs within tolerance."""
    clf_a = FaultClassifier.load(device_a, weights_path)
    clf_b = FaultClassifier.load(device_b, weights_path)
    probs_a = clf_a.predict_proba(windows)
    probs_b = clf_b.predict_proba(windows)
    max_diff = float(np.abs(probs_a - probs_b).max())
    return {
        "device_a": clf_a.device_info.as_dict(),
        "device_b": clf_b.device_info.as_dict(),
        "max_abs_difference": max_diff,
        "atol": atol,
        "passed": bool(max_diff <= atol),
    }


def available_devices() -> list:
    """Device strings usable on this machine (for parity tests/benchmarks)."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices
