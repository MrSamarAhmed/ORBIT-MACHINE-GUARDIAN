"""Model package with lazy imports.

``device``/``dataset`` are lightweight, but ``inference`` and ``cnn`` pull
in torch at import time. The deterministic pipeline and the hosted demo
must load without torch, so the torch-backed names are resolved lazily via
``__getattr__`` (PEP 562) and only import torch when first accessed.
"""
from __future__ import annotations

from orbit.model.device import (
    DeviceInfo,
    describe_environment,
    describe_runtime,
    resolve_device,
)
from orbit.model.dataset import CLASSES

_LAZY = {
    "DatasetSplit": "orbit.model.dataset",
    "generate_split": "orbit.model.dataset",
    "generate_all_splits": "orbit.model.dataset",
    "FaultClassifier": "orbit.model.inference",
    "parity_check": "orbit.model.inference",
    "available_devices": "orbit.model.inference",
}

__all__ = [
    "CLASSES",
    "DatasetSplit",
    "DeviceInfo",
    "FaultClassifier",
    "available_devices",
    "describe_environment",
    "describe_runtime",
    "generate_all_splits",
    "generate_split",
    "parity_check",
    "resolve_device",
]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
