"""Execution-device resolution for torch workloads.

Resolution order (auto):
1. CUDA/ROCm — on ROCm builds ``torch.version.hip`` is set and CUDA is
   available; reported as "rocm".
2. MPS — Apple Silicon, local development only (never claimed as AMD).
3. CPU — always available, the development and fallback path.

Fail-closed: requesting a device that is not actually available raises
with a clear explanation instead of silently falling back, so a
benchmark can never misreport where it ran.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device_type: str        # "cpu" | "cuda" | "mps"
    platform: str           # "cpu" | "rocm" | "cuda" | "mps"
    device_name: str        # human-readable hardware name
    torch_version: str
    hip_version: Optional[str]
    cuda_version: Optional[str]

    @property
    def torch_device(self) -> torch.device:
        if self.device_type == "cuda":
            return torch.device("cuda", 0)
        return torch.device(self.device_type)

    def as_dict(self) -> dict:
        return {
            "device_type": self.device_type,
            "platform": self.platform,
            "device_name": self.device_name,
            "torch_version": self.torch_version,
            "hip_version": self.hip_version,
            "cuda_version": self.cuda_version,
        }


def resolve_device(requested: Optional[str] = None) -> DeviceInfo:
    """Resolve the execution device. ``requested`` may be None (auto),
    'auto', 'cpu', 'cuda'/'rocm', or 'mps'."""
    requested = (requested or os.environ.get("ORBIT_DEVICE") or "auto").strip().lower()
    if requested == "rocm":
        requested = "cuda"  # ROCm accelerators are exposed via the CUDA API

    cuda_available = torch.cuda.is_available()
    mps_available = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    if requested == "auto":
        if cuda_available:
            device_type = "cuda"
        elif mps_available:
            device_type = "mps"
        else:
            device_type = "cpu"
    elif requested in ("cpu", "cuda", "mps"):
        device_type = requested
    else:
        raise ValueError(
            f"unknown device {requested!r}; expected auto|cpu|cuda|rocm|mps"
        )

    if device_type == "cuda" and not cuda_available:
        raise RuntimeError(
            "A CUDA/ROCm GPU was requested but torch.cuda.is_available() is "
            "False. On AMD hardware this usually means torch was installed "
            "without ROCm support (check torch.version.hip). Refusing to "
            "silently fall back to CPU."
        )
    if device_type == "mps" and not mps_available:
        raise RuntimeError("MPS requested but not available on this system.")

    if device_type == "cuda":
        hip = getattr(torch.version, "hip", None)
        platform = "rocm" if hip is not None else "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif device_type == "mps":
        platform = "mps"
        device_name = "Apple Silicon (MPS)"
    else:
        platform = "cpu"
        device_name = _cpu_name()

    return DeviceInfo(
        device_type=device_type,
        platform=platform,
        device_name=device_name,
        torch_version=torch.__version__,
        hip_version=getattr(torch.version, "hip", None),
        cuda_version=getattr(torch.version, "cuda", None),
    )


def _cpu_name() -> str:
    import platform

    return f"{platform.machine()} CPU ({platform.processor() or 'unknown'})"


def describe_environment() -> dict:
    """Environment snapshot for benchmark records."""
    import platform
    import sys

    info = resolve_device()
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        **info.as_dict(),
    }
