"""Reproducible CPU vs GPU inference benchmark.

Usage: ``python -m orbit.model.benchmark [--out runtime/benchmark/]``

Measures model inference latency and throughput across batch sizes on
every device available in the current environment. If no GPU is
present, the report states this explicitly — it never fabricates GPU
numbers. Results (hardware, software versions, model config, warm-up,
timings) are written to JSON for the README and demo UI.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from orbit.model.cnn import FaultCNN
from orbit.model.dataset import generate_split
from orbit.model.device import describe_environment, resolve_device
from orbit.model.inference import WEIGHTS_PATH, parity_check

BATCH_SIZES = (1, 8, 64, 512, 2048)
WARMUP_ITERATIONS = 10
TIMED_ITERATIONS = 30
DATASET_SEED = 777


def _sync(device_type: str) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _bench_configuration(
    model: FaultCNN,
    X: np.ndarray,
    device_type: str,
    batch_size: int,
) -> Dict[str, object]:
    device = torch.device(device_type if device_type != "cuda" else "cuda:0")
    batch = torch.from_numpy(np.ascontiguousarray(X[:batch_size], dtype=np.float32)).to(device)
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            model(batch)
        _sync(device_type)
        timings = []
        for _ in range(TIMED_ITERATIONS):
            start = time.perf_counter()
            model(batch)
            _sync(device_type)
            timings.append(time.perf_counter() - start)
    mean_s = statistics.mean(timings)
    p50_s = statistics.median(timings)
    p99_s = sorted(timings)[min(len(timings) - 1, int(0.99 * len(timings)))]
    return {
        "batch_size": batch_size,
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
        "latency_mean_ms": round(mean_s * 1000, 3),
        "latency_p50_ms": round(p50_s * 1000, 3),
        "latency_p99_ms": round(p99_s * 1000, 3),
        "throughput_windows_per_s": round(batch_size / mean_s, 1),
    }


def run_benchmark(out_dir: Path = Path("runtime/benchmark")) -> Dict[str, object]:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"trained weights not found at {WEIGHTS_PATH}; run python -m orbit.model.train first"
        )
    env = describe_environment()

    # Identical workload for every device: fixed-seed synthetic windows.
    split = generate_split(name="benchmark", windows_per_class=max(BATCH_SIZES) // 3 + 1,
                           seed=DATASET_SEED)
    X = split.X

    from orbit.model.inference import available_devices

    devices = available_devices()
    results: List[Dict[str, object]] = []
    for device_type in devices:
        info = resolve_device(device_type)
        model = FaultCNN()
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        model.to(info.torch_device).eval()
        device_results = []
        for batch_size in BATCH_SIZES:
            if batch_size > len(X):
                continue
            device_results.append(
                _bench_configuration(model, X, device_type, batch_size)
            )
        results.append(
            {
                "device": info.as_dict(),
                "measurements": device_results,
            }
        )
        print(f"[{info.platform}] {info.device_name}")
        for m in device_results:
            print(
                f"  batch {m['batch_size']:>5}: mean {m['latency_mean_ms']:>9.3f} ms | "
                f"throughput {m['throughput_windows_per_s']:>10.1f} windows/s"
            )

    parity = None
    if len(devices) > 1:
        parity = parity_check(X[:64], device_a=devices[0], device_b=devices[-1])
        print(
            f"parity {devices[0]} vs {devices[-1]}: "
            f"max|diff|={parity['max_abs_difference']:.2e} "
            f"({'PASS' if parity['passed'] else 'FAIL'})"
        )

    gpu_devices = [r for r in results if r["device"]["device_type"] != "cpu"]
    report = {
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "environment": env,
        "model": {
            "name": "FaultCNN",
            "parameters": FaultCNN().num_parameters(),
            "weights": str(WEIGHTS_PATH),
            "input_shape": [None, split.X.shape[1]],
        },
        "workload": {
            "source": "synthetic motor simulator windows (fixed seed)",
            "dataset_seed": DATASET_SEED,
            "batch_sizes": list(BATCH_SIZES),
        },
        "devices_benchmarked": devices,
        "gpu_available": bool(gpu_devices),
        "results": results,
        "parity_check": parity,
        "notes": [],
    }
    if not gpu_devices:
        report["notes"].append(
            "No GPU present in this environment: CPU-only measurements. "
            "GPU numbers will be added from an AMD ROCm instance and are "
            "never extrapolated or fabricated."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runtime/benchmark"))
    args = parser.parse_args()
    run_benchmark(out_dir=args.out)


if __name__ == "__main__":
    main()
