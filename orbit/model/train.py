"""Train the fault classifier on synthetic simulator data.

Usage: ``python -m orbit.model.train [--epochs 12] [--device auto]``

Training runs on CPU by default for full reproducibility (seeded torch,
seeded data). Artifacts land in ``runtime/model/``: weights state dict
plus a metrics JSON with per-split accuracy and per-class F1 on the
held-out test split (seeds disjoint from training — no leakage).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from orbit.model.cnn import FaultCNN
from orbit.model.dataset import CLASSES, generate_all_splits
from orbit.model.device import resolve_device

MODEL_DIR = Path("runtime/model")
SEED = 42


def _loader(split, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(split.X), torch.from_numpy(split.y)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray]:
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += int((preds == y).sum())
            total += y.numel()
            all_preds.append(preds.cpu().numpy())
    return correct / max(total, 1), np.concatenate(all_preds) if all_preds else np.array([])


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = {}
    for index, name in enumerate(CLASSES):
        tp = int(((y_pred == index) & (y_true == index)).sum())
        fp = int(((y_pred == index) & (y_true != index)).sum())
        fn = int(((y_pred != index) & (y_true == index)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int((y_true == index).sum()),
        }
    return report


def train(epochs: int = 12, batch_size: int = 64, device_request: str = "cpu") -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device_info = resolve_device(device_request)
    device = device_info.torch_device
    if device_info.platform == "mps":
        # Keep training fully deterministic: run on CPU even if MPS exists.
        device = torch.device("cpu")

    train_split, val_split, test_split = generate_all_splits()
    train_loader = _loader(train_split, batch_size, shuffle=True)
    val_loader = _loader(val_split, batch_size, shuffle=False)
    test_loader = _loader(test_split, batch_size, shuffle=False)

    model = FaultCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * y.numel()
        val_acc, _ = _evaluate(model, val_loader, device)
        history.append(
            {"epoch": epoch + 1, "train_loss": round(running_loss / len(train_split), 4),
             "val_accuracy": round(val_acc, 4)}
        )
        print(f"epoch {epoch + 1}/{epochs} loss={history[-1]['train_loss']:.4f} "
              f"val_acc={val_acc:.4f}")

    test_acc, test_preds = _evaluate(model, test_loader, device)
    report = _per_class_f1(test_split.y, test_preds)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODEL_DIR / "fault_cnn.pt"
    torch.save(model.state_dict(), weights_path)
    metrics = {
        "model": "FaultCNN",
        "parameters": model.num_parameters(),
        "training_device": str(device),
        "seed": SEED,
        "epochs": epochs,
        "class_counts": {
            "train": train_split.class_counts,
            "val": val_split.class_counts,
            "test": test_split.class_counts,
        },
        "split_seeds": {"train": train_split.seeds_used, "val": val_split.seeds_used,
                        "test": test_split.seeds_used},
        "data_source": "synthetic motor simulator (ORBIT), seeds disjoint per split",
        "history": history,
        "test_accuracy": round(test_acc, 4),
        "test_per_class": report,
    }
    metrics_path = MODEL_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"test accuracy: {test_acc:.4f} | saved {weights_path} and {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu",
                        help="auto|cpu|cuda|rocm|mps (training is CPU-deterministic by default)")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, device_request=args.device)


if __name__ == "__main__":
    main()
