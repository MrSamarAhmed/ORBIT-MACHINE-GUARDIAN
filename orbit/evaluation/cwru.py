"""OPTIONAL evaluation on the Case Western Reserve University (CWRU)
Bearing Data Center dataset.

This module is entirely separate from the synthetic evaluation in
``orbit.evaluation.metrics`` — it never runs during normal application
startup, never downloads anything, and only executes when invoked
explicitly::

    python -m orbit.evaluation.cwru --data-dir data/cwru

Provenance and preprocessing (documented, not assumed):

* CWRU ``.mat`` files do not self-describe their fault type, and the
  official channel-number -> fault table lives on the CWRU website.
  Rather than fabricate that mapping, ORBIT uses a
  **directory-name-as-label** convention: the user places each
  recording under the folder that names its true fault class
  (``normal``, ``outer_race``, ``inner_race``, ``ball``). This keeps
  provenance explicit and user-controlled.

* Label mapping to ORBIT's synthetic classes is deliberately narrow:
  - ``normal``          -> ``normal``
  - ``outer_race``      -> ``bearing``  (the synthetic bearing mode is an
    outer-race defect; the classifier keys on the BPFO envelope ratio)
  - ``inner_race``/``ball`` are **excluded from classifier evaluation**
    (BPFI / BSF mechanisms differ from the outer-race model) but are
    still used for detector transfer evaluation and are counted in
    ``excluded_records``.

* CWRU has no imbalance class and no temperature channel, so the
  ``imbalance`` class and the thermal limit are not exercised here.

Leakage control:

* Splits are **recording-level**: each ``.mat`` file is hashed into
  train/val/test, and every window derived from a recording stays in
  that recording's split. Overlapping analysis windows therefore never
  cross a split boundary.

* The baseline detector is fit ONLY on ``normal`` training recordings.

Nothing here claims real-world industrial validation. The classifier is
trained on synthetic data; its CWRU numbers are a *transfer* measurement
and are reported with that caveat attached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio

from orbit.config import MotorConfig
from orbit.detection.baseline import BaselineDetector
from orbit.model.dataset import CLASSES, CLASS_TO_INDEX
from orbit.model.inference import WEIGHTS_PATH, FaultClassifier
from orbit.signalproc.features import FEATURE_NAMES, extract_features
from orbit.simulator.motor import MachineState, SensorWindow

# Folder name -> ORBIT class. None means "parse but exclude from the
# classifier" (unsupported synthetic label).
FOLDER_TO_CLASS: Dict[str, Optional[str]] = {
    "normal": "normal",
    "outer_race": "bearing",
    "inner_race": None,
    "ball": None,
}
SUPPORTED_FOLDERS = tuple(FOLDER_TO_CLASS)

WINDOW_S = 0.2
HOP_FRAC = 0.5
RPM_DEFAULT = 1772.0
FS_DEFAULT = 12000

EVAL_OUT = Path("runtime/evaluation/cwru_results.json")


@dataclass(frozen=True)
class CwruRecording:
    """One parsed CWRU .mat recording (drive-end channel)."""

    path: Path
    folder_label: str            # directory-derived label
    orbit_class: Optional[str]   # mapped ORBIT class, or None if unsupported
    rpm: float
    signal: np.ndarray           # 1-D drive-end accelerometer samples
    fs: float


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _read_channel(mat: dict, path: Path) -> np.ndarray:
    """Extract the drive-end time series from a loaded .mat dict."""
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if key.endswith("_DE_time"):
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size:
                return arr
    # Fallback: any '*_time' variable, else the first numeric vector.
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if key.endswith("_time"):
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size:
                return arr
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size:
            return arr
    raise ValueError(f"no usable time-series channel in {path.name}")


def _read_rpm(mat: dict, default: float) -> float:
    for key in mat:
        if key.startswith("__"):
            continue
        if key.upper().endswith("RPM"):
            arr = np.asarray(mat[key], dtype=float).ravel()
            if arr.size and np.isfinite(arr[0]) and arr[0] > 0:
                return float(arr[0])
    return float(default)


def parse_recordings(
    data_dir: Path,
    fs: float = FS_DEFAULT,
    rpm_default: float = RPM_DEFAULT,
) -> Tuple[List[CwruRecording], List[Dict[str, str]]]:
    """Parse every supported folder under ``data_dir``.

    Returns (recordings, excluded_records). Unreadable files, files with
    no usable channel, and folders outside the supported set are reported
    in ``excluded_records`` with a reason — never silently dropped."""
    data_dir = Path(data_dir)
    recordings: List[CwruRecording] = []
    excluded: List[Dict[str, str]] = []

    if not data_dir.exists():
        return [], [{"path": str(data_dir), "reason": "data directory does not exist"}]

    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in FOLDER_TO_CLASS:
            excluded.append({"path": str(folder), "reason": f"unsupported folder {folder.name!r}"})
            continue
        orbit_class = FOLDER_TO_CLASS[folder.name]
        for mat_path in sorted(folder.glob("*.mat")):
            try:
                mat = sio.loadmat(mat_path)
                signal = _read_channel(mat, mat_path)
                rpm = _read_rpm(mat, rpm_default)
                recordings.append(
                    CwruRecording(
                        path=mat_path,
                        folder_label=folder.name,
                        orbit_class=orbit_class,
                        rpm=rpm,
                        signal=signal,
                        fs=float(fs),
                    )
                )
            except Exception as exc:  # unreadable / malformed file
                excluded.append({"path": str(mat_path), "reason": f"parse error: {exc}"})
    return recordings, excluded


# ---------------------------------------------------------------------------
# Windowing + recording-level splits
# ---------------------------------------------------------------------------
def window_signal(
    signal: np.ndarray, window_samples: int, hop_samples: int
) -> List[np.ndarray]:
    """Split a 1-D signal into overlapping fixed-length windows."""
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    if window_samples < 16:
        raise ValueError("window_samples must be >= 16")
    if hop_samples < 1:
        raise ValueError("hop_samples must be >= 1")
    if signal.size < window_samples:
        return []
    starts = range(0, signal.size - window_samples + 1, hop_samples)
    return [signal[s:s + window_samples] for s in starts]


def _split_for(path: Path) -> str:
    """Deterministic recording-level split assignment (50/25/25)."""
    digest = int(hashlib.md5(str(path).encode()).hexdigest(), 16)
    bucket = digest % 100
    return "train" if bucket < 50 else ("val" if bucket < 75 else "test")


def _cwru_config(fs: float, rpm: float, window_samples: int) -> MotorConfig:
    """A MotorConfig whose derived window length matches the CWRU window.

    shaft_hz is the true shaft speed (rpm/60); bearing geometry stays
    6205. Temperature is absent from CWRU, so thermal features are set to
    ambient on the constructed SensorWindow and the thermal limit is not
    exercised."""
    window_s = window_samples / fs
    return MotorConfig(fs=int(fs), window_s=window_s, shaft_hz=rpm / 60.0)


def _features_for(rec: CwruRecording, window: np.ndarray, config: MotorConfig) -> Dict[str, float]:
    sw = SensorWindow(
        window_index=0,
        t_start_s=0.0,
        vibration=np.asarray(window, dtype=float),
        temperature_c=config.temp_ambient_c,  # CWRU has no temperature channel
        shaft_hz=config.shaft_hz,
        state=MachineState(),
    )
    return extract_features(sw, config)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> List[List[int]]:
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def _prf_from_cm(cm: List[List[int]], index: int) -> Dict[str, float]:
    tp = cm[index][index]
    fp = sum(cm[r][index] for r in range(len(cm)) if r != index)
    fn = sum(cm[index][c] for c in range(len(cm[index])) if c != index)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": tp + fn,
    }


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------
def evaluate_detector_transfer(
    recordings: Sequence[CwruRecording],
    window_samples: int,
    hop_samples: int,
    z_threshold: float = 3.5,
) -> Dict[str, object]:
    """Fit the baseline detector on normal-train windows; report FPR on
    normal-test windows and per-fault-type detection rate on all fault
    recordings. The detector is unsupervised, so every fault folder
    (including unsupported inner_race/ball) is scored here."""
    by_split: Dict[str, List[CwruRecording]] = {"train": [], "val": [], "test": []}
    for rec in recordings:
        by_split[_split_for(rec.path)].append(rec)

    train_normal = [r for r in by_split["train"] if r.folder_label == "normal"]
    calib_rows: List[Dict[str, float]] = []
    for rec in train_normal:
        config = _cwru_config(rec.fs, rec.rpm, window_samples)
        for w in window_signal(rec.signal, window_samples, hop_samples):
            calib_rows.append(_features_for(rec, w, config))

    if len(calib_rows) < 10:
        return {
            "available": False,
            "reason": f"need >=10 normal-train calibration windows, got {len(calib_rows)}",
        }

    detector = BaselineDetector(z_threshold=z_threshold).fit(calib_rows)

    # FPR on held-out normal test windows.
    test_normal = [r for r in by_split["test"] if r.folder_label == "normal"]
    fp = tn = 0
    for rec in test_normal:
        config = _cwru_config(rec.fs, rec.rpm, window_samples)
        for w in window_signal(rec.signal, window_samples, hop_samples):
            tn += 1
            if detector.predict(_features_for(rec, w, config)).is_anomaly:
                fp += 1

    # Per-fault-type detection rate across ALL fault recordings (any split).
    fault_stats: Dict[str, Dict[str, int]] = {}
    for rec in recordings:
        if rec.folder_label == "normal":
            continue
        stat = fault_stats.setdefault(rec.folder_label, {"tp": 0, "n": 0})
        config = _cwru_config(rec.fs, rec.rpm, window_samples)
        for w in window_signal(rec.signal, window_samples, hop_samples):
            stat["n"] += 1
            if detector.predict(_features_for(rec, w, config)).is_anomaly:
                stat["tp"] += 1

    return {
        "available": True,
        "calibration_windows": len(calib_rows),
        "calibration_source": "normal TRAIN recordings only",
        "normal_false_positive_rate": round(fp / tn, 4) if tn else None,
        "normal_test_windows": tn,
        "per_fault_type": {
            name: {"detection_rate": round(s["tp"] / s["n"], 4) if s["n"] else None,
                   "windows": s["n"]}
            for name, s in sorted(fault_stats.items())
        },
        "note": "detector fit on the provided normal TRAIN recordings (intended "
                "source: CWRU); evaluates whether ORBIT detection features "
                "transfer to real vibration. Not a claim of industrial "
                "deployment accuracy.",
    }


def evaluate_classifier_transfer(
    recordings: Sequence[CwruRecording],
    window_samples: int,
    hop_samples: int,
    weights_path: Path = WEIGHTS_PATH,
) -> Dict[str, object]:
    """Evaluate the synthetic-trained classifier on the mapped subset
    (normal vs outer_race). inner_race/ball are excluded (unsupported)
    and reported. A 3-class confusion matrix is still reported over the
    model's full output space."""
    if not Path(weights_path).exists():
        return {"available": False,
                "reason": f"trained weights not found at {weights_path}; "
                          f"run: python -m orbit.model.train"}

    mapped = [r for r in recordings if r.orbit_class is not None]
    excluded = sorted({r.folder_label for r in recordings if r.orbit_class is None})

    clf = FaultClassifier.load("cpu")
    X: List[np.ndarray] = []
    y: List[int] = []
    for rec in mapped:
        cls_idx = CLASS_TO_INDEX[rec.orbit_class]
        for w in window_signal(rec.signal, window_samples, hop_samples):
            X.append(np.asarray(w, dtype=np.float32))
            y.append(cls_idx)
    if not X:
        return {"available": False, "reason": "no mapped (normal/outer_race) windows found"}

    probs = clf.predict_proba(np.stack(X))
    preds = probs.argmax(axis=1)
    y_arr = np.array(y, dtype=np.int64)
    cm = _confusion_matrix(y_arr, preds, len(CLASSES))
    per_class = {name: _prf_from_cm(cm, idx) for idx, name in enumerate(CLASSES)}
    acc = float((preds == y_arr).mean())

    return {
        "available": True,
        "windows": int(len(X)),
        "recordings": len(mapped),
        "classes_present": sorted({rec.orbit_class for rec in mapped}),
        "excluded_folders": excluded,
        "excluded_reason": "inner_race/ball use BPFI/BSF mechanisms; the synthetic "
                           "bearing class is outer-race (BPFO) specific",
        "accuracy": round(acc, 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(CLASSES), "matrix": cm},
        "weights": str(weights_path),
        "note": "transfer evaluation: the classifier was TRAINED ON SYNTHETIC data; "
                "these numbers measure synthetic->real transfer, not real-world "
                "industrial accuracy.",
    }


def run_cwru_evaluation(
    data_dir: Path = Path("data/cwru"),
    fs: float = FS_DEFAULT,
    rpm_default: float = RPM_DEFAULT,
    window_s: float = WINDOW_S,
    hop_frac: float = HOP_FRAC,
    out_path: Path = EVAL_OUT,
) -> Dict[str, object]:
    window_samples = int(round(window_s * fs))
    hop_samples = max(1, int(round(window_samples * hop_frac)))

    recordings, excluded = parse_recordings(data_dir, fs=fs, rpm_default=rpm_default)
    result: Dict[str, object] = {
        "available": bool(recordings),
        "dataset": "Case Western Reserve University Bearing Data Center",
        "source": str(data_dir),
        "provenance": "directory-name-as-label; .mat drive-end channel; user places "
                      "recordings under the folder naming their true fault class",
        "fs_hz": fs,
        "window_s": window_s,
        "window_samples": window_samples,
        "hop_fraction": hop_frac,
        "rpm_default": rpm_default,
        "n_recordings": len(recordings),
        "recordings_by_folder": {
            name: sum(1 for r in recordings if r.folder_label == name)
            for name in sorted({r.folder_label for r in recordings})
        },
        "excluded_records": excluded,
        "label_mapping": {
            "normal": "normal",
            "outer_race": "bearing",
            "inner_race": "excluded (unsupported)",
            "ball": "excluded (unsupported)",
        },
        "split_policy": "recording-level (md5 of path, 50/25/25); overlapping windows "
                        "never cross splits",
        "imbalance_note": "CWRU has no imbalance class; the 'imbalance' synthetic class "
                         "is not exercised here",
        "temperature_note": "CWRU has no temperature channel; thermal features set to "
                            "ambient, thermal limit not exercised",
    }

    if not recordings:
        result["reason"] = (
            f"no CWRU recordings parsed from {data_dir}. Place .mat files under "
            f"{data_dir}/<normal|outer_race|inner_race|ball>/ and re-run."
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        return result

    result["detector_transfer"] = evaluate_detector_transfer(
        recordings, window_samples, hop_samples)
    result["classifier_transfer"] = evaluate_classifier_transfer(
        recordings, window_samples, hop_samples)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/cwru"))
    parser.add_argument("--fs", type=float, default=FS_DEFAULT, help="sampling rate of the .mat recordings")
    parser.add_argument("--rpm-default", type=float, default=RPM_DEFAULT,
                        help="shaft rpm used when a .mat file has no RPM variable")
    parser.add_argument("--window-s", type=float, default=WINDOW_S)
    parser.add_argument("--hop-frac", type=float, default=HOP_FRAC)
    parser.add_argument("--out", type=Path, default=EVAL_OUT)
    args = parser.parse_args()

    results = run_cwru_evaluation(
        data_dir=args.data_dir,
        fs=args.fs,
        rpm_default=args.rpm_default,
        window_s=args.window_s,
        hop_frac=args.hop_frac,
        out_path=args.out,
    )
    print(json.dumps(results, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
