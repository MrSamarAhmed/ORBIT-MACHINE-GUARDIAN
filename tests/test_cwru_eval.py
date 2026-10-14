"""Tests for the OPTIONAL CWRU external-dataset evaluation.

Fixtures are tiny synthetic .mat files written with scipy.io.savemat —
they validate the parsing/windowing/split/metric machinery without
needing the real CWRU download. No test claims real CWRU performance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from orbit.evaluation import cwru

FS = 12000
N = 8000  # samples per fixture recording (~0.67 s -> several 0.2 s windows)


def _normal_signal(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 0.05, N)


def _impulsive_signal(rng: np.random.Generator, period: int = 134) -> np.ndarray:
    """Outer-race-like: periodic impulses ringing a resonance, on noise."""
    x = rng.normal(0.0, 0.05, N)
    t = np.arange(N) / FS
    for k in range(0, N // period):
        tk = k * period / FS
        dt = t - tk
        resp = np.exp(-dt / 0.004) * np.sin(2 * np.pi * 3400 * dt)
        resp[dt < 0] = 0.0
        x += 1.5 * resp
    return x


def _write_mat(path: Path, signal: np.ndarray, rpm: float = 1772.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = path.stem
    sio.savemat(path, {f"X{key}_DE_time": signal, f"X{key}RPM": np.array([[rpm]])})


@pytest.fixture()
def cwru_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(123)
    data = tmp_path / "cwru"
    # 16 normal recordings so the recording-level train split reliably
    # holds >=2 (the md5 split is path-dependent; volume removes flakiness).
    for i in range(16):
        _write_mat(data / "normal" / f"n{i}.mat", _normal_signal(rng))
    for i in range(3):
        _write_mat(data / "outer_race" / f"or{i}.mat", _impulsive_signal(rng))
        _write_mat(data / "inner_race" / f"ir{i}.mat", _impulsive_signal(rng))
    # one unsupported folder + one corrupt file to exercise exclusion paths
    _write_mat(data / "ball" / "b0.mat", _impulsive_signal(rng))
    (data / "normal" / "broken.mat").write_bytes(b"not a mat file")
    return data


# --- parsing ---------------------------------------------------------------
def test_parse_recordings_reads_channels_and_rpm(cwru_dir):
    recs, excluded = cwru.parse_recordings(cwru_dir, fs=FS)
    assert len(recs) == 23  # 16 normal + 3 outer_race + 3 inner_race + 1 ball
    by_folder = {}
    for r in recs:
        by_folder.setdefault(r.folder_label, []).append(r)
    assert set(by_folder) == {"normal", "outer_race", "inner_race", "ball"}
    assert len(by_folder["normal"]) == 16
    assert len(by_folder["inner_race"]) == 3
    assert all(r.signal.ndim == 1 and r.signal.size == N for r in recs)
    assert all(r.rpm == 1772.0 for r in recs)
    assert all(r.fs == float(FS) for r in recs)
    # orbit_class mapping: normal->normal, outer_race->bearing, inner_race/ball->None
    cls = {r.folder_label: r.orbit_class for r in recs}
    assert cls["normal"] == "normal"
    assert cls["outer_race"] == "bearing"
    assert cls["inner_race"] is None
    assert cls["ball"] is None
    # corrupt file reported, not dropped silently
    reasons = [e["reason"] for e in excluded]
    assert any("parse error" in r for r in reasons)


def test_parse_recordings_missing_dir(tmp_path):
    recs, excluded = cwru.parse_recordings(tmp_path / "nope", fs=FS)
    assert recs == []
    assert excluded and "does not exist" in excluded[0]["reason"]


def test_parse_unsupported_folder_reported(tmp_path):
    data = tmp_path / "cwru"
    _write_mat(data / "weird_fault" / "w0.mat", np.zeros(N))
    recs, excluded = cwru.parse_recordings(data, fs=FS)
    assert recs == []
    assert any("unsupported folder" in e["reason"] for e in excluded)


# --- windowing -------------------------------------------------------------
def test_windowing_counts_and_overlap():
    sig = np.arange(1000, dtype=float)
    wins = cwru.window_signal(sig, window_samples=200, hop_samples=100)
    assert len(wins) == 9  # (1000-200)/100 + 1
    assert all(len(w) == 200 for w in wins)
    assert wins[0][0] == 0.0 and wins[1][0] == 100.0  # 50% overlap


def test_windowing_too_short_returns_empty():
    assert cwru.window_signal(np.zeros(100), 200, 100) == []


# --- split integrity -------------------------------------------------------
def test_recording_level_split_no_overlap(cwru_dir):
    recs, _ = cwru.parse_recordings(cwru_dir, fs=FS)
    splits = {str(r.path): cwru._split_for(r.path) for r in recs}
    # every recording in exactly one split
    assert all(s in {"train", "val", "test"} for s in splits.values())
    # deterministic
    assert splits == {str(r.path): cwru._split_for(r.path) for r in recs}


# --- metrics ---------------------------------------------------------------
def test_confusion_matrix_and_prf():
    y_true = np.array([0, 0, 1, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    cm = cwru._confusion_matrix(y_true, y_pred, 3)
    assert cm == [[1, 1, 0], [1, 2, 0], [0, 0, 1]]
    prf0 = cwru._prf_from_cm(cm, 0)
    assert prf0["precision"] == 0.5
    assert prf0["recall"] == 0.5
    assert prf0["support"] == 2


# --- full pipeline ---------------------------------------------------------
def test_full_run_on_fixture_produces_expected_keys(cwru_dir, tmp_path):
    out = tmp_path / "cwru_results.json"
    res = cwru.run_cwru_evaluation(
        data_dir=cwru_dir, fs=FS, out_path=out,
        window_s=0.2, hop_frac=0.5,
    )
    assert res["available"] is True
    assert res["n_recordings"] == 23
    assert out.exists()

    det = res["detector_transfer"]
    assert det["available"] is True
    assert "normal_false_positive_rate" in det
    assert "per_fault_type" in det
    # fixture impulses are detectable -> outer_race detection rate is high
    assert det["per_fault_type"]["outer_race"]["detection_rate"] >= 0.9

    clf = res["classifier_transfer"]
    assert clf["available"] is True
    assert clf["confusion_matrix"]["labels"] == ["normal", "bearing", "imbalance"]
    assert "inner_race" in clf["excluded_folders"]
    assert "ball" in clf["excluded_folders"]
    assert clf["windows"] > 0
    # synthetic-trained transfer on a trivial fixture should be near-perfect,
    # but the caveat string must be present regardless.
    assert "SYNTHETIC" in clf["note"]


def test_missing_dataset_writes_unavailable(tmp_path):
    out = tmp_path / "cwru_results.json"
    res = cwru.run_cwru_evaluation(data_dir=tmp_path / "absent", fs=FS, out_path=out)
    assert res["available"] is False
    assert "Place .mat files" in res["reason"]
    assert json.loads(out.read_text())["available"] is False


def test_classifier_transfer_missing_weights(cwru_dir, tmp_path):
    from orbit.evaluation.cwru import evaluate_classifier_transfer
    recs, _ = cwru.parse_recordings(cwru_dir, fs=FS)
    res = evaluate_classifier_transfer(
        recs, window_samples=int(0.2 * FS), hop_samples=int(0.1 * FS),
        weights_path=tmp_path / "no_weights.pt",
    )
    assert res["available"] is False
    assert "not found" in res["reason"]
