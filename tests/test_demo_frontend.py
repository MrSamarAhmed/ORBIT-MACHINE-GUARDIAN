"""Frontend-acceptance regression tests for the redesigned ORBIT demo.

Exercised through Streamlit's AppTest (no browser, no live API credentials):
the real service path runs in-process and the rendered element tree is
inspected. These tests lock the judge-facing presentation rules:

- the simulation disclosure and the navigation tabs exist,
- readable feature labels are shown while the raw recorded field names stay
  accessible in the evidence views / ledger,
- the verification table lists failed criteria first and matches the
  verifier's recorded results exactly,
- a failed recovery shows the prominent FAILED RECOVERY explanation and is
  never presented as a successful recovery,
- session history marks the current run apart from historical runs,
- the CPU-vs-MPS latency chart is drawn from exactly the recorded benchmark
  measurements (unusual CPU timings preserved, as measured),
- Apple Silicon MPS is labeled as MPS, never as AMD.

Only presentation behaviour is asserted; no backend semantics are re-tested
here (see test_demo_scenarios.py / test_ledger.py for that).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest

from orbit.api import service
from orbit.demo.scenarios import DEMO_SCENARIOS

APP_PATH = Path(__file__).resolve().parents[1] / "orbit" / "demo" / "app.py"
PRESET_LABEL = {s.key: s.label for s in DEMO_SCENARIOS}
DEV_NAMES = {"cpu": "CPU", "mps": "Apple Silicon (MPS)",
             "cuda": "CUDA GPU", "rocm": "AMD ROCm GPU"}


# --- AppTest helpers -------------------------------------------------------
def _render() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception, f"initial render failed: {at.exception}"
    return at


def _run_preset(at: AppTest, key: str) -> None:
    label = PRESET_LABEL[key]
    at.selectbox[0].select(label)
    at.run()
    at.button[0].click()
    at.run()
    assert not at.exception, f"pressed run ({key}) raised: {at.exception}"


def _md_text(at: AppTest) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def _captions_text(at: AppTest) -> str:
    return "\n".join(str(c.value) for c in at.caption)


def _metrics(at: AppTest) -> Dict[str, object]:
    return {m.label: m.value for m in at.metric}


def _find_df(at: AppTest, required: List[str]) -> pd.DataFrame:
    for el in at.dataframe:
        df = el.value
        if all(col in df.columns for col in required):
            return df
    raise AssertionError(f"no rendered dataframe with columns {required}")


def _latency_chart_element(at: AppTest):
    for el in at.get("arrow_vega_lite_chart"):
        spec = json.loads(el.proto.spec)
        y = spec.get("encoding", {}).get("y", {})
        if y.get("field") == "latency_mean_ms":
            return el, spec
    return None, None


def _chart_dataframe(el) -> pd.DataFrame:
    payload = el.proto.datasets[0].data.data
    return pa.ipc.open_stream(payload).read_all().to_pandas()


# --- Navigation and simulation disclosure ----------------------------------
def test_navigation_tabs_and_simulation_disclosure():
    at = _render()
    labels = [t.label for t in at.tabs]
    assert labels == [
        "02 \u00b7 Incident diagnosis & recovery",
        "03 \u00b7 Sensor trends",
        "04 \u00b7 Evaluation & benchmarks",
        "05 \u00b7 Session history",
    ]
    md = _md_text(at)
    assert "DEMO MODE - SIMULATED" in md
    assert "No real equipment is connected." in md
    assert "Executing an action never implies resolution" in md


# --- Failed recovery (preset 3) --------------------------------------------
def test_failed_recovery_is_prominent_and_never_presented_as_resolved():
    at = _render()
    _run_preset(at, "failed_recovery")

    metrics = _metrics(at)
    assert metrics["Outcome"] == "ESCALATED"
    assert metrics["Recovery"] == "NOT RECOVERED"
    assert metrics["Machine condition"] == "ANOMALOUS"

    md = _md_text(at)
    assert "FAILED RECOVERY" in md
    assert "escalated for" in md and "human intervention" in md
    assert "unresolved" in md
    assert "VERIFICATION FAILED" in md
    assert "VERIFICATION PASSED" not in md
    assert "ACTION EXECUTED != INCIDENT RESOLVED" in _captions_text(at)
    # The designed escalation still matched the scenario expectation; machine
    # recovery is tracked separately and reported as NOT verified.
    assert "SCENARIO EXPECTATION MET" in md
    assert "Machine recovery: NOT verified" in _captions_text(at)


def test_failed_recovery_verification_table_matches_verifier_failed_first():
    at = _render()
    _run_preset(at, "failed_recovery")

    inc = at.session_state["last_result"]["cycles"][-1]["incident"]
    ref = inc["verification"]["results"]
    n_fail = sum(1 for r in ref if not r["passed"])
    assert n_fail >= 1

    vdf = _find_df(at, ["Criterion", "Expected", "Observed", "Window", "Result"])
    assert list(vdf.columns) == ["Criterion", "Expected", "Observed", "Window", "Result"]
    results = list(vdf["Result"])
    assert results.count("FAIL") == n_fail
    assert results.count("PASS") == len(ref) - n_fail
    # Failed criteria are listed first.
    first_pass = results.index("PASS") if "PASS" in results else len(results)
    assert all(r == "FAIL" for r in results[:first_pass])
    # Readable criterion names, observed values exactly the recorded ones.
    assert "Envelope BPFO ratio" in set(vdf["Criterion"])
    assert set(vdf["Observed"]) == {r["value"] for r in ref}
    assert "Failed criteria are listed first" in _captions_text(at)

    # The raw recorded rows (original field names) stay accessible in the ledger.
    found_raw = False
    for el in at.dataframe:
        text = el.value.astype(str)
        if text.apply(lambda c: c.str.contains("env_bpfo_ratio", regex=False)).any().any():
            found_raw = True
            break
    assert found_raw, "raw feature key env_bpfo_ratio not found in any ledger view"


# --- Successful recovery (preset 2) ----------------------------------------
def test_successful_recovery_shows_verified_verdict():
    at = _render()
    _run_preset(at, "bearing_recovery")

    metrics = _metrics(at)
    assert metrics["Outcome"] == "RESOLVED"
    assert metrics["Recovery"] == "VERIFIED"

    md = _md_text(at)
    assert "VERIFICATION PASSED" in md
    assert "FAILED RECOVERY" not in md

    vdf = _find_df(at, ["Criterion", "Result"])
    assert set(vdf["Result"]) == {"PASS"}


# --- Healthy run (preset 1) -------------------------------------------------
def test_healthy_run_takes_no_action():
    at = _render()
    _run_preset(at, "healthy")

    metrics = _metrics(at)
    assert metrics["Outcome"] == "HEALTHY_NO_ACTION"
    assert metrics["Recovery"] == "NO ACTION"
    assert metrics["Latest anomaly"] == "NONE"
    assert "No incident was opened" in _captions_text(at)


# --- Session history --------------------------------------------------------
def test_history_marks_current_run_apart_from_historical():
    at = _render()
    _run_preset(at, "healthy")
    _run_preset(at, "bearing_recovery")

    hist = at.session_state["history"]
    assert {h["run"] for h in hist} == {1, 2}

    hdf = _find_df(at, ["Current", "Run", "Scenario", "Outcome", "Incident"])
    assert list(hdf.columns) == ["Run", "Scenario", "Expected", "Outcome",
                                 "Incident", "Current"]
    assert hdf["Current"].tolist().count("current") == sum(
        1 for h in hist if h["run"] == 2)
    assert set(hdf.loc[hdf["Current"] == "current", "Run"]) == {2}
    # The current run's outcome is the one shown in the overview.
    assert _metrics(at)["Outcome"] == "RESOLVED"


# --- Benchmarks and evaluation ---------------------------------------------
@pytest.mark.parametrize("platforms", [("cpu",), ("cpu", "mps"), ("cpu", "cuda"), ("cpu", "rocm")])
def test_latency_chart_is_drawn_from_recorded_measurements(monkeypatch, platforms):
    bench = {"results": [
        {"device": {"platform": platform, "device_type": platform,
                    "device_name": DEV_NAMES[platform]},
         "measurements": [
             {"batch_size": 1, "latency_mean_ms": 7.0,
              "throughput_windows_per_s": 1000 / 7},
             {"batch_size": 8, "latency_mean_ms": 3.0,
              "throughput_windows_per_s": 8000 / 3},
         ]}
        for platform in platforms
    ]}
    monkeypatch.setattr(service, "latest_benchmark", lambda: bench)
    at = _render()

    el, spec = _latency_chart_element(at)
    assert el is not None, "CPU/MPS latency chart not rendered"
    x = spec["encoding"]["x"]
    assert x.get("field") == "batch_size"
    assert x.get("scale", {}).get("type") == "log"

    chart_df = _chart_dataframe(el)
    expected = {(DEV_NAMES.get(str(r["device"]["platform"]),
                               str(r["device"].get("device_name"))),
                 m["batch_size"], m["latency_mean_ms"], m["throughput_windows_per_s"])
                for r in bench["results"] for m in r["measurements"]}
    actual = set(zip(chart_df["device"], chart_df["batch_size"],
                     chart_df["latency_mean_ms"], chart_df["throughput_windows_per_s"]))
    assert actual == expected, "chart data must be the recorded measurements, unmodified"
    # Unusual CPU timings are preserved exactly (no smoothing/correction).
    cpu = chart_df[chart_df["device"] == "CPU"].sort_values("batch_size")
    assert cpu["latency_mean_ms"].tolist() == [m["latency_mean_ms"]
                                               for r in bench["results"]
                                               if r["device"]["platform"] == "cpu"
                                               for m in r["measurements"]]

    caps = _captions_text(at)
    assert "without smoothing or correction" in caps
    if "mps" in platforms:
        assert "Apple Silicon (MPS)" in caps
    has_amd = any(str(r["device"].get("device_type", "")).lower() in ("rocm", "cuda")
                  for r in bench["results"])
    if not has_amd:
        assert "No AMD ROCm device was available" in caps
        assert "labeled as MPS, never as AMD" in caps


def test_evaluation_metrics_separated_from_current_incident(monkeypatch):
    monkeypatch.setattr(service, "evaluation_results", lambda: {
        "data_source": "Synthetic presentation fixture, not measured performance",
        "detection": {"note": "UI layout test only"},
        "recovery": {
            "recovery_success_rate": 1.0, "resolve_correct": 1, "resolve_designed": 1,
            "correct_escalation_rate": 1.0, "escalate_correct": 1, "escalate_designed": 1,
            "mean_end_to_end_latency_ms": 12.0,
        },
    })
    at = _render()
    _run_preset(at, "failed_recovery")
    caps = _captions_text(at)
    assert "Aggregate results over the fixed evaluation scenario suite" in caps
    assert "not the single incident summarized in section 01" in caps
    # The aggregate metrics are clearly labeled as evaluation-suite results.
    metrics = _metrics(at)
    assert "Recovery success rate" in metrics
