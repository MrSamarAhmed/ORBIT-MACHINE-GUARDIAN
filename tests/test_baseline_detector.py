"""Baseline detector tests: calibration, false alarms, fault detection."""
from __future__ import annotations

import numpy as np
import pytest

from orbit.detection.baseline import BaselineDetector
from orbit.signalproc.features import FEATURE_NAMES
from orbit.simulator.motor import MotorSimulator
from tests.conftest import SCENARIOS_STATE
from tests.test_simulator import sample_rows


HELD_OUT_HEALTHY = 30


@pytest.fixture(scope="module")
def held_out_healthy_rows(config):
    # Different seed than the calibration fixture in conftest (7).
    sim = MotorSimulator(config, seed=99)
    sim.advance(600.0)
    return sample_rows(sim, config, HELD_OUT_HEALTHY)


class TestCalibrationValidation:
    def test_rejects_too_few_rows(self, config):
        sim = MotorSimulator(config, seed=7)
        sim.advance(600.0)
        rows = sample_rows(sim, config, 5)
        with pytest.raises(ValueError, match="calibration rows"):
            BaselineDetector().fit(rows)

    def test_rejects_inconsistent_keys(self, healthy_rows):
        bad = [dict(healthy_rows[0]) for _ in range(12)]
        bad[0].pop("vib_rms")
        with pytest.raises(ValueError, match="same feature keys|missing features"):
            BaselineDetector().fit(bad)

    def test_rejects_missing_feature(self, healthy_rows):
        bad = [dict(r) for r in healthy_rows[:12]]
        for r in bad:
            r.pop("temp_c")
        with pytest.raises(ValueError, match="missing features"):
            BaselineDetector().fit(bad)

    def test_rejects_nan_values(self, healthy_rows):
        bad = [dict(r) for r in healthy_rows[:12]]
        bad[3]["vib_rms"] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            BaselineDetector().fit(bad)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            BaselineDetector().predict({name: 0.0 for name in FEATURE_NAMES})

    def test_predict_missing_feature_raises(self, detector, healthy_rows):
        row = dict(healthy_rows[0])
        row.pop("vib_kurtosis")
        with pytest.raises(ValueError, match="missing features"):
            detector.predict(row)

    def test_predict_nonfinite_raises(self, detector, healthy_rows):
        row = dict(healthy_rows[0])
        row["vib_rms"] = float("inf")
        with pytest.raises(ValueError, match="not finite"):
            detector.predict(row)


class TestHealthyOperation:
    def test_no_false_alarms_on_held_out(self, detector, held_out_healthy_rows):
        results = [detector.predict(r) for r in held_out_healthy_rows]
        flagged = [r for r in results if r.is_anomaly]
        assert not flagged, f"false alarms: {[r.primary_breach for r in flagged]}"

    def test_calibration_data_itself_is_normal(self, detector, healthy_rows):
        assert not any(detector.predict(r).is_anomaly for r in healthy_rows)


class TestFaultDetection:
    @pytest.mark.parametrize(
        "scenario_name",
        ["bearing", "imbalance", "overtemperature"],
    )
    def test_all_faults_detected(self, config, detector, scenario_name):
        sim = MotorSimulator(config, seed=31)
        sim.set_state(SCENARIOS_STATE[scenario_name])
        if scenario_name == "overtemperature":
            sim.advance(600.0)  # let temperature reach the alarm limit
        rows = sample_rows(sim, config, 10)
        detections = [detector.predict(r).is_anomaly for r in rows]
        assert all(detections), (
            f"{scenario_name}: only {sum(detections)}/10 windows detected"
        )

    def test_breach_carrying_evidence(self, config, detector):
        sim = MotorSimulator(config, seed=31)
        sim.set_state(SCENARIOS_STATE["bearing"])
        rows = sample_rows(sim, config, 10)
        result = detector.predict(rows[0])
        assert result.is_anomaly
        assert result.breaches
        for breach in result.breaches:
            evidence = breach.as_evidence()
            assert evidence["feature"] in FEATURE_NAMES
            assert isinstance(evidence["value"], float)

    def test_overtemperature_flags_limit(self, config, detector):
        sim = MotorSimulator(config, seed=31)
        sim.set_state(SCENARIOS_STATE["overtemperature"])
        sim.advance(600.0)
        rows = sample_rows(sim, config, 5)
        for row in rows:
            result = detector.predict(row)
            assert result.temp_limit_exceeded
            assert any(b.feature == "temp_c" for b in result.breaches)


class TestSeverityMonotonicity:
    def test_higher_bearing_severity_stronger_signal(self, config):
        ratios = {}
        for severity in (0.2, 0.6, 1.0):
            sim = MotorSimulator(config, seed=17)
            sim.set_state(
                SCENARIOS_STATE["bearing"].__class__(bearing_severity=severity)
            )
            rows = sample_rows(sim, config, 10)
            ratios[severity] = np.mean([r["env_bpfo_ratio"] for r in rows])
        assert ratios[0.2] < ratios[0.6] < ratios[1.0]
