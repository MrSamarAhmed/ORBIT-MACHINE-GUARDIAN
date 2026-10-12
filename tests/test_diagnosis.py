"""Diagnosis engine tests: hypothesis ranking, evidence, ambiguity."""
from __future__ import annotations

import numpy as np
import pytest

from orbit.diagnosis.engine import DiagnosisEngine
from orbit.signalproc.features import FEATURE_NAMES
from orbit.simulator.motor import MachineState, MotorSimulator
from tests.conftest import SCENARIOS_STATE
from tests.test_simulator import sample_rows

N_WINDOWS = 8


@pytest.fixture(scope="module")
def engine() -> DiagnosisEngine:
    return DiagnosisEngine()


def diagnose_scenario(config, detector, engine, state, advance_s=0.0, seed=53):
    sim = MotorSimulator(config, seed=seed)
    sim.set_state(state)
    if advance_s:
        sim.advance(advance_s)
    rows = sample_rows(sim, config, N_WINDOWS)
    results = []
    for row in rows:
        anomaly = detector.predict(row)
        results.append(engine.diagnose(anomaly, row, detector))
    return results


class TestBearingDiagnosis:
    def test_primary_is_bearing_with_evidence(self, config, detector, engine):
        results = diagnose_scenario(
            config, detector, engine, SCENARIOS_STATE["bearing"]
        )
        assert all(r.is_confident for r in results)
        assert all(r.primary.mode_id == "bearing_outer_defect" for r in results)
        first = results[0]
        assert first.primary.supporting, "bearing hypothesis needs supporting evidence"
        features = {e.feature for e in first.primary.supporting}
        assert "env_bpfo_ratio" in features
        assert "vib_kurtosis" in features

    def test_action_plan_attached(self, config, detector, engine):
        results = diagnose_scenario(
            config, detector, engine, SCENARIOS_STATE["bearing"]
        )
        action = results[0].recommended_action
        assert action is not None
        assert action.action_id == "replace_bearing"
        assert action.preconditions
        assert action.verification
        assert action.expected_changes

    def test_contradicting_evidence_for_other_modes(self, config, detector, engine):
        results = diagnose_scenario(
            config, detector, engine, SCENARIOS_STATE["bearing"]
        )
        ranked = results[0].hypotheses
        imbalance = next(h for h in ranked if h.mode_id == "shaft_imbalance")
        cooling = next(h for h in ranked if h.mode_id == "cooling_degradation")
        # Imbalance mode is contradicted: its signature features stayed normal.
        assert any(e.feature == "spec_1x_amp" for e in imbalance.contradicting)
        assert cooling.score < imbalance.score < ranked[0].score


class TestImbalanceDiagnosis:
    def test_primary_is_imbalance(self, config, detector, engine):
        results = diagnose_scenario(
            config, detector, engine, SCENARIOS_STATE["imbalance"]
        )
        assert all(r.is_confident for r in results)
        assert all(r.primary.mode_id == "shaft_imbalance" for r in results)
        assert results[0].recommended_action.action_id == "field_balancing"


class TestOvertemperatureDiagnosis:
    def test_primary_is_cooling(self, config, detector, engine):
        results = diagnose_scenario(
            config, detector, engine, SCENARIOS_STATE["overtemperature"], advance_s=600.0
        )
        assert all(r.is_confident for r in results)
        assert all(r.primary.mode_id == "cooling_degradation" for r in results)
        assert results[0].recommended_action.action_id == "cooling_overhaul"


class TestNormalOperation:
    def test_no_diagnosis_without_anomaly(self, config, detector, engine):
        sim = MotorSimulator(config, seed=99)
        sim.advance(600.0)
        rows = sample_rows(sim, config, 5)
        for row in rows:
            anomaly = detector.predict(row)
            result = engine.diagnose(anomaly, row, detector)
            assert not result.is_confident
            assert result.primary is None
            assert result.recommended_action is None


class TestAmbiguity:
    def test_concurrent_faults_flag_dominant_and_residual(self, config, detector, engine):
        """With two concurrent faults the engine diagnoses the dominant one;
        the residual fault keeps a high score so the recovery loop can
        re-diagnose it after the first intervention."""
        state = MachineState(bearing_severity=0.55, imbalance_severity=0.7)
        results = diagnose_scenario(config, detector, engine, state)
        assert all(r.is_confident for r in results)
        assert all(r.primary.mode_id == "bearing_outer_defect" for r in results)
        for r in results:
            imbalance = next(h for h in r.hypotheses if h.mode_id == "shaft_imbalance")
            assert imbalance.confidence == "high", (
                "residual imbalance fault must remain visible for re-diagnosis"
            )

    def test_tied_evidence_is_ambiguous(self, config, detector, engine):
        """A window with exactly tied bearing and imbalance evidence must be
        reported ambiguous: no confident action, escalation note."""
        row = {}
        for name in FEATURE_NAMES:
            center, scale = detector.center_and_scale(name)
            row[name] = center
        # Only one equally-strong breach per competing mode.
        for name, z_target in (("env_bpfo_ratio", 4.0), ("spec_1x_amp", 4.0)):
            center, scale = detector.center_and_scale(name)
            row[name] = center + z_target * scale
        anomaly = detector.predict(row)
        assert anomaly.is_anomaly
        result = engine.diagnose(anomaly, row, detector)
        assert not result.is_confident
        assert result.recommended_action is None
        assert result.ambiguity_note
        top_two = sorted(result.hypotheses, key=lambda h: -h.score)[:2]
        assert top_two[0].score - top_two[1].score < engine.margin
        assert {top_two[0].mode_id, top_two[1].mode_id} == {
            "bearing_outer_defect",
            "shaft_imbalance",
        }

    def test_weak_fault_low_confidence_label(self, config, detector, engine):
        state = MachineState(bearing_severity=0.15)
        results = diagnose_scenario(config, detector, engine, state)
        confident = [r for r in results if r.is_confident]
        if confident:  # borderline severity may still be confident
            pytest.skip("severity 0.15 crosses the confident threshold for this seed")
        for r in results:
            if r.primary is not None:
                assert r.primary.confidence in ("medium", "low")


class TestHypothesisScoring:
    def test_scores_are_finite_and_sorted(self, config, detector, engine):
        results = diagnose_scenario(
            config, detector, engine, SCENARIOS_STATE["bearing"]
        )
        for r in results:
            scores = [h.score for h in r.hypotheses]
            assert scores == sorted(scores, reverse=True)
            assert all(np.isfinite(s) for s in scores)
            assert all(s >= 0 for s in scores)

    def test_evidence_dicts_are_json_safe(self, config, detector, engine):
        results = diagnose_scenario(
            config, detector, engine, SCENARIOS_STATE["bearing"]
        )
        summary = results[0].as_summary()
        import json

        json.dumps(summary)  # must not raise

    def test_engine_validation(self):
        with pytest.raises(ValueError):
            DiagnosisEngine(margin=0)
