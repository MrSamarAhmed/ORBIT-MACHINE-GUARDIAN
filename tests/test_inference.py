"""Inference + controller corroboration tests.

Require trained weights at runtime/model/fault_cnn.pt; skipped otherwise
(run ``python -m orbit.model.train`` first)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from orbit.model.cnn import FaultCNN
from orbit.model.dataset import CLASSES, generate_split
from orbit.model.inference import WEIGHTS_PATH, FaultClassifier, available_devices, parity_check
from orbit.model.device import resolve_device

requires_weights = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(), reason="trained weights not present"
)


@pytest.fixture(scope="module")
def classifier() -> FaultClassifier:
    return FaultClassifier.load("cpu")


@requires_weights
class TestInference:
    def test_load_and_predict(self, classifier):
        split = generate_split(windows_per_class=3, seed=4242)
        probs = classifier.predict_proba(split.X)
        assert probs.shape == (9, len(CLASSES))
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_synthetic_classes_are_separable(self, classifier):
        split = generate_split(windows_per_class=10, seed=4242)
        preds = classifier.predict_proba(split.X).argmax(axis=1)
        assert (preds == split.y).all(), "trained model should separate clean synthetic classes"

    def test_classify_reports_device(self, classifier):
        split = generate_split(windows_per_class=1, seed=4242)
        out = classifier.classify(split.X[:1])
        assert out["predicted_class"] in CLASSES
        assert out["device"]["device_type"] == "cpu"

    def test_predict_requires_2d(self, classifier):
        with pytest.raises(ValueError, match="2-D"):
            classifier.predict_proba(np.zeros(100, dtype=np.float32))

    def test_missing_weights_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="trained weights not found"):
            FaultClassifier.load("cpu", weights_path=tmp_path / "nope.pt")


@requires_weights
class TestParity:
    def test_cpu_self_parity(self):
        split = generate_split(windows_per_class=2, seed=4242)
        result = parity_check(split.X, device_a="cpu", device_b="cpu")
        assert result["passed"]
        assert result["max_abs_difference"] == pytest.approx(0.0, abs=1e-6)

    def test_multi_device_parity_if_available(self):
        devices = available_devices()
        if len(devices) < 2:
            pytest.skip("only one device available; parity across devices not testable")
        split = generate_split(windows_per_class=4, seed=4242)
        result = parity_check(split.X, device_a=devices[0], device_b=devices[-1])
        # CPU vs accelerator: same weights, tolerant agreement expected.
        assert result["passed"], (
            f"{devices[0]} vs {devices[-1]} max|diff|={result['max_abs_difference']:.2e}"
        )


@requires_weights
class TestControllerCorroboration:
    def test_corroboration_recorded_and_agrees(self, config, detector):
        from orbit.diagnosis.engine import DiagnosisEngine
        from orbit.incident.controller import IncidentController
        from orbit.actions.environment import MaintenanceEnvironment
        from orbit.simulator.motor import MotorSimulator, MachineState

        clf = FaultClassifier.load("cpu")
        sim = MotorSimulator(config, seed=53)
        sim.advance(600.0)
        sim.set_state(MachineState(bearing_severity=0.6))
        controller = IncidentController(
            simulator=sim, detector=detector, engine=DiagnosisEngine(),
            environment=MaintenanceEnvironment(), classifier=clf, config=config,
        )
        result = controller.run_cycle()
        incident = result.incident
        assert incident.final_status == "RESOLVED"
        assert incident.corroboration is not None
        assert incident.corroboration["predicted_class"] == "bearing"
        assert incident.corroboration["mapped_hypothesis"] == "bearing_outer_defect"
        assert incident.corroboration["agrees_with_diagnosis"] is True
        assert incident.corroboration["device"]["device_type"] == "cpu"

    def test_record_json_serializable_with_corroboration(self, config, detector):
        from orbit.diagnosis.engine import DiagnosisEngine
        from orbit.incident.controller import IncidentController
        from orbit.actions.environment import MaintenanceEnvironment
        from orbit.simulator.motor import MotorSimulator, MachineState

        clf = FaultClassifier.load("cpu")
        sim = MotorSimulator(config, seed=53)
        sim.advance(600.0)
        sim.set_state(MachineState(bearing_severity=0.6))
        controller = IncidentController(
            simulator=sim, detector=detector, engine=DiagnosisEngine(),
            environment=MaintenanceEnvironment(), classifier=clf, config=config,
        )
        result = controller.run_cycle()
        json.dumps(result.incident.as_dict())

    def test_no_classifier_means_no_corroboration(self, config, detector):
        from orbit.diagnosis.engine import DiagnosisEngine
        from orbit.incident.controller import IncidentController
        from orbit.actions.environment import MaintenanceEnvironment
        from orbit.simulator.motor import MotorSimulator, MachineState

        sim = MotorSimulator(config, seed=53)
        sim.advance(600.0)
        sim.set_state(MachineState(bearing_severity=0.6))
        controller = IncidentController(
            simulator=sim, detector=detector, engine=DiagnosisEngine(),
            environment=MaintenanceEnvironment(), config=config,
        )
        result = controller.run_cycle()
        assert result.incident.corroboration is None
