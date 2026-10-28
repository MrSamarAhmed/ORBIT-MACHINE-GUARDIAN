from unittest.mock import Mock

from orbit.api import service
from orbit.evaluation import cwru
from orbit.model.inference import FaultClassifier


def test_service_loads_the_absolute_checked_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "custom.pt"
    checkpoint.touch()
    monkeypatch.setattr(service, "MODEL_WEIGHTS_PATH", checkpoint)
    monkeypatch.chdir(tmp_path)
    loader = Mock(return_value=object())
    monkeypatch.setattr(FaultClassifier, "load", loader)
    assert service._load_classifier() is loader.return_value
    loader.assert_called_once_with(weights_path=checkpoint)


def test_transfer_loads_the_reported_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "custom.pt"
    checkpoint.touch()
    loader = Mock(return_value=object())
    monkeypatch.setattr(FaultClassifier, "load", loader)
    result = cwru.evaluate_classifier_transfer([], 100, 50, weights_path=checkpoint)
    assert result["available"] is False
    loader.assert_called_once_with("cpu", weights_path=checkpoint)
