"""Device resolution, dataset, and model tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from orbit.model.cnn import FaultCNN
from orbit.model.dataset import (
    CLASSES,
    CLASS_TO_INDEX,
    generate_all_splits,
    generate_split,
)
from orbit.model.device import resolve_device


class TestDeviceResolution:
    def test_auto_resolves_to_something(self):
        info = resolve_device("auto")
        assert info.device_type in ("cpu", "cuda", "mps")

    def test_cpu_explicit(self):
        info = resolve_device("cpu")
        assert info.device_type == "cpu"
        assert info.torch_device == torch.device("cpu")

    def test_mps_present_on_dev_machine(self):
        info = resolve_device("auto")
        if info.device_type != "mps":
            pytest.skip("MPS not the auto device in this environment")
        assert info.platform == "mps"
        assert "Apple" in info.device_name

    def test_rocm_maps_to_cuda(self):
        try:
            info = resolve_device("rocm")
            assert info.device_type == "cuda"
        except RuntimeError:
            pass  # no CUDA/ROCm on this machine — fail-closed is correct

    def test_unavailable_cuda_fails_closed(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA present; fail-closed path not testable")
        with pytest.raises(RuntimeError, match="Refusing to silently fall back"):
            resolve_device("cuda")

    def test_unknown_device_raises(self):
        with pytest.raises(ValueError, match="unknown device"):
            resolve_device("tpu")

    def test_info_dict_fields(self):
        info = resolve_device("cpu")
        d = info.as_dict()
        assert {"device_type", "platform", "device_name", "torch_version"} <= set(d)


class TestDataset:
    def test_split_shapes_and_labels(self):
        split = generate_split(windows_per_class=4, seed=11)
        assert split.X.shape == (12, 2560)
        assert split.y.shape == (12,)
        assert set(split.y.tolist()) == set(range(len(CLASSES)))
        assert split.class_counts == {name: 4 for name in CLASSES}

    def test_splits_have_disjoint_seeds(self):
        train, val, test = generate_all_splits(
            train_windows_per_class=3, val_windows_per_class=3, test_windows_per_class=3
        )
        seeds = train.seeds_used + val.seeds_used + test.seeds_used
        assert len(seeds) == len(set(seeds)), "splits must not share simulator seeds"

    def test_splits_differ_by_seed(self):
        a = generate_split(windows_per_class=2, seed=100)
        b = generate_split(windows_per_class=2, seed=200)
        assert not np.array_equal(a.X, b.X)

    def test_deterministic_generation(self):
        a = generate_split(windows_per_class=2, seed=11)
        b = generate_split(windows_per_class=2, seed=11)
        assert np.array_equal(a.X, b.X)

    def test_bearing_windows_differ_from_normal(self):
        split = generate_split(windows_per_class=5, seed=11)
        normal = split.X[split.y == CLASS_TO_INDEX["normal"]]
        bearing = split.X[split.y == CLASS_TO_INDEX["bearing"]]
        assert np.abs(bearing).max() > np.abs(normal).max()

    def test_invalid_window_count(self):
        with pytest.raises(ValueError):
            generate_split(windows_per_class=0)


class TestFaultCNN:
    def test_forward_shapes(self):
        model = FaultCNN()
        x = torch.randn(4, 2560)
        logits = model(x)
        assert logits.shape == (4, len(CLASSES))

    def test_parameter_count_compact(self):
        model = FaultCNN()
        n = model.num_parameters()
        assert 10_000 < n < 500_000, f"model should stay compact, got {n} params"

    def test_batched_forward_consistent(self):
        """Logical equivalence of batch sizes: batching must not change
        per-sample outputs beyond float tolerance."""
        torch.manual_seed(0)
        model = FaultCNN().eval()
        x = torch.randn(8, 2560)
        with torch.no_grad():
            batched = torch.softmax(model(x), dim=1)
            single = torch.softmax(torch.cat([model(x[i:i + 1]) for i in range(8)]), dim=1)
        assert torch.allclose(batched, single, atol=1e-5)
