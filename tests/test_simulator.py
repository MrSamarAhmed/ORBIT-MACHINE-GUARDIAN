"""Simulator tests: determinism, scenario signatures, validation."""
from __future__ import annotations

import numpy as np
import pytest

from orbit.config import MotorConfig
from orbit.simulator.motor import MachineState, MotorSimulator, SCENARIOS
from tests.conftest import CALIBRATION_WINDOWS, SCENARIOS_STATE


def sample_rows(sim, config, count):
    from orbit.signalproc.features import extract_features

    return [extract_features(w, config) for w in sim.sample_windows(count)]


class TestDeterminism:
    def test_same_seed_same_state_bit_identical(self, config):
        windows = []
        for _ in range(2):
            sim = MotorSimulator(config, seed=123)
            sim.set_state(SCENARIOS_STATE["bearing"])
            windows.append(sim.sample_window().vibration)
        assert np.array_equal(windows[0], windows[1])

    def test_different_seeds_differ(self, config):
        a = MotorSimulator(config, seed=1).sample_window().vibration
        b = MotorSimulator(config, seed=2).sample_window().vibration
        assert not np.array_equal(a, b)

    def test_window_index_changes_noise(self, config):
        sim = MotorSimulator(config, seed=5)
        w0 = sim.sample_window().vibration
        w1 = sim.sample_window().vibration
        assert not np.array_equal(w0, w1)


class TestScenarioSignatures:
    def test_bearing_raises_envelope_bpfo(self, config):
        sim_h = MotorSimulator(config, seed=7)
        sim_f = MotorSimulator(config, seed=11)
        sim_f.set_state(SCENARIOS_STATE["bearing"])
        healthy = sample_rows(sim_h, config, 10)
        faulty = sample_rows(sim_f, config, 10)
        assert np.mean([r["env_bpfo_ratio"] for r in faulty]) > 5 * np.mean(
            [r["env_bpfo_ratio"] for r in healthy]
        )

    def test_bearing_raises_kurtosis(self, config):
        sim_h = MotorSimulator(config, seed=7)
        sim_f = MotorSimulator(config, seed=11)
        sim_f.set_state(SCENARIOS_STATE["bearing"])
        healthy = sample_rows(sim_h, config, 10)
        faulty = sample_rows(sim_f, config, 10)
        assert np.mean([r["vib_kurtosis"] for r in faulty]) > np.mean(
            [r["vib_kurtosis"] for r in healthy]
        )

    def test_imbalance_raises_1x_not_kurtosis(self, config):
        sim_h = MotorSimulator(config, seed=7)
        sim_f = MotorSimulator(config, seed=13)
        sim_f.set_state(SCENARIOS_STATE["imbalance"])
        healthy = sample_rows(sim_h, config, 10)
        faulty = sample_rows(sim_f, config, 10)
        assert np.mean([r["spec_1x_amp"] for r in faulty]) > 3 * np.mean(
            [r["spec_1x_amp"] for r in healthy]
        )
        # Imbalance is sinusoidal: kurtosis stays near-Gaussian (<4.5)
        assert np.mean([r["vib_kurtosis"] for r in faulty]) < 4.5

    def test_overtemperature_equilibrium(self, config):
        sim = MotorSimulator(config, seed=3)
        sim.set_state(SCENARIOS_STATE["overtemperature"])
        sim.advance(10 * config.temp_tau_s)
        assert sim.temperature_c == pytest.approx(
            sim.equilibrium_temperature_c(), abs=0.1
        )
        assert sim.temperature_c > config.temp_limit_c

    def test_healthy_temperature_below_limit(self, config):
        sim = MotorSimulator(config, seed=3)
        sim.advance(10 * config.temp_tau_s)
        assert sim.temperature_c < config.temp_limit_c


class TestBearingFrequencies:
    def test_bpfo_bpfi_values(self):
        cfg = MotorConfig()
        # 6205 bearing at 25 Hz shaft: BPFO = 4.5*25*(1-0.00794/0.03904) ~ 89.62 Hz,
        # BPFI = 4.5*25*(1+0.00794/0.03904) ~ 135.38 Hz (matches CWRU geometry).
        assert cfg.bpfo_hz == pytest.approx(89.62, abs=0.1)
        assert cfg.bpfi_hz == pytest.approx(135.38, abs=0.1)


class TestValidation:
    def test_rejects_bad_severity(self):
        with pytest.raises(ValueError):
            MachineState(bearing_severity=1.5)

    def test_rejects_bad_cooling(self):
        with pytest.raises(ValueError):
            MachineState(cooling_efficiency=0.0)

    def test_rejects_bad_load(self):
        with pytest.raises(ValueError):
            MachineState(load=2.0)

    def test_rejects_unknown_scenario(self, config):
        sim = MotorSimulator(config)
        with pytest.raises(ValueError):
            sim.set_scenario("nonexistent")

    def test_rejects_negative_advance(self, config):
        sim = MotorSimulator(config)
        with pytest.raises(ValueError):
            sim.advance(-1.0)

    def test_rejects_zero_window_count(self, config):
        with pytest.raises(ValueError):
            MotorSimulator(config).sample_windows(0)

    def test_config_rejects_bad_geometry(self):
        with pytest.raises(ValueError):
            MotorConfig(roller_d=0.1, pitch_d=0.039)

    def test_config_rejects_bad_temp_limit(self):
        with pytest.raises(ValueError):
            MotorConfig(temp_limit_c=30.0)

    def test_scenario_objects_present(self):
        assert set(SCENARIOS) == {"normal", "bearing", "imbalance", "overtemperature"}
