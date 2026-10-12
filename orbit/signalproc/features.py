"""Condition-monitoring feature extraction.

Standard vibration-analysis features used in rotating-machinery
diagnostics:

* Time domain: RMS, peak, crest factor, kurtosis (impulsive faults raise
  kurtosis; pure sinusoidal imbalance does not).
* Frequency domain: amplitude at 1x and 2x shaft frequency (imbalance
  raises 1x), and band amplitude around BPFO.
* Envelope domain: the signal is band-passed around the structural
  resonance, analytic-envelope demodulated (Hilbert), and the amplitude of
  the envelope spectrum at BPFO is measured relative to its median — the
  classic outer-race bearing signature.
* Thermal: reported temperature (absolute limit handled by the detector).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis as scipy_kurtosis

from orbit.config import MotorConfig
from orbit.simulator.motor import SensorWindow

FEATURE_NAMES = (
    "vib_rms",
    "vib_peak",
    "vib_crest",
    "vib_kurtosis",
    "spec_1x_amp",
    "spec_2x_amp",
    "spec_bpfo_band_amp",
    "env_bpfo_ratio",
    "hf_energy_frac",
    "temp_c",
)

_ENVELOPE_BAND_HZ = (2500.0, 5000.0)
_ENVELOPE_LOWHZ = 10.0
_ENVELOPE_HIGHHZ = 400.0


def _bandpass(x: np.ndarray, fs: float, band: tuple) -> np.ndarray:
    sos = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def _spectrum_amplitudes(x: np.ndarray, fs: float) -> tuple:
    spec = np.abs(np.fft.rfft(x)) * 2.0 / len(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return spec, freqs


def _amp_at(spec: np.ndarray, freqs: np.ndarray, target_hz: float, half_width_hz: float) -> float:
    mask = (freqs >= target_hz - half_width_hz) & (freqs <= target_hz + half_width_hz)
    if not mask.any():
        raise ValueError(f"target frequency {target_hz} Hz outside spectrum range")
    return float(spec[mask].max())


def extract_features(window: SensorWindow, config: MotorConfig) -> Dict[str, float]:
    """Extract the full feature vector from one sensor window.

    Pure function of (window, config): no randomness, no hidden state.
    """
    if len(window.vibration) != config.window_samples:
        raise ValueError(
            f"vibration length {len(window.vibration)} does not match "
            f"config.window_samples {config.window_samples}"
        )
    x = np.asarray(window.vibration, dtype=float)
    fs = float(config.fs)

    # --- Time domain ---
    vib_rms = float(np.sqrt(np.mean(x ** 2)))
    vib_peak = float(np.max(np.abs(x)))
    vib_crest = float(vib_peak / vib_rms) if vib_rms > 0 else float("inf")
    vib_kurtosis = float(scipy_kurtosis(x, fisher=False, bias=False))

    # --- Frequency domain ---
    spec, freqs = _spectrum_amplitudes(x, fs)
    resolution = fs / len(x)
    spec_1x = _amp_at(spec, freqs, config.shaft_hz, resolution)
    spec_2x = _amp_at(spec, freqs, 2 * config.shaft_hz, resolution)
    spec_bpfo = _amp_at(spec, freqs, config.bpfo_hz, 2 * resolution)

    hf_mask = (freqs >= 2000.0) & (freqs <= 5000.0)
    total_power = float(np.sum(spec ** 2))
    hf_energy_frac = float(np.sum(spec[hf_mask] ** 2) / total_power) if total_power > 0 else 0.0

    # --- Envelope domain (bearing signature) ---
    bandpassed = _bandpass(x, fs, _ENVELOPE_BAND_HZ)
    envelope = np.abs(hilbert(bandpassed))
    envelope = envelope - envelope.mean()
    env_spec, env_freqs = _spectrum_amplitudes(envelope, fs)
    env_bpfo_amp = _amp_at(env_spec, env_freqs, config.bpfo_hz, resolution)
    norm_mask = (env_freqs >= _ENVELOPE_LOWHZ) & (env_freqs <= _ENVELOPE_HIGHHZ)
    normalizer = float(np.median(env_spec[norm_mask]))
    env_bpfo_ratio = float(env_bpfo_amp / normalizer) if normalizer > 0 else float("inf")

    features = {
        "vib_rms": vib_rms,
        "vib_peak": vib_peak,
        "vib_crest": vib_crest,
        "vib_kurtosis": vib_kurtosis,
        "spec_1x_amp": spec_1x,
        "spec_2x_amp": spec_2x,
        "spec_bpfo_band_amp": spec_bpfo,
        "env_bpfo_ratio": env_bpfo_ratio,
        "hf_energy_frac": hf_energy_frac,
        "temp_c": float(window.temperature_c),
    }
    if set(features) != set(FEATURE_NAMES):
        raise AssertionError("feature set drifted from FEATURE_NAMES")
    for name, value in features.items():
        if not np.isfinite(value):
            raise ValueError(f"feature {name} is not finite")
    return features
