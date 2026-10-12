"""Shared motor and simulation configuration.

Physical parameters mirror a small induction motor with a 6205-type ball
bearing (the bearing geometry used by the Case Western Reserve University
Bearing Data Center drive-end test rig) so that synthetic features are on
the same scale as real experimental data. All values are documented
assumptions of the simulator, not measured facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MotorConfig:
    shaft_hz: float = 25.0          # nominal shaft frequency (1500 RPM)
    fs: int = 12800                 # vibration sampling rate [Hz]
    window_s: float = 0.2           # analysis window length [s]

    # Bearing geometry (6205 single-row ball bearing)
    n_rollers: int = 9
    pitch_d: float = 0.03904        # [m]
    roller_d: float = 0.00794       # [m]
    contact_angle_deg: float = 0.0

    # Structural resonance excited by bearing impacts
    resonance_hz: float = 3400.0
    resonance_decay_s: float = 0.004  # exponential decay time constant

    # Healthy vibration amplitudes [m/s^2]
    healthy_1x_amp: float = 0.15
    healthy_noise_std: float = 0.05

    # Fault amplitude scaling at severity 1.0 [m/s^2]
    bearing_impulse_amp: float = 3.0
    imbalance_1x_amp: float = 2.5

    # Thermal model
    temp_ambient_c: float = 25.0
    temp_healthy_rise_c: float = 30.0   # rise at load 1.0, healthy cooling
    temp_limit_c: float = 75.0          # engineering alarm limit
    temp_tau_s: float = 30.0            # thermal time constant

    # Bearing defect impact-time jitter (fraction of defect period)
    defect_jitter: float = 0.02

    seed: int = 42

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        if not 0 < self.window_s <= 10:
            raise ValueError("window_s must be in (0, 10] seconds")
        if self.n_rollers <= 0:
            raise ValueError("n_rollers must be positive")
        if not 0 < self.roller_d < self.pitch_d:
            raise ValueError("roller_d must be in (0, pitch_d)")
        if self.temp_limit_c <= self.temp_ambient_c + self.temp_healthy_rise_c:
            raise ValueError("temp_limit_c must exceed healthy operating temperature")
        if not 0 <= self.defect_jitter < 0.5:
            raise ValueError("defect_jitter must be in [0, 0.5)")

    @property
    def window_samples(self) -> int:
        n = int(round(self.window_s * self.fs))
        if n <= 0:
            raise ValueError("window_s * fs must yield at least one sample")
        return n

    # --- Bearing characteristic frequencies ---
    @property
    def bpfo_hz(self) -> float:
        """Ball-pass frequency, outer race."""
        return self.n_rollers / 2 * self.shaft_hz * (1 - self.roller_d / self.pitch_d)

    @property
    def bpfi_hz(self) -> float:
        """Ball-pass frequency, inner race."""
        return self.n_rollers / 2 * self.shaft_hz * (1 + self.roller_d / self.pitch_d)

    @property
    def bsf_hz(self) -> float:
        """Ball spin frequency."""
        ratio = self.roller_d / self.pitch_d
        return self.pitch_d / (2 * self.roller_d) * self.shaft_hz * (1 - ratio ** 2)
