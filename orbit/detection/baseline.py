"""Baseline anomaly detection.

Fits a robust statistical baseline on healthy feature vectors only (no
fault labels needed). Each feature gets a robust z-score

    z = (x - median) / (1.4826 * MAD)

which equals the classic z-score for Gaussian data but is insensitive to
outliers in the calibration set. A window is anomalous when any feature
exceeds ``z_threshold`` or the temperature breaches the absolute
engineering limit. All decisions are deterministic and every breach is
reported with its evidence (feature, value, z-score, direction) — this is
the evidence source downstream diagnosis consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from orbit.signalproc.features import FEATURE_NAMES

_MIN_CALIBRATION_ROWS = 10
_MAD_SCALE = 1.4826
_MAD_FLOOR = 1e-9


@dataclass(frozen=True)
class FeatureBreach:
    feature: str
    value: float
    z: float
    direction: str  # "high" or "low"

    def as_evidence(self) -> Dict[str, object]:
        return {
            "feature": self.feature,
            "value": round(self.value, 4),
            "z": round(self.z, 2),
            "direction": self.direction,
        }


@dataclass(frozen=True)
class AnomalyResult:
    is_anomaly: bool
    max_abs_z: float
    breaches: Tuple[FeatureBreach, ...]
    temp_limit_exceeded: bool

    @property
    def primary_breach(self) -> FeatureBreach:
        if not self.breaches:
            raise ValueError("no breaches on a non-anomalous result")
        return max(self.breaches, key=lambda b: abs(b.z))


class BaselineDetector:
    """Robust per-feature baseline with deterministic alarm rules.

    Features whose healthy spread is much smaller than their magnitude
    (e.g. RMS, which barely varies between healthy windows) get a
    relative scale floor so that numerically tiny fluctuations cannot
    produce enormous z-scores.
    """

    def __init__(
        self,
        z_threshold: float = 3.5,
        temp_limit_c: float = 75.0,
        rel_scale_floor: float = 0.01,
    ):
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if rel_scale_floor < 0:
            raise ValueError("rel_scale_floor must be >= 0")
        self.z_threshold = float(z_threshold)
        self.temp_limit_c = float(temp_limit_c)
        self.rel_scale_floor = float(rel_scale_floor)
        self._centers: Dict[str, float] = {}
        self._scales: Dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self._centers)

    def fit(self, feature_rows: Sequence[Mapping[str, float]]) -> "BaselineDetector":
        if len(feature_rows) < _MIN_CALIBRATION_ROWS:
            raise ValueError(
                f"at least {_MIN_CALIBRATION_ROWS} calibration rows required, "
                f"got {len(feature_rows)}"
            )
        keys = set(feature_rows[0])
        for row in feature_rows:
            if set(row) != keys:
                raise ValueError("all calibration rows must share the same feature keys")
        missing = set(FEATURE_NAMES) - keys
        if missing:
            raise ValueError(f"calibration rows missing features: {sorted(missing)}")

        centers: Dict[str, float] = {}
        scales: Dict[str, float] = {}
        for name in FEATURE_NAMES:
            values = np.array([row[name] for row in feature_rows], dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"feature {name} has non-finite values in calibration data")
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            scale = max(
                _MAD_SCALE * mad,
                float(values.std()),
                self.rel_scale_floor * abs(median),
                _MAD_FLOOR,
            )
            centers[name] = median
            scales[name] = scale
        self._centers = centers
        self._scales = scales
        return self

    def z_scores(self, features: Mapping[str, float]) -> Dict[str, float]:
        self._require_fitted()
        missing = set(FEATURE_NAMES) - set(features)
        if missing:
            raise ValueError(f"input missing features: {sorted(missing)}")
        result = {}
        for name in FEATURE_NAMES:
            value = float(features[name])
            if not np.isfinite(value):
                raise ValueError(f"feature {name} is not finite")
            result[name] = (value - self._centers[name]) / self._scales[name]
        return result

    def predict(self, features: Mapping[str, float]) -> AnomalyResult:
        """Deterministic anomaly decision with full breach evidence."""
        z = self.z_scores(features)
        breaches: List[FeatureBreach] = []
        temp_limit_exceeded = float(features["temp_c"]) > self.temp_limit_c
        for name, z_value in z.items():
            if abs(z_value) > self.z_threshold:
                breaches.append(
                    FeatureBreach(
                        feature=name,
                        value=float(features[name]),
                        z=z_value,
                        direction="high" if z_value > 0 else "low",
                    )
                )
        if temp_limit_exceeded and not any(b.feature == "temp_c" for b in breaches):
            breaches.append(
                FeatureBreach(
                    feature="temp_c",
                    value=float(features["temp_c"]),
                    z=z["temp_c"],
                    direction="high",
                )
            )
        max_abs_z = max((abs(v) for v in z.values()), default=0.0)
        is_anomaly = bool(breaches)
        return AnomalyResult(
            is_anomaly=is_anomaly,
            max_abs_z=float(max_abs_z),
            breaches=tuple(breaches),
            temp_limit_exceeded=temp_limit_exceeded,
        )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("detector is not fitted; call fit() with healthy data first")
