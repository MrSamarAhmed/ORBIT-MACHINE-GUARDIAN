"""Tests for the transparent business-value projection calculator."""
from __future__ import annotations

import pytest

from orbit.evaluation.business import (
    DISCLAIMER,
    MEASURED_NAIVE_FALSE_RESOLUTION_RATE,
    ValueInputs,
    estimate_value,
)


class TestValueInputs:
    def test_defaults_are_valid(self):
        ValueInputs()  # must not raise

    def test_negative_operational_input_rejected(self):
        with pytest.raises(ValueError):
            ValueInputs(incidents_per_year=-1)

    def test_fraction_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            ValueInputs(failed_recovery_fraction=1.5)


class TestEstimateValue:
    def test_zero_incidents_zero_savings(self):
        result = estimate_value(ValueInputs(incidents_per_year=0))
        assert result.total_estimate == 0
        assert result.low == result.high == 0

    def test_zero_fractions_zero_savings(self):
        result = estimate_value(ValueInputs(
            failed_recovery_fraction=0.0,
            repeat_downtime_share=0.0,
            diagnosis_time_reduction=0.0,
        ))
        assert result.total_estimate == 0
        # Baseline cost is still reported for context.
        assert result.baseline_downtime_cost > 0

    def test_known_arithmetic(self):
        p = ValueInputs(
            incidents_per_year=10,
            downtime_hours_per_incident=2,
            cost_per_downtime_hour=1000,
            technician_hours_per_incident=2,
            technician_cost_per_hour=100,
            failed_recovery_fraction=0.5,
            repeat_downtime_share=0.5,
            diagnosis_time_reduction=0.25,
            sensitivity=0.1,
        )
        result = estimate_value(p)
        assert result.baseline_downtime_cost == 10 * 2 * 1000
        # repeat: 10 × 0.5 × 0.5 × 2 × 1000 = 5000
        assert result.avoided_repeat_downtime_cost == 5000
        # diagnosis: 10 × 0.25 × 2 × 100 = 500
        assert result.diagnosis_time_savings == 500
        assert result.total_estimate == 5500

    def test_range_orders_low_base_high(self):
        result = estimate_value(ValueInputs())
        assert result.low <= result.total_estimate <= result.high
        assert result.low < result.high  # defaults must produce a real range

    def test_lines_document_formulas(self):
        result = estimate_value(ValueInputs())
        formulas = " ".join(line["formula"] for line in result.lines)
        assert "failed_recovery_fraction" in formulas
        assert "diagnosis_time_reduction" in formulas
        terms = " ".join(line["term"] for line in result.lines)
        assert "Baseline" in terms

    def test_measured_preset_flag_is_simulator_only(self):
        # The default failed-recovery fraction comes from the synthetic
        # comparison harness and must stay honest about its origin.
        assert 0 <= MEASURED_NAIVE_FALSE_RESOLUTION_RATE <= 1
        assert ValueInputs().failed_recovery_fraction == \
            MEASURED_NAIVE_FALSE_RESOLUTION_RATE

    def test_projection_disclaimer_required(self):
        from orbit.evaluation.business import estimate_value as _
        result = estimate_value(ValueInputs()).as_dict()
        assert result["disclaimer"] == DISCLAIMER
        assert "NOT measured savings" in result["disclaimer"]
