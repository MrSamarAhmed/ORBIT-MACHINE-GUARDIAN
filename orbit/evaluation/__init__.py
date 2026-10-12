"""Evaluation harness for ORBIT (Phase 6: testing and evaluation).

All metrics are measured, never asserted. Results are explicitly tagged
as synthetic-simulator outputs and kept separate from any experimental
(CWRU) dataset results.
"""
from orbit.evaluation.metrics import (
    EVAL_OUT,
    evaluate_classifier,
    evaluate_detection,
    evaluate_recovery,
    run_evaluation,
)

__all__ = [
    "EVAL_OUT",
    "evaluate_classifier",
    "evaluate_detection",
    "evaluate_recovery",
    "run_evaluation",
]
