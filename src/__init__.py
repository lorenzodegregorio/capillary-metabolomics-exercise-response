"""
Reusable utilities for the capillary metabolomics exercise-response portfolio project.

The public repository uses synthetic data to demonstrate the methodological
structure of the original thesis analysis without sharing participant-level data.
"""

from .synthetic_data import simulate_metabolomics_study, simulate_vo2peak_prediction_dataset
from .reliability import compute_baseline_reliability
from .response_analysis import (
    paired_exercise_tests,
    compute_signal_to_noise,
    summarize_exercise_response,
    run_participant_determinant_analysis,
)
from .modeling import (
    build_elasticnet_pipeline,
    regression_metrics,
    nested_elasticnet_oof_predictions,
    compare_feature_sets,
    stability_selection,
    leave_one_feature_out,
)

__all__ = [
    "simulate_metabolomics_study",
    "simulate_vo2peak_prediction_dataset",
    "compute_baseline_reliability",
    "paired_exercise_tests",
    "compute_signal_to_noise",
    "summarize_exercise_response",
    "run_participant_determinant_analysis",
    "build_elasticnet_pipeline",
    "regression_metrics",
    "nested_elasticnet_oof_predictions",
    "compare_feature_sets",
    "stability_selection",
    "leave_one_feature_out",
]
