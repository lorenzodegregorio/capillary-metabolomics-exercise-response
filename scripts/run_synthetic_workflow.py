"""Run a lightweight synthetic workflow for the public portfolio repository.

This script is optional and is meant as a quick sanity check for the reusable
functions in the src/ package. It uses synthetic data only.

Run from the repository root:

    python scripts/run_synthetic_workflow.py
"""

from pathlib import Path
import sys

# Allow running the script directly from the repository root without installing
# the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import pandas as pd

from src.synthetic_data import simulate_metabolomics_study, simulate_vo2peak_prediction_dataset
from src.reliability import compute_baseline_reliability
from src.response_analysis import summarize_exercise_response, run_participant_determinant_analysis
from src.modeling import compare_feature_sets, stability_selection


def run_noise_adjusted_metabolomics_demo() -> None:
    """Run the baseline reliability and exercise-response workflow."""

    simulation = simulate_metabolomics_study(n_subjects=100, random_state=42)
    data = simulation.data

    baseline_1 = data[data["Timepoint"] == "T0.1"].set_index("Subject_ID")[simulation.metabolite_columns]
    baseline_2 = data[data["Timepoint"] == "T0.2"].set_index("Subject_ID")[simulation.metabolite_columns]
    post_exercise = data[data["Timepoint"] == "PostExercise"].set_index("Subject_ID")[simulation.metabolite_columns]

    overall_r, reliability = compute_baseline_reliability(
        baseline_1,
        baseline_2,
        simulation.class_map,
    )

    response_summary = summarize_exercise_response(
        baseline_1,
        baseline_2,
        post_exercise,
        simulation.class_map,
    )

    response_matrix = post_exercise - (baseline_1 + baseline_2) / 2
    metadata = simulation.metadata.set_index("Subject_ID")

    determinant_results = run_participant_determinant_analysis(
        response_matrix=response_matrix,
        metadata=metadata,
        predictors=[
            "VO2peak",
            "BMI",
            "Age",
            "Sex_num",
            "Fasted_num",
            "Not_exhausted_num",
        ],
    )

    print("\n=== Noise-adjusted metabolomics demo ===")
    print(f"Overall baseline correlation: {overall_r:.3f}")
    print("\nTop baseline-reliability results:")
    print(reliability.head(5).to_string(index=False))
    print("\nTop exercise-response results:")
    print(response_summary.head(5).to_string(index=False))
    print("\nTop participant-determinant results:")
    print(determinant_results.head(5).to_string(index=False))


def run_vo2peak_prediction_demo() -> None:
    """Run the synthetic VO2peak prediction workflow."""

    simulation = simulate_vo2peak_prediction_dataset(n_subjects=100, random_state=42)

    feature_sets = {
        "Clinical only": simulation.clinical,
        "Omics only": simulation.omics,
        "Clinical + omics": pd.concat([simulation.clinical, simulation.omics], axis=1),
    }

    metrics, _, _, _ = compare_feature_sets(
        feature_sets=feature_sets,
        y=simulation.target,
        n_splits=4,
        random_state=42,
    )

    stability = stability_selection(
        X=pd.concat([simulation.clinical, simulation.omics], axis=1),
        y=simulation.target,
        n_splits=4,
        n_repeats=2,
        random_state=42,
    )

    print("\n=== VO2peak prediction demo ===")
    print("\nModel comparison:")
    print(metrics.to_string(index=False))
    print("\nMost frequently selected features:")
    print(stability.head(10).to_string(index=False))


if __name__ == "__main__":
    run_noise_adjusted_metabolomics_demo()
    run_vo2peak_prediction_demo()
