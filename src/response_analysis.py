"""Exercise-response and signal-to-noise analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


def paired_exercise_tests(
    baseline: pd.DataFrame,
    post: pd.DataFrame,
    correction_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Run paired baseline-vs-post tests for each metabolite.

    Parameters
    ----------
    baseline:
        Baseline feature matrix indexed by subject.
    post:
        Post-exercise feature matrix indexed by subject.
    correction_method:
        Multiple-testing correction method accepted by statsmodels.

    Returns
    -------
    pd.DataFrame
        Per-metabolite mean log2 fold-change, p-value and q-value.
    """

    common_subjects = baseline.index.intersection(post.index)
    common_features = baseline.columns.intersection(post.columns)

    if len(common_subjects) == 0 or len(common_features) == 0:
        raise ValueError("baseline and post must share subjects and feature columns.")

    b = baseline.loc[common_subjects, common_features]
    p = post.loc[common_subjects, common_features]

    rows = []
    for feature in common_features:
        statistic, p_value = stats.ttest_rel(p[feature], b[feature], nan_policy="omit")
        rows.append(
            {
                "Metabolite": feature,
                "mean_log2FC": float((p[feature] - b[feature]).mean()),
                "t_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

    results = pd.DataFrame(rows)
    results["q_value"] = multipletests(results["p_value"], method=correction_method)[1]
    return results


def compute_signal_to_noise(
    baseline_1: pd.DataFrame,
    baseline_2: pd.DataFrame,
    post: pd.DataFrame,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute metabolite-level exercise signal-to-noise ratios.

    Exercise reactivity is defined as the mean absolute difference between the
    post-exercise value and the average of the two baselines. Methodological
    noise is estimated as the mean absolute difference between the two baseline
    samples.

    Parameters
    ----------
    baseline_1, baseline_2, post:
        Matrices indexed by subject with matching metabolite columns.
    eps:
        Small value used to avoid division by zero.

    Returns
    -------
    pd.DataFrame
        Per-metabolite exercise reactivity, baseline noise and SNR.
    """

    common_subjects = baseline_1.index.intersection(baseline_2.index).intersection(post.index)
    common_features = baseline_1.columns.intersection(baseline_2.columns).intersection(post.columns)

    if len(common_subjects) == 0 or len(common_features) == 0:
        raise ValueError("baseline_1, baseline_2 and post must share subjects and feature columns.")

    b1 = baseline_1.loc[common_subjects, common_features]
    b2 = baseline_2.loc[common_subjects, common_features]
    p = post.loc[common_subjects, common_features]
    baseline_mean = (b1 + b2) / 2

    rows = []
    for feature in common_features:
        exercise_reactivity = float((p[feature] - baseline_mean[feature]).abs().mean())
        methodological_noise = float((b2[feature] - b1[feature]).abs().mean())
        rows.append(
            {
                "Metabolite": feature,
                "exercise_reactivity": exercise_reactivity,
                "methodological_noise": methodological_noise,
                "SNR": exercise_reactivity / max(methodological_noise, eps),
            }
        )

    return pd.DataFrame(rows)


def summarize_exercise_response(
    baseline_1: pd.DataFrame,
    baseline_2: pd.DataFrame,
    post: pd.DataFrame,
    class_map: pd.DataFrame | None = None,
    correction_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Combine conventional paired tests with signal-to-noise analysis."""

    conventional = paired_exercise_tests(baseline_1, post, correction_method=correction_method)
    snr = compute_signal_to_noise(baseline_1, baseline_2, post)

    out = conventional.merge(snr, on="Metabolite", how="inner")
    if class_map is not None:
        out = out.merge(class_map, on="Metabolite", how="left")

    out["significant_FDR_0.05"] = out["q_value"] < 0.05
    out["response_exceeds_noise"] = out["SNR"] >= 1.0
    out = out.sort_values("SNR", ascending=False).reset_index(drop=True)
    return out


def run_participant_determinant_analysis(
    response_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    predictors: list[str],
    correction_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Run univariable linear models for response determinants.

    Parameters
    ----------
    response_matrix:
        Subject-by-metabolite matrix containing response values.
    metadata:
        Participant-level table indexed by subject or containing ``Subject_ID``.
    predictors:
        Metadata columns to test as univariable predictors.
    correction_method:
        Multiple-testing correction method accepted by statsmodels.

    Returns
    -------
    pd.DataFrame
        Regression coefficients, p-values and q-values for each
        metabolite-predictor pair.
    """

    if "Subject_ID" in metadata.columns:
        metadata = metadata.set_index("Subject_ID")

    common_subjects = response_matrix.index.intersection(metadata.index)
    if len(common_subjects) == 0:
        raise ValueError("response_matrix and metadata must share subject identifiers.")

    response = response_matrix.loc[common_subjects]
    meta = metadata.loc[common_subjects]

    missing_predictors = [p for p in predictors if p not in meta.columns]
    if missing_predictors:
        raise ValueError(f"Predictors not found in metadata: {missing_predictors}")

    rows = []
    for metabolite in response.columns:
        y = response[metabolite]
        for predictor in predictors:
            X = sm.add_constant(meta[[predictor]], has_constant="add")
            model = sm.OLS(y, X, missing="drop").fit()
            rows.append(
                {
                    "Metabolite": metabolite,
                    "Predictor": predictor,
                    "beta": float(model.params[predictor]),
                    "p_value": float(model.pvalues[predictor]),
                    "n_observations": int(model.nobs),
                }
            )

    results = pd.DataFrame(rows)
    results["q_value"] = multipletests(results["p_value"], method=correction_method)[1]
    return results.sort_values("q_value").reset_index(drop=True)
