"""Baseline reliability utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_baseline_reliability(
    baseline_1: pd.DataFrame,
    baseline_2: pd.DataFrame,
    class_map: pd.DataFrame | None = None,
) -> tuple[float, pd.DataFrame]:
    """Compute overall and metabolite-level baseline reliability.

    Parameters
    ----------
    baseline_1:
        Matrix of first baseline measurements, indexed by subject.
    baseline_2:
        Matrix of second baseline measurements, indexed by subject.
    class_map:
        Optional DataFrame with columns ``Metabolite`` and ``Class``.

    Returns
    -------
    overall_r:
        Pearson correlation across all subject-metabolite pairs.
    reliability:
        DataFrame with metabolite-level correlations and mean absolute
        inter-baseline differences.
    """

    common_subjects = baseline_1.index.intersection(baseline_2.index)
    common_features = baseline_1.columns.intersection(baseline_2.columns)

    if len(common_subjects) == 0 or len(common_features) == 0:
        raise ValueError("baseline_1 and baseline_2 must share subjects and feature columns.")

    b1 = baseline_1.loc[common_subjects, common_features]
    b2 = baseline_2.loc[common_subjects, common_features]

    overall_r = float(np.corrcoef(b1.to_numpy().ravel(), b2.to_numpy().ravel())[0, 1])

    rows = []
    for feature in common_features:
        r, p_value = stats.pearsonr(b1[feature], b2[feature])
        rows.append(
            {
                "Metabolite": feature,
                "Baseline_r": float(r),
                "Baseline_p_value": float(p_value),
                "Mean_abs_baseline_difference": float((b2[feature] - b1[feature]).abs().mean()),
            }
        )

    reliability = pd.DataFrame(rows)

    if class_map is not None:
        reliability = reliability.merge(class_map, on="Metabolite", how="left")

    return overall_r, reliability
