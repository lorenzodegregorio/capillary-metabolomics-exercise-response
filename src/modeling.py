"""Predictive-modelling utilities for the VO₂peak demo."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_elasticnet_pipeline(random_state: int = 42) -> Pipeline:
    """Build an imputation-scaling-ElasticNetCV pipeline.

    The preprocessing steps are kept inside the pipeline so that imputation and
    scaling are fitted only on the training data inside each cross-validation
    fold.
    """

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9, 1.0],
                    alphas=np.logspace(-3, 1, 40),
                    cv=3,
                    random_state=random_state,
                    max_iter=50000,
                ),
            ),
        ]
    )


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Return standard regression metrics."""

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def nested_elasticnet_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 4,
    random_state: int = 42,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Compute out-of-fold predictions using nested ElasticNet validation.

    Parameters
    ----------
    X:
        Feature matrix indexed by subject.
    y:
        Target variable indexed by subject.
    n_splits:
        Number of outer cross-validation folds.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    oof_pred:
        Out-of-fold predictions aligned with ``y``.
    fold_results:
        Fold-level metrics and selected hyperparameters.
    coefficients:
        Non-zero coefficients selected in each outer fold.
    """

    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_pred = pd.Series(index=y.index, dtype=float, name="prediction")
    fold_rows: list[dict] = []
    coefficient_rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        train_ids = y.index[train_idx]
        test_ids = y.index[test_idx]

        X_train = X.loc[train_ids]
        X_test = X.loc[test_ids]
        y_train = y.loc[train_ids]
        y_test = y.loc[test_ids]

        pipe = build_elasticnet_pipeline(random_state=random_state + fold)
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        oof_pred.loc[test_ids] = pred

        metrics = regression_metrics(y_test, pred)
        model = pipe.named_steps["model"]

        fold_rows.append(
            {
                "Fold": fold,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "alpha": float(model.alpha_),
                "l1_ratio": float(model.l1_ratio_),
                **metrics,
            }
        )

        for feature, coef in zip(X.columns, model.coef_):
            if abs(coef) > 1e-10:
                coefficient_rows.append(
                    {
                        "Fold": fold,
                        "Feature": feature,
                        "Coefficient": float(coef),
                    }
                )

    return oof_pred, pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows)


def compare_feature_sets(
    feature_sets: dict[str, pd.DataFrame],
    y: pd.Series,
    n_splits: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Compare multiple feature sets using the same nested ElasticNet workflow."""

    summary_rows = []
    predictions = {}
    fold_tables = {}
    coefficient_tables = {}

    for name, X in feature_sets.items():
        pred, fold_table, coef_table = nested_elasticnet_oof_predictions(
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
        )

        summary_rows.append({"Model": name, **regression_metrics(y.loc[pred.index], pred)})
        predictions[name] = pred
        fold_tables[name] = fold_table
        coefficient_tables[name] = coef_table

    summary = pd.DataFrame(summary_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return summary, predictions, fold_tables, coefficient_tables


def stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 4,
    n_repeats: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate feature-selection stability across repeated CV folds.

    A feature is counted as selected when its ElasticNet coefficient is non-zero
    in a given fold.
    """

    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    selected_features: list[str] = []
    total_folds = 0

    for fold, (train_idx, _) in enumerate(cv.split(X), start=1):
        train_ids = y.index[train_idx]
        pipe = build_elasticnet_pipeline(random_state=random_state + fold)
        pipe.fit(X.loc[train_ids], y.loc[train_ids])

        coefs = pipe.named_steps["model"].coef_
        selected_features.extend([feature for feature, coef in zip(X.columns, coefs) if abs(coef) > 1e-10])
        total_folds += 1

    counts = Counter(selected_features)
    stability = (
        pd.DataFrame(
            {
                "Feature": list(X.columns),
                "Selection_count": [counts.get(feature, 0) for feature in X.columns],
                "Total_folds": total_folds,
            }
        )
        .assign(Selection_frequency=lambda df: df["Selection_count"] / df["Total_folds"])
        .sort_values(["Selection_frequency", "Feature"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return stability


def leave_one_feature_out(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run a leave-one-feature-out sensitivity analysis."""

    rows = []

    baseline_pred, _, _ = nested_elasticnet_oof_predictions(
        X=X,
        y=y,
        n_splits=n_splits,
        random_state=random_state,
    )
    rows.append({"Removed_feature": "None", **regression_metrics(y.loc[baseline_pred.index], baseline_pred)})

    for feature in X.columns:
        reduced_X = X.drop(columns=[feature])
        pred, _, _ = nested_elasticnet_oof_predictions(
            X=reduced_X,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
        )
        rows.append({"Removed_feature": feature, **regression_metrics(y.loc[pred.index], pred)})

    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
