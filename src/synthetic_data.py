"""Synthetic data generators used in the public portfolio notebooks.

The original thesis dataset contains biomedical and metabolomics data from human
participants and cannot be shared publicly. These functions generate synthetic
datasets with a similar *structure* so that the analysis workflow can be run
without exposing participant-level data.

The simulated values are not intended to reproduce the original study results.
They are only meant to make the code examples executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetabolomicsSimulation:
    """Container returned by :func:`simulate_metabolomics_study`.

    Attributes
    ----------
    data:
        Long-format metabolomics table with one row per subject-timepoint.
    metadata:
        Participant-level table.
    class_map:
        Mapping between metabolite names and broad metabolite classes.
    metabolite_columns:
        Ordered list of metabolite feature columns.
    """

    data: pd.DataFrame
    metadata: pd.DataFrame
    class_map: pd.DataFrame
    metabolite_columns: list[str]


def simulate_metabolomics_study(
    n_subjects: int = 100,
    n_amino: int = 20,
    n_amino_related: int = 23,
    n_fatty: int = 8,
    random_state: int = 42,
) -> MetabolomicsSimulation:
    """Simulate a repeated-baseline exercise metabolomics dataset.

    The generated dataset mimics the structure used in the public demo:
    two resting baseline samples and one immediate post-exercise sample
    for each subject.

    Parameters
    ----------
    n_subjects:
        Number of simulated participants.
    n_amino:
        Number of synthetic amino-acid features.
    n_amino_related:
        Number of synthetic amino-acid-related features.
    n_fatty:
        Number of synthetic fatty-acid features.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    MetabolomicsSimulation
        Synthetic metabolomics data, metadata and feature annotations.
    """

    rng = np.random.default_rng(random_state)

    subject_ids = [f"S{i:03d}" for i in range(1, n_subjects + 1)]
    age = np.clip(rng.normal(27, 8, n_subjects), 18, 63)
    sex_num = rng.binomial(1, 0.68, n_subjects)  # 1 = male, 0 = female
    bmi = np.clip(rng.normal(24.3, 3.5, n_subjects), 18, 36)
    fasted = rng.binomial(1, 0.21, n_subjects)
    not_exhausted = rng.binomial(1, 0.23, n_subjects)

    vo2peak = (
        48
        + 5.5 * sex_num
        - 0.35 * age
        - 0.85 * (bmi - 23)
        - 3.0 * not_exhausted
        + rng.normal(0, 4.5, n_subjects)
    )

    metadata = pd.DataFrame(
        {
            "Subject_ID": subject_ids,
            "Age": age,
            "Sex_num": sex_num,
            "BMI": bmi,
            "Fasted_num": fasted,
            "Not_exhausted_num": not_exhausted,
            "VO2peak": vo2peak,
        }
    )

    metabolite_names = (
        [f"AA_{i:02d}" for i in range(1, n_amino + 1)]
        + [f"AA_related_{i:02d}" for i in range(1, n_amino_related + 1)]
        + [f"FA_{i:02d}" for i in range(1, n_fatty + 1)]
        + ["Lactate"]
    )

    metabolite_class = (
        ["Amino acid"] * n_amino
        + ["Amino-acid related"] * n_amino_related
        + ["Fatty acid"] * n_fatty
        + ["Lactate"]
    )

    class_map = pd.DataFrame({"Metabolite": metabolite_names, "Class": metabolite_class})

    # Subject-specific metabolic fingerprint.
    subject_fingerprint = rng.normal(0, 0.55, size=(n_subjects, len(metabolite_names)))

    # Metabolite-level baseline means and noise.
    metabolite_baseline_mean = rng.normal(8.0, 0.6, len(metabolite_names))
    baseline_noise = rng.uniform(0.03, 0.18, len(metabolite_names))

    # Exercise effect by class.
    exercise_effect = np.zeros(len(metabolite_names))
    for idx, cls in enumerate(metabolite_class):
        if cls == "Amino acid":
            exercise_effect[idx] = rng.normal(-0.35, 0.12)
        elif cls == "Amino-acid related":
            exercise_effect[idx] = rng.normal(-0.25, 0.16)
        elif cls == "Fatty acid":
            exercise_effect[idx] = rng.normal(0.10, 0.12)
        else:  # Lactate
            exercise_effect[idx] = 1.6

    # Attenuate response in participants simulated as not fully exhausted.
    exhaustion_modifier = 1 - 0.35 * not_exhausted[:, None]

    rows = []
    for i, subject_id in enumerate(subject_ids):
        base_profile = metabolite_baseline_mean + subject_fingerprint[i]

        b1 = base_profile + rng.normal(0, baseline_noise)
        b2 = base_profile + rng.normal(0, baseline_noise)

        individual_response_noise = rng.normal(0, 0.10, len(metabolite_names))
        post = (
            base_profile
            + exercise_effect * exhaustion_modifier[i]
            + individual_response_noise
            + rng.normal(0, baseline_noise * 1.2)
        )

        for timepoint, values in zip(["T0.1", "T0.2", "PostExercise"], [b1, b2, post]):
            row = {
                "Subject_ID": subject_id,
                "Timepoint": timepoint,
                "Age": age[i],
                "Sex_num": sex_num[i],
                "BMI": bmi[i],
                "Fasted_num": fasted[i],
                "Not_exhausted_num": not_exhausted[i],
                "VO2peak": vo2peak[i],
            }
            row.update(dict(zip(metabolite_names, values)))
            rows.append(row)

    data = pd.DataFrame(rows)

    return MetabolomicsSimulation(
        data=data,
        metadata=metadata,
        class_map=class_map,
        metabolite_columns=metabolite_names,
    )


@dataclass(frozen=True)
class VO2PredictionSimulation:
    """Container returned by :func:`simulate_vo2peak_prediction_dataset`."""

    clinical: pd.DataFrame
    omics: pd.DataFrame
    target: pd.Series


def simulate_vo2peak_prediction_dataset(
    n_subjects: int = 100,
    n_metabolites: int = 52,
    random_state: int = 42,
) -> VO2PredictionSimulation:
    """Simulate a baseline feature table for VO₂peak prediction.

    The dataset includes clinical variables, baseline metabolomic features and a
    synthetic VO₂peak target. Clinical features carry stronger signal than omics
    features, mimicking the structure of many small-sample biomedical prediction
    tasks.

    Parameters
    ----------
    n_subjects:
        Number of simulated participants.
    n_metabolites:
        Number of synthetic baseline metabolomic features.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    VO2PredictionSimulation
        Clinical features, omics features and target series.
    """

    rng = np.random.default_rng(random_state)
    subject_ids = [f"S{i:03d}" for i in range(1, n_subjects + 1)]

    age = np.clip(rng.normal(27, 8, n_subjects), 18, 63)
    sex_num = rng.binomial(1, 0.68, n_subjects)
    bmi = np.clip(rng.normal(24.3, 3.5, n_subjects), 18, 36)
    relative_fat_mass = np.clip(13 + 0.9 * bmi - 5 * sex_num + rng.normal(0, 4, n_subjects), 5, 50)
    relative_fat_free_mass = 100 - relative_fat_mass
    endurance_weekly = rng.poisson(2.0 + 1.5 * sex_num, n_subjects)
    not_exhausted = rng.binomial(1, 0.23, n_subjects)

    clinical = pd.DataFrame(
        {
            "Subject_ID": subject_ids,
            "Age": age,
            "Sex_num": sex_num,
            "BMI": bmi,
            "Relative_fat_mass": relative_fat_mass,
            "Relative_fat_free_mass": relative_fat_free_mass,
            "Endurance_weekly": endurance_weekly,
            "Not_exhausted_num": not_exhausted,
        }
    ).set_index("Subject_ID")

    latent_fitness = (
        0.35 * (relative_fat_free_mass - relative_fat_free_mass.mean()) / relative_fat_free_mass.std()
        + 0.25 * (endurance_weekly - endurance_weekly.mean()) / max(endurance_weekly.std(), 1e-8)
        - 0.20 * (age - age.mean()) / age.std()
        + rng.normal(0, 0.7, n_subjects)
    )

    metabolite_names = [f"Met_{i:02d}" for i in range(1, n_metabolites + 1)]
    omics = pd.DataFrame(index=subject_ids)

    for j, met in enumerate(metabolite_names):
        loading = rng.normal(0.20, 0.12) if j < 12 else rng.normal(0.0, 0.06)
        omics[met] = 8 + loading * latent_fitness + rng.normal(0, 0.8, n_subjects)

    y = (
        52
        - 0.35 * age
        + 4.0 * sex_num
        - 0.65 * (bmi - 23)
        + 0.20 * relative_fat_free_mass
        + 0.65 * endurance_weekly
        - 2.8 * not_exhausted
        + 1.2 * latent_fitness
        + rng.normal(0, 4.5, n_subjects)
    )

    target = pd.Series(y, index=subject_ids, name="VO2peak")

    return VO2PredictionSimulation(clinical=clinical, omics=omics, target=target)
