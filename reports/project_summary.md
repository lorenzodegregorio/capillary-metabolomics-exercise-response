# Project Summary

## Context

This repository is a public portfolio version of a Master's thesis project on capillary blood metabolomics, cardiorespiratory fitness and acute exercise response.

The original research project investigated whether repeated-baseline capillary blood metabolomics can improve the interpretation of acute metabolic responses to maximal exercise.

## Core idea

Acute exercise metabolomics is often based on a single pre-exercise baseline sample. This makes it difficult to determine whether post-exercise changes are clearly larger than short-term resting variability.

The central idea of this project is to use two pre-exercise baseline samples to estimate baseline-derived methodological noise. Post-exercise metabolite changes can then be interpreted not only in terms of statistical significance, but also relative to the variability observed before exercise.

## Public repository scope

The original biomedical and metabolomics data cannot be shared publicly.

This repository therefore includes:

- a clean code structure;
- reusable Python utilities;
- synthetic-data notebooks;
- selected non-sensitive figures;
- a short methodological summary.

The synthetic examples are not intended to reproduce the original study results. They are included to make the workflow executable and understandable without exposing participant-level data.

## Main workflow

The public workflow is organized around two demo notebooks:

1. `01_noise_adjusted_metabolomics_demo.ipynb`
   - repeated-baseline simulation;
   - baseline reliability;
   - PCA-style exploratory structure;
   - paired post-exercise response testing;
   - signal-to-noise response ranking;
   - participant-level determinant analysis.

2. `02_vo2peak_prediction_demo.ipynb`
   - synthetic clinical and metabolomics features;
   - ElasticNet regression;
   - nested cross-validation;
   - model comparison across feature sets;
   - feature-selection stability.

## Reusable modules

The `src/` folder contains reusable functions for:

- synthetic data generation;
- baseline reliability analysis;
- exercise-response analysis;
- signal-to-noise calculation;
- participant determinant analysis;
- ElasticNet modelling and cross-validation.

## Limitations

The public repository is a portfolio and reproducibility-oriented adaptation of the original thesis work. It does not contain raw data, participant-level records, unpublished manuscript files, or full internal analysis notebooks.

The aim is to demonstrate the methodology, code organization and analytical reasoning behind the project while respecting privacy and collaboration constraints.
