# Noise-aware Capillary Metabolomics for Exercise Response Analysis

This repository presents a public portfolio version of my Master's thesis project on capillary blood metabolomics, cardiorespiratory fitness and acute exercise response.

The original research project was carried out in collaboration with TU Dortmund and Biolyz. It focused on the analysis of repeated-baseline capillary blood metabolomics data collected around maximal cardiopulmonary exercise testing.

The public version of this repository does **not** include raw biomedical data or participant-level datasets. Instead, it provides a clean and reproducible demonstration of the main methodological ideas using synthetic data.

---

## Project overview

Acute exercise metabolomics is often based on a single pre-exercise baseline measurement. However, this makes it difficult to determine whether post-exercise metabolite changes truly exceed short-term resting variability.

This project addresses that limitation by using two pre-exercise baseline samples to estimate baseline-derived methodological noise. Post-exercise metabolite changes are then interpreted not only in terms of statistical significance, but also relative to short-term baseline variability.

The main objective is to distinguish broad group-level metabolic perturbations from responses that more clearly exceed baseline-derived noise.

---

## Main analytical components

The original analysis workflow included:

- metabolomics data preprocessing and quality control;
- metabolite filtering and transformation;
- repeated-baseline reliability analysis;
- exploratory PCA and multivariate analysis;
- single-baseline post-exercise response analysis;
- signal-to-noise analysis of exercise-induced metabolite responses;
- participant-level determinant analysis;
- VO₂peak prediction using regularized regression and nested cross-validation.

---

## Visual overview

### Study design

![Study design](figures/study_design.png)

### Baseline reliability

![Baseline agreement](figures/baseline_agreement.png)

### Exercise-induced metabolomic shift

![PCA by timepoint](figures/pca_timepoint.png)

### Noise-aware response analysis

![Signal-to-noise analysis](figures/signal_to_noise.png)

> Note: Figures are included for portfolio and methodological presentation purposes. Raw participant-level data are not publicly available due to privacy and collaboration constraints.

---

## Repository structure

```text
capillary-metabolomics-exercise-response/
├── data/
│   └── README.md
├── figures/
├── notebooks/
│   ├── 01_noise_adjusted_metabolomics_demo.ipynb
│   └── 02_vo2peak_prediction_demo.ipynb
├── reports/
│   └── project_summary.md
├── scripts/
│   └── run_synthetic_workflow.py
├── src/
│   ├── __init__.py
│   ├── synthetic_data.py
│   ├── reliability.py
│   ├── response_analysis.py
│   └── modeling.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Demo notebooks

The repository includes two public notebooks based on synthetic data.

The notebooks are designed to demonstrate the methodological structure of the original analysis without exposing private biomedical data.

### 1. Noise-aware metabolomics workflow

[`notebooks/01_noise_adjusted_metabolomics_demo.ipynb`](notebooks/01_noise_adjusted_metabolomics_demo.ipynb)

This notebook demonstrates the core workflow used to analyse repeated-baseline metabolomics data:

- synthetic metabolomics dataset generation;
- baseline 1 vs baseline 2 reliability analysis;
- PCA-based exploration of timepoint structure;
- post-exercise response analysis;
- signal-to-noise ranking of metabolite responses;
- participant-level determinant analysis.

### 2. VO₂peak prediction workflow

[`notebooks/02_vo2peak_prediction_demo.ipynb`](notebooks/02_vo2peak_prediction_demo.ipynb)

This notebook demonstrates the predictive modelling part of the project:

- synthetic clinical and metabolomics feature generation;
- comparison of clinical-only, omics-only and combined feature sets;
- ElasticNet regression;
- nested cross-validation;
- out-of-fold performance evaluation;
- feature-selection stability analysis.

---

## Reusable Python modules

The `src/` folder contains reusable Python modules extracted from the methodological structure of the project.

- `synthetic_data.py`  
  Synthetic data generation for metabolomics and VO₂peak prediction demos.

- `reliability.py`  
  Functions for baseline agreement and repeated-baseline reliability analysis.

- `response_analysis.py`  
  Functions for post-exercise response analysis, signal-to-noise ranking and determinant analysis.

- `modeling.py`  
  Functions for ElasticNet modelling, nested cross-validation, feature-set comparison and feature stability.

---

## Running the project

Clone the repository:

```bash
git clone https://github.com/lorenzodegregorio/capillary-metabolomics-exercise-response.git
cd capillary-metabolomics-exercise-response
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the synthetic workflow script:

```bash
python scripts/run_synthetic_workflow.py
```

Or open the notebooks:

```bash
jupyter notebook
```

---

## Data availability

The original dataset cannot be shared publicly due to privacy and collaboration constraints.

The original study involved biomedical and metabolomics data collected from human participants. For this reason, raw data, participant-level data and internal collaboration files are not included in this repository.

The public notebooks use synthetic data to demonstrate the main methodological steps of the analysis workflow.

---

## Technical keywords

`Python` · `pandas` · `NumPy` · `scikit-learn` · `statsmodels` · `metabolomics` · `biomedical data science` · `exercise physiology` · `VO₂peak` · `signal-to-noise analysis` · `nested cross-validation` · `ElasticNet` · `PCA`

---

## Project status

This repository is a cleaned and public-facing portfolio version of a Master's thesis project.

It is intended to demonstrate the methodological structure, coding style and analytical reasoning behind the original work, without exposing private biomedical data or unpublished research material.
