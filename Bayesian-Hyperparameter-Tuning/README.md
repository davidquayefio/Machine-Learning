# Bayesian Hyperparameter Tuning for Santander Customer Transaction Prediction

This repository compares a fixed LightGBM baseline, random search, and Gaussian-process Bayesian optimization for the Kaggle Santander Customer Transaction Prediction problem. The objective is to tune a binary-classification model using ROC AUC as the primary metric, while keeping the optimization budget realistic for experimentation.

## Research question
Can Bayesian optimization find strong LightGBM hyperparameter settings with a similar or better validation performance than random search under a constrained evaluation budget?

## Project goals
- reproduce a baseline LightGBM score
- compare random search and Bayesian optimization under the same data split and CV configuration
- save trial-level results for downstream analysis
- document the experimental workflow and interpret the final comparison carefully

## Dataset
This project expects the Kaggle training CSV to be available locally. The repository is designed to read from either:

- `data/processed/train.csv`
- or `data/raw/train.csv`

The script also supports a root-level `train.csv` if the file is placed there and the path is adjusted manually. Do not commit raw Kaggle data to version control.

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the experiment
From the repository root:

```bash
python run_experiment.py
```

This script runs:
1. baseline evaluation
2. random search with 20 trials
3. Bayesian optimization with 5 initialization points and 20 iterations
4. CSV export of the optimization results

## Model and evaluation setup
- Model: LightGBM classifier
- Target: binary classification (`target`)
- Metric: ROC AUC
- Cross-validation: stratified 3-fold CV
- Optimization budget: capped to 100,000 rows for practical iteration speed
- Evaluation strategy: optimize on development data only and keep final conclusions restricted to the tested settings

## Current preliminary result
A single development-run comparison produced the following approximate validation scores:

- Baseline: 0.8743 CV ROC AUC
- Random search best: 0.8838 CV ROC AUC
- Bayesian optimization best: 0.8839 CV ROC AUC

This indicates that Bayesian optimization was marginally better than random search in this run, but the gain was very small. The project should be treated as a preliminary result rather than evidence of general superiority.

## Repository structure
```text
Bayesian-Hyperparameter-Tuning/
├── README.md
├── requirements.txt
├── LICENSE
├── run_experiment.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 00_background.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_random_search.ipynb
│   └── 04_bayesian_optimization.ipynb
├── src/
│   ├── data.py
│   ├── objective.py
│   ├── model.py
│   ├── optimization.py
│   └── __init__.py
├── results/
├── docs/
├── reports/
├── figures/
├── experiments/
└── .gitignore
```

## Interpretation
The main takeaway is that both random search and Bayesian optimization can find strong LightGBM configurations for this dataset. Bayesian optimization was slightly stronger in this particular run, but the improvement over random search was minor. A stronger conclusion would require repeating the comparison across multiple seeds and reporting the distribution of best-so-far performance, not just one run.

## References
- Kaggle Santander Customer Transaction Prediction
- LightGBM
- scikit-learn
- Bayesian optimization literature
- project notes in `docs/` and `reports/`
