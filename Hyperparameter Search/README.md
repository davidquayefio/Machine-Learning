# Hyperparameter Search

A machine-learning project that compares hyperparameter-search approaches for a Random Forest classifier predicting `price_range` from mobile-phone feature data.

## Contents

- `data/train.csv`: training data used by the search scripts.
- `data/test.csv`: test data for later evaluation or prediction.
- `src/grid_search.py`: exhaustive scikit-learn `GridSearchCV` search.
- `src/ran_search.py`: scikit-learn `RandomizedSearchCV` search.
- `src/Bay_Op_Gaussain.py`: Gaussian-process Bayesian optimization with scikit-optimize.
- `src/hyperpo.py`: Tree-structured Parzen estimator search with Hyperopt.
- `src/optuna.py`: Optuna hyperparameter search.
- `src/figures/`: generated optimization graphs.

## Setup

Python 3.10 or newer is recommended. From this directory, create and install the virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run searches

Run these commands from this project directory:

```bash
python src/grid_search.py
python src/ran_search.py
python src/Bay_Op_Gaussain.py
python src/hyperpo.py
python src/optuna.py
```

Each script evaluates a Random Forest using five-fold cross-validation. Searches may take several minutes.

## Graph outputs

- `src/Bay_Op_Gaussain.py` saves Bayesian optimization convergence and evaluation plots in `src/figures/`.
- `src/hyperpo.py` saves `hyperopt_results.png` in `src/figures/`.
- `src/optuna.py` saves `optuna_results.png` in this project directory.
