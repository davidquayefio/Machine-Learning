# Hyperparameter Search

A small machine-learning project that compares hyperparameter-search approaches for a Random Forest classifier. The model predicts `price_range` from the supplied mobile-phone feature data.

## Project contents

- `data/train.csv`: training data used by the search scripts.
- `data/test.csv`: test data for later evaluation or prediction.
- `src/grid_search.py`: exhaustive scikit-learn `GridSearchCV` search.
- `src/ran_search.py`: scikit-learn `RandomizedSearchCV` search.
- `src/Bay_Op_Gaussain.py`: Gaussian-process Bayesian optimization with scikit-optimize.
- `src/hyperpo.py`: Tree-structured Parzen estimator search with Hyperopt; saves trial charts to `src/figures/hyperopt_results.png`.
- `src/search2.py`: Optuna search; saves trial charts to `optuna_results.png` in the project root.
- `src/figures/`: graph output directory for the Bayesian optimization script.

## Requirements

Python 3.10 or newer is recommended. Dependencies are listed in `requirements.txt`.

### Create and install the environment

From the project root in Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run a search

Run commands from the project root:

```bash
python src/grid_search.py
python src/ran_search.py
python src/Bay_Op_Gaussain.py
python src/hyperpo.py
python src/Optuna.py
```

The search scripts use five-fold cross-validation and print the best score and/or parameters when they finish. The full searches can take several minutes depending on the machine.

## Graphs

- `src/Bay_Op_Gaussain.py` writes `bayesian_optimization_convergence.png` and `bayesian_optimization_evaluations.png` to `src/figures/`.
- `src/hyperpo.py` writes `hyperopt_results.png` to `src/figures/`.
- `src/search2.py` writes `optuna_results.png` to the project root.

## Data path note

Some scripts currently use the absolute path `C:/Users/user/Desktop/Hyperparameter/data/train.csv`. If you move the project, update that path in those scripts or replace it with a project-relative path such as `data/train.csv` when running from the project root.
