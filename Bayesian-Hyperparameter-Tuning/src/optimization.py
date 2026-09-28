import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

from .objective import cv_auc


BOUNDS = {
    "n_estimators": (150, 700),
    "learning_rate": (0.01, 0.20),
    "num_leaves": (8, 128),
    "max_depth": (3, 12),
    "min_child_samples": (10, 150),
    "subsample": (0.60, 1.00),
    "colsample_bytree": (0.50, 1.00),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.0, 10.0),
}

INTS = {
    "n_estimators",
    "num_leaves",
    "max_depth",
    "min_child_samples",
}


def cast(params):
    params = dict(params)

    for name in INTS:
        params[name] = int(round(params[name]))

    return params


def random_search(X, y, n_iter=20, seed=42, cfg=None):
    rng = np.random.default_rng(seed)
    rows = []

    for trial in range(1, n_iter + 1):
        params = {}

        for name, (lower, upper) in BOUNDS.items():
            if name in INTS:
                params[name] = int(
                    rng.integers(lower, upper + 1)
                )
            else:
                params[name] = float(
                    rng.uniform(lower, upper)
                )

        mean_auc, cv_std = cv_auc(
            X,
            y,
            params,
            cfg,
        )

        rows.append({
            "trial": trial,
            "method": "random_search",
            "value": mean_auc,
            "cv_std": cv_std,
            **params,
        })

    return pd.DataFrame(rows)


def bayesian_search(
    X,
    y,
    init_points=5,
    n_iter=20,
    seed=42,
    cfg=None,
):
    rows = []

    def objective(**params):
        params = cast(params)

        mean_auc, cv_std = cv_auc(
            X,
            y,
            params,
            cfg,
        )

        rows.append({
            "trial": len(rows) + 1,
            "method": "bayesian_optimization",
            "value": mean_auc,
            "cv_std": cv_std,
            **params,
        })

        return mean_auc

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=BOUNDS,
        random_state=seed,
        verbose=0,
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    return optimizer, pd.DataFrame(rows)