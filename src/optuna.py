import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing

from sklearn import pipeline
from functools import partial
from skopt import space
from skopt import gp_minimize
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

import optuna

def optimize(trial, x, y):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)

    model = ensemble.RandomForestClassifier(
        criterion=criterion,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features
    )
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = [] 
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1.0 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("C:/Users/user/Desktop/Hyperparameter/data/train.csv")

    X = df.drop("price_range", axis=1).values
    y = df.price_range.values
    optimization_function = partial(optimize, x=X, y=y)

    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function, n_trials=15)

    completed_trials = [trial for trial in study.trials if trial.value is not None]
    trial_numbers = [trial.number + 1 for trial in completed_trials]
    accuracies = [-trial.value * 100 for trial in completed_trials]
    best_accuracy_so_far = np.maximum.accumulate(accuracies)

    figure, (accuracy_axis, importance_axis) = plt.subplots(1, 2, figsize=(13, 5))
    accuracy_axis.plot(trial_numbers, accuracies, marker="o", label="Trial accuracy")
    accuracy_axis.plot(
        trial_numbers,
        best_accuracy_so_far,
        marker=".",
        linestyle="--",
        label="Best so far",
    )
    accuracy_axis.set(title="Optuna Search Progress", xlabel="Trial", ylabel="CV accuracy (%)")
    accuracy_axis.grid(True, alpha=0.3)
    accuracy_axis.legend()

    importances = optuna.importance.get_param_importances(study)
    importance_names = list(importances.keys())[::-1]
    importance_values = list(importances.values())[::-1]
    importance_axis.barh(importance_names, importance_values, color="teal")
    importance_axis.set(title="Hyperparameter Importance", xlabel="Importance")
    importance_axis.grid(axis="x", alpha=0.3)

    figure.tight_layout()
    graph_path = Path(__file__).resolve().parent.parent / "optuna_results.png"
    figure.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved optimization graphs to: {graph_path}")