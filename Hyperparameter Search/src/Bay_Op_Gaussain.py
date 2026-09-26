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
from skopt.plots import plot_convergence, plot_evaluations

def optimize(params, param_names, x, y):
    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)
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
    train_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
    df = pd.read_csv(train_path)

    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 600, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, prior="uniform",name="max_features")
    ]
    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features"
    ]

    optimization_function = partial(
        optimize, 
        param_names=param_names, 
        x=X,
        y=y
    )

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
    )

    best_params = dict(zip(param_names, result.x))
    print("Best cross-validation accuracy:", -result.fun)
    print("Best parameters:", best_params)

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_convergence(result)
    convergence_path = output_dir / "bayesian_optimization_convergence.png"
    plt.gcf().savefig(convergence_path, dpi=150, bbox_inches="tight")
    plt.close()

    plot_evaluations(result)
    evaluations_path = output_dir / "bayesian_optimization_evaluations.png"
    plt.gcf().savefig(evaluations_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved convergence graph to:", convergence_path)
    print("Saved parameter evaluation graph to:", evaluations_path)