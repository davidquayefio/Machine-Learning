import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from hyperopt import space_eval

def optimize(params, x, y):
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


def save_trial_plots(trials, output_dir):
    completed_trials = [trial for trial in trials.trials if "loss" in trial["result"]]
    trial_numbers = np.arange(1, len(completed_trials) + 1)
    accuracies = np.array([-trial["result"]["loss"] * 100 for trial in completed_trials])
    best_accuracy_so_far = np.maximum.accumulate(accuracies)

    parameter_values = {
        name: [trial["misc"]["vals"][name][0] for trial in completed_trials]
        for name in ("max_depth", "n_estimators", "max_features", "criterion")
    }
    parameter_values["criterion"] = ["gini" if value == 0 else "entropy" for value in parameter_values["criterion"]]

    figure, axes = plt.subplots(3, 2, figsize=(12, 12))
    accuracy_axis = axes[0, 0]
    accuracy_axis.plot(trial_numbers, accuracies, marker="o", label="Trial accuracy")
    accuracy_axis.plot(trial_numbers, best_accuracy_so_far, linestyle="--", label="Best so far")
    accuracy_axis.set(title="Hyperopt Search Progress", xlabel="Trial", ylabel="CV accuracy (%)")
    accuracy_axis.grid(True, alpha=0.3)
    accuracy_axis.legend()

    for axis, name, label in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]),
        ("max_depth", "n_estimators", "max_features"),
        ("Maximum depth", "Number of trees", "Maximum features"),
    ):
        axis.scatter(trial_numbers, parameter_values[name], color="teal")
        axis.set(title=f"Sampled {label}", xlabel="Trial", ylabel=label)
        axis.grid(True, alpha=0.3)

    criterion_axis = axes[2, 0]
    criterion_axis.scatter(
        trial_numbers,
        [0 if value == "gini" else 1 for value in parameter_values["criterion"]],
        color="darkorange",
    )
    criterion_axis.set(
        title="Sampled Criterion",
        xlabel="Trial",
        ylabel="Criterion",
        yticks=[0, 1],
        yticklabels=["gini", "entropy"],
    )
    criterion_axis.grid(True, axis="x", alpha=0.3)
    axes[2, 1].axis("off")

    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "hyperopt_results.png"
    figure.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return graph_path


if __name__ == "__main__":
    train_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
    df = pd.read_csv(train_path)

    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    param_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0.01, 1.0),
    }
    trials = Trials()

    result = fmin(
        lambda params: optimize(params, x=X, y=y),
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials
    )

    best_params = space_eval(param_space, result)
    print("Best cross-validation accuracy:", -min(trial["result"]["loss"] for trial in trials.trials))
    print("Best parameters:", best_params)

    graph_path = save_trial_plots(trials, Path(__file__).resolve().parent / "figures")
    print("Saved Hyperopt graphs to:", graph_path)