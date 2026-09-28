from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_data, split_features_target
from src.objective import CVConfig, cv_auc
from src.model import BASELINE_PARAMS
from src.optimization import random_search, bayesian_search

ROOT = Path(__file__).parent


def resolve_data_path():
    candidates = [
        ROOT / "data" / "processed" / "train.csv",
        ROOT / "data" / "raw" / "train.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find train.csv in data/processed or data/raw. "
        "Place the Kaggle train.csv in one of those folders first."
    )


def main():
    data_path = resolve_data_path()
    df = load_data(data_path)
    X, y = split_features_target(df)

    cfg = CVConfig(n_splits=3, random_state=42, max_rows=100000)
    print("Data:", X.shape, "positive rate:", y.mean())

    m, s = cv_auc(X, y, BASELINE_PARAMS, cfg)
    print("Baseline CV AUC:", m, "std:", s)

    rs = random_search(X, y, 20, 42, cfg)
    rs.to_csv(ROOT / "results" / "random_search_results.csv", index=False)

    opt, bo = bayesian_search(X, y, 5, 20, 42, cfg)
    bo.to_csv(ROOT / "results" / "bayesian_optimization_results.csv", index=False)

    print("Random best:", rs.loc[rs.value.idxmax()].to_dict())
    print("Bayesian best:", bo.loc[bo.value.idxmax()].to_dict())


if __name__ == "__main__":
    main()
