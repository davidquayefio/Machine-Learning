from lightgbm import LGBMClassifier

BASELINE_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    max_depth=-1, min_child_samples=20, subsample=0.9,
    colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0
)

def build_model(params, seed=42):
    return LGBMClassifier(
        **params, objective="binary", metric="auc",
        verbosity=-1, n_jobs=-1, random_state=seed,
        importance_type="gain"
    )
