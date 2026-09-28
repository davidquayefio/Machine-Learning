from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from .model import build_model

@dataclass
class CVConfig:
    n_splits: int = 3
    random_state: int = 42
    max_rows: int | None = 100000

def cv_auc(X, y, params, cfg=None):
    cfg = cfg or CVConfig()
    if cfg.max_rows and len(X) > cfg.max_rows:
        rng = np.random.default_rng(cfg.random_state)
        idx = rng.choice(len(X), cfg.max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    skf = StratifiedKFold(cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    scores=[]
    for tr, va in skf.split(X,y):
        model=build_model(params,cfg.random_state)
        model.fit(X.iloc[tr],y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[va],model.predict_proba(X.iloc[va])[:,1]))
    return float(np.mean(scores)), float(np.std(scores))
