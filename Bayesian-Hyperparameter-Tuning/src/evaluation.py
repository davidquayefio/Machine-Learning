from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, confusion_matrix

def evaluate(model,X,y):
    p=model.predict_proba(X)[:,1]
    return {
        "roc_auc":float(roc_auc_score(y,p)),
        "log_loss":float(log_loss(y,p)),
        "accuracy":float(accuracy_score(y,(p>=.5).astype(int))),
        "confusion_matrix":confusion_matrix(y,(p>=.5).astype(int)).tolist()
    }

def plot_history(df,path):
    plt.figure(figsize=(9,5))
    for method,g in df.groupby("method"):
        g=g.sort_values("trial")
        plt.plot(g["trial"],g["value"].cummax(),marker="o",markersize=3,label=method.replace("_"," ").title())
    plt.xlabel("Evaluation number"); plt.ylabel("Best-so-far CV ROC AUC")
    plt.title("Hyperparameter optimization history"); plt.grid(alpha=.2); plt.legend()
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()
