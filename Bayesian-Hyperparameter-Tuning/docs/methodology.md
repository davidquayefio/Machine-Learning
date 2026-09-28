# Methodology

The Santander dataset is treated as an anonymized binary classification problem. `ID_code` is excluded from predictors and `target` is the response.

A stratified development/test split must be created before hyperparameter optimization. The test set is never used to choose hyperparameters.

LightGBM is used as the model because it is well suited to large tabular data. The primary metric is ROC AUC.

Three methods are compared under the same search space and evaluation budget:

1. Fixed baseline.
2. Random search.
3. Gaussian-process Bayesian optimization.

For the final study, repeat the comparison over multiple random seeds and report the distribution of best-so-far performance, not only one run.
