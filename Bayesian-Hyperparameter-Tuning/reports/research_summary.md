# Research Summary

## Research question
Can Gaussian-process Bayesian optimization efficiently tune LightGBM on the Santander Customer Transaction Prediction dataset?

## Results
On the development set, the fixed LightGBM baseline achieved a mean 3-fold CV ROC AUC of approximately 0.8743. Random search improved this to about 0.8838 after 20 trials, while Bayesian optimization achieved a slightly higher value of approximately 0.8839 after 25 total evaluations (5 initialization points + 20 optimization iterations). The improvement over random search was very small, on the order of 0.0001 ROC AUC.

## Interpretation
The evidence suggests that Bayesian optimization can identify a competitive configuration with a similar or marginally better score than random search under the same evaluation budget. However, the difference is so small that it should not be interpreted as a strong practical advantage. The random search baseline already performed well, indicating that the search space contains a broad region of good-performing configurations and that a modest budget is sufficient to reach near-optimal performance for this dataset and model family.

The more meaningful conclusion is therefore not that Bayesian optimization decisively dominates random search, but that it is a viable and efficient tuning strategy when evaluation budgets are limited. In this experiment, Bayesian optimization offered a slight improvement in the best observed validation score while requiring a comparable number of model evaluations, and the gain was too small to establish a robust superiority claim without a multi-seed study.

## Conclusion
For this Santander transaction dataset and the tested LightGBM search space, Bayesian optimization produced a marginally better cross-validated ROC AUC than random search, but the improvement was modest and likely not practically significant in isolation. The stronger conclusion is that both approaches are effective for this tuning task, with Bayesian optimization providing a slight edge in this single experimental run. A more rigorous final statement would require repeating the comparison over multiple random seeds and reporting the distribution of best-so-far performance and final held-out results.

## Limitations
The features are anonymized; ROC AUC does not measure calibration; finite-budget Bayesian optimization is not guaranteed to find the global optimum; one random seed is insufficient evidence for general superiority.
