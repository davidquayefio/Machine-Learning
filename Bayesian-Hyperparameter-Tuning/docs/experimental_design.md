# Experimental Design

Primary question: Can Bayesian optimization reach strong validation performance with fewer model evaluations than random search?

Initial development budget:
- 3-fold stratified CV
- 20 random-search trials
- 5 Bayesian initialization evaluations
- 20 Bayesian iterations

Final research run:
- increase the number of evaluations
- repeat across multiple seeds
- record wall-clock time
- keep the test set untouched until the final evaluation

Primary figure: best-so-far ROC AUC versus evaluation number.

Do not claim universal superiority. Conclusions are restricted to the dataset, model, search space and budget actually tested.
