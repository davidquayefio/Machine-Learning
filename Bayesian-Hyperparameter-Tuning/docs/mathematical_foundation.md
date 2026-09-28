# Mathematical Foundation

For hyperparameters `theta`, the validation objective is `J(theta)` and the goal is

`theta* = argmax_theta J(theta)`.

The expensive ML evaluation is treated as a noisy black-box function:

`y = f(theta) + epsilon`.

A Gaussian process places a distribution over possible functions. After observations `D_t`, it provides a predictive mean `mu_t(theta)` and uncertainty `sigma_t(theta)`.

An acquisition function `alpha(theta | D_t)` decides where to evaluate next:

`theta_(t+1) = argmax_theta alpha(theta | D_t)`.

The key conceptual point is that uncertainty is not itself the objective. The acquisition function combines predicted performance and uncertainty according to its policy.

The project uses mean K-fold ROC AUC as the objective:

`J(theta) = (1/K) sum_k AUC_k(theta)`.
