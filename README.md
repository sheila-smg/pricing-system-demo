# Pricing System Demo

A collection of end-to-end machine learning projects around insurance pricing.

## Projects

### [`competitive-pricing/`](./competitive-pricing/)

Models the **distribution of competitor prices** for a given insurance quote using monotonic multi-quantile XGBoost regression.

Key ideas:
- Predict 19 quantiles (q5 → q95) simultaneously to capture the full market price distribution
- Bayesian hyperparameter search with Optuna
- Market position scoring: where does our price sit in the predicted competitor distribution?
- Conversion rate analysis by market position

**Run order:** `training/training.ipynb` → `validating/validation_notebook.ipynb` → `results/*.ipynb`

---

### `pricing/` *(coming soon)*

Own price optimisation — using the competitive model to recommend prices that balance conversion probability and margin.
