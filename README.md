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

### [`pricing/`](./pricing/)

Builds the **own price** from the ground up using frequency–severity modelling on a synthetic insurance portfolio.

Key ideas:
- Frequency model: Poisson GLM baseline + XGBoost challenger using the **frozen estimator** pattern (base model predictions become a feature in the challenger)
- Severity model: Gamma GLM baseline + XGBoost challenger, trained on claimants only and predicted on the full portfolio
- Pure premium = frequency × severity, assembled for both base and challenger
- Validation with **Ordered Lorenz curves** and normalised Gini coefficients

**Run order:** `python data/generate_data.py` → `01_frequency_model.ipynb` → `02_severity_model.ipynb` → `03_validation.ipynb`
