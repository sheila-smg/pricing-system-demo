"""
Synthetic Insurance Dataset Generator

Generates a realistic synthetic auto insurance portfolio for demonstrating
Poisson frequency and Gamma severity pricing models. All features are
anonymized (feature_1 … feature_14) and the data-generating process bakes
in relationships that the models should be able to learn.

Dataset schema
--------------
feature_1   : str       Date of birth (YYYY-MM-DD) — policyholder age proxy
feature_2   : int       Vehicle model year — vehicle age proxy
feature_3   : float     Insured value (log-normal)
feature_4   : float     Geographic coordinate 1 (continuous)
feature_5   : float     Geographic coordinate 2 (continuous)
feature_6   : str       Geographic region code (categorical, 20 levels)
feature_7   : str       Vehicle category code (categorical, 10 levels)
feature_8   : str       Binary policyholder attribute ('A' or 'B')
feature_9   : str       Coverage type (categorical, 5 levels T1–T5)
feature_10  : float     Geo-numerical enrichment — density-like
feature_11  : float     Geo-numerical enrichment — ratio [0,1]
feature_12  : float     Geo-numerical enrichment — ratio [0,1]
feature_13  : float     Geo-numerical enrichment — economic indicator
feature_14  : float     Geo-numerical enrichment — area measure
exposure    : float     Fraction of year insured [0.08, 1.0]
n_claims    : int       Observed claim count (Poisson)
claim_amount: float     Total incurred loss (0 when n_claims == 0, Gamma otherwise)

Usage
-----
    python data/generate_data.py
    # → writes data/insurance_data.csv
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_insurance_data(
    n_samples: int = 50_000,
    random_state: int = 42,
    output_path: str | None = "data/insurance_data.csv",
) -> pd.DataFrame:
    """
    Generate a synthetic insurance portfolio dataset.

    Parameters
    ----------
    n_samples : int
        Number of policy records to generate.
    random_state : int
        Random seed for reproducibility.
    output_path : str or None
        If provided, saves the dataset to this path as CSV.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(random_state)
    today = date.today()

    # ------------------------------------------------------------------
    # 1. CATEGORICAL LEVELS AND THEIR LATENT RISK EFFECTS
    #    (effects are baked into the DGP; the model must discover them)
    # ------------------------------------------------------------------
    regions = [f"R{i:02d}" for i in range(1, 21)]          # 20 geo regions
    categories = [f"CAT{i:02d}" for i in range(1, 11)]     # 10 vehicle categories
    coverage_types = ["T1", "T2", "T3", "T4", "T5"]        # T1=safest, T5=riskiest

    # Frequency log-linear effects
    region_freq_effects = rng.normal(0.0, 0.30, len(regions))
    cat_freq_effects    = rng.normal(0.0, 0.20, len(categories))
    cov_freq_effects    = np.array([-0.40, -0.15, 0.00, 0.25, 0.50])

    # Severity log-linear effects
    region_sev_effects  = rng.normal(0.0, 0.25, len(regions))
    cat_sev_effects     = rng.normal(0.0, 0.35, len(categories))
    cov_sev_effects     = np.array([-0.20, -0.05, 0.00, 0.10, 0.20])

    # Lookup maps (used in vectorised DGP)
    r_freq = dict(zip(regions, region_freq_effects))
    r_sev  = dict(zip(regions, region_sev_effects))
    c_freq = dict(zip(categories, cat_freq_effects))
    c_sev  = dict(zip(categories, cat_sev_effects))
    v_freq = dict(zip(coverage_types, cov_freq_effects))
    v_sev  = dict(zip(coverage_types, cov_sev_effects))

    # ------------------------------------------------------------------
    # 2. GENERATE FEATURES
    # ------------------------------------------------------------------

    # feature_1: date of birth — age range 18–80
    min_ord = (today - timedelta(days=80 * 366)).toordinal()
    max_ord = (today - timedelta(days=18 * 365)).toordinal()
    dob_ord = rng.integers(min_ord, max_ord, n_samples)
    feature_1 = [date.fromordinal(int(d)).strftime("%Y-%m-%d") for d in dob_ord]
    # Latent age used for data generation (not stored separately)
    age = (today.toordinal() - dob_ord) / 365.25

    # feature_2: vehicle model year (1–20 years old)
    current_year = today.year
    feature_2 = rng.integers(current_year - 20, current_year, n_samples).astype(int)
    vehicle_age = current_year - feature_2  # latent vehicle age

    # feature_3: insured value (log-normal, clipped)
    feature_3 = np.clip(
        rng.lognormal(mean=np.log(25_000), sigma=0.70, size=n_samples),
        2_000, 150_000,
    )

    # feature_4, feature_5: continuous geo-coordinates
    feature_4 = rng.uniform(-25.0, -10.0, n_samples)
    feature_5 = rng.uniform(-55.0, -35.0, n_samples)

    # feature_6: geographic region (20 levels, uniform distribution)
    feature_6 = rng.choice(regions, n_samples)

    # feature_7: vehicle category (10 levels)
    feature_7 = rng.choice(categories, n_samples)

    # feature_8: binary categorical attribute
    feature_8 = rng.choice(["A", "B"], n_samples, p=[0.58, 0.42])

    # feature_9: coverage type (5 ordered levels, realistic prevalence)
    feature_9 = rng.choice(
        coverage_types, n_samples, p=[0.15, 0.25, 0.30, 0.20, 0.10]
    )

    # features 10–14: geo-numerical enrichment
    feature_10 = np.clip(rng.lognormal(1.5, 1.2, n_samples), 0.1, 500)   # density
    feature_11 = rng.uniform(0.0, 1.0, n_samples)                          # ratio
    feature_12 = rng.uniform(0.0, 0.6, n_samples)                          # ratio
    feature_13 = np.clip(rng.lognormal(3.5, 0.6, n_samples), 5, 1_000)    # econ. index
    feature_14 = rng.uniform(0.0, 100.0, n_samples)                        # area measure

    # Exposure: fraction of year with active coverage
    exposure = rng.uniform(0.08, 1.0, n_samples)

    # ------------------------------------------------------------------
    # 3. DATA-GENERATING PROCESS — FREQUENCY (Poisson)
    # ------------------------------------------------------------------
    # Log-linear predictor for expected annual claim rate
    log_mu_freq = (
        -3.80                                                       # baseline
        + 0.025 * np.clip(30 - age, 0, None)                       # youth penalty
        - 0.008 * np.clip(age - 55, 0, None)                       # senior effect
        + 0.018 * vehicle_age                                       # vehicle age ↑ freq
        + 0.08  * np.log1p(feature_10 / 50)                        # urban density
        + np.vectorize(r_freq.get)(feature_6)                      # regional effect
        + np.vectorize(c_freq.get)(feature_7)                      # vehicle category
        + 0.12  * (feature_8 == "B").astype(float)                 # binary attribute
        + np.vectorize(v_freq.get)(feature_9)                      # coverage type
    )
    mu_freq = np.exp(log_mu_freq)
    n_claims = rng.poisson(mu_freq * exposure)

    # ------------------------------------------------------------------
    # 4. DATA-GENERATING PROCESS — SEVERITY (Gamma)
    # ------------------------------------------------------------------
    # Log-linear predictor for expected cost per claim
    log_mu_sev = (
        7.60                                                        # baseline ≈ 2 000
        + 0.50 * np.log(feature_3 / 25_000)                        # insured value ↑ sev
        - 0.012 * vehicle_age                                       # newer → costlier
        + np.vectorize(r_sev.get)(feature_6)                       # regional sev
        + np.vectorize(c_sev.get)(feature_7)                       # vehicle category sev
        + np.vectorize(v_sev.get)(feature_9)                       # coverage scope sev
    )
    mu_sev = np.exp(log_mu_sev)

    # For policies with n claims, total loss ~ Gamma(n * shape, mu_sev / shape)
    sev_shape = 2.5
    claim_amount = np.zeros(n_samples)
    mask = n_claims > 0
    claim_amount[mask] = rng.gamma(
        shape=n_claims[mask] * sev_shape,
        scale=mu_sev[mask] / sev_shape,
    )

    # ------------------------------------------------------------------
    # 5. INTRODUCE REALISTIC MISSINGNESS (~2–3 % per feature)
    # ------------------------------------------------------------------
    for arr in [feature_3, feature_4, feature_5, feature_10, feature_13]:
        idx = rng.choice(n_samples, size=int(0.025 * n_samples), replace=False)
        arr[idx] = np.nan

    # Missing DOB (~1 %)
    dob_list = list(feature_1)
    for i in rng.choice(n_samples, size=int(0.01 * n_samples), replace=False):
        dob_list[i] = None
    feature_1 = dob_list

    # ------------------------------------------------------------------
    # 6. ASSEMBLE DATAFRAME
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "feature_1":  feature_1,    # DOB → age (via ComputeAgeFromDOB)
            "feature_2":  feature_2,    # model year → vehicle age (via ComputeVehicleAge)
            "feature_3":  feature_3,    # insured value (numerical, log-normal)
            "feature_4":  feature_4,    # geo coordinate (numerical)
            "feature_5":  feature_5,    # geo coordinate (numerical)
            "feature_6":  feature_6,    # region (categorical, 20 levels)
            "feature_7":  feature_7,    # vehicle category (categorical, 10 levels)
            "feature_8":  feature_8,    # binary attribute ('A'/'B')
            "feature_9":  feature_9,    # coverage type (categorical, 5 levels)
            "feature_10": feature_10,   # geo-numerical enrichment
            "feature_11": feature_11,   # geo-numerical enrichment
            "feature_12": feature_12,   # geo-numerical enrichment
            "feature_13": feature_13,   # geo-numerical enrichment
            "feature_14": feature_14,   # geo-numerical enrichment
            "exposure":   np.round(exposure, 4),
            "n_claims":   n_claims,
            "claim_amount": np.round(claim_amount, 2),
        }
    )

    # ------------------------------------------------------------------
    # 7. SUMMARY STATISTICS
    # ------------------------------------------------------------------
    claimants = df[df["n_claims"] > 0]
    print("=" * 55)
    print("Synthetic insurance portfolio generated")
    print("=" * 55)
    print(f"  Records           : {len(df):,}")
    print(f"  Policies w/ claims: {len(claimants):,} ({len(claimants)/len(df):.1%})")
    print(
        f"  Observed frequency: "
        f"{df['n_claims'].sum() / df['exposure'].sum():.4f} claims/year"
    )
    if len(claimants) > 0:
        avg_sev = claimants["claim_amount"].sum() / claimants["n_claims"].sum()
        print(f"  Average severity  : {avg_sev:,.0f}")
    print(f"  Avg exposure      : {df['exposure'].mean():.3f} years")

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n  Saved → {output_path}")

    return df


if __name__ == "__main__":
    generate_insurance_data(
        n_samples=50_000,
        random_state=42,
        output_path="data/insurance_data.csv",
    )
