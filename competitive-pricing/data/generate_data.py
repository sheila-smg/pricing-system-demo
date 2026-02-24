"""
Synthetic data generator for the competitive pricing demo.

Produces two CSVs:

  train_data.csv  — long format (one row per quote × competitor), used for training.
  results_data.csv — wide format (one row per quote), used for validation / analysis.

Pricing is driven by a latent log-linear model so the features carry real
signal and the quantile regressor has something meaningful to learn.
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
N_QUOTES = 1_000
COMPETITORS = [f"comp_{i}" for i in range(1, 6)]

# ── Categorical vocabularies ──────────────────────────────────────────────────
CAT_VOCABS = {
    "cat_feature_1": [f"region_{c}" for c in "ABCDEFGH"],
    "cat_feature_2": ["standard", "commercial", "special"],
    "cat_feature_3": [f"tariff_{i}" for i in range(1, 7)],
    "cat_feature_4": ["domestic", "imported"],
    "cat_feature_5": ["gasoline", "flex", "diesel", "electric"],
    "cat_feature_6": ["hydraulic", "electric", "manual"],
    "cat_feature_7": ["SOHC", "DOHC"],
    "cat_feature_8": [f"gen_{i}" for i in range(1, 5)],
    "cat_feature_9": ["M", "F"],
}

REGION_EFFECT     = dict(zip(CAT_VOCABS["cat_feature_1"], RNG.uniform(-0.3, 0.3, 8)))
COMPETITOR_EFFECT = dict(zip(COMPETITORS, RNG.uniform(-0.2, 0.2, 5)))


# ── Quote generation ──────────────────────────────────────────────────────────

def generate_quotes(n: int) -> pd.DataFrame:
    df = pd.DataFrame()
    df["quote_id"] = [f"Q{i:05d}" for i in range(n)]

    df["num_feature_1"]  = RNG.integers(20, 71, n)
    df["num_feature_2"]  = RNG.integers(0, 16, n)
    df["num_feature_3"]  = RNG.uniform(0, 0.30, n).round(3)
    df["num_feature_4"]  = RNG.integers(0, 11, n)
    df["num_feature_5"]  = RNG.choice([2, 4], n, p=[0.1, 0.9])
    df["num_feature_6"]  = np.exp(RNG.normal(5.0, 0.4, n)).round(0)
    df["num_feature_7"]  = RNG.uniform(1.0, 5.5, n).round(1)
    df["num_feature_8"]  = RNG.integers(13, 21, n)
    df["num_feature_9"]  = RNG.integers(13, 21, n)
    df["num_feature_10"] = RNG.uniform(60, 100, n).round(1)
    df["num_feature_11"] = RNG.choice([2, 4, 8, 16], n, p=[0.05, 0.6, 0.25, 0.1])
    df["num_feature_12"] = RNG.uniform(8.0, 14.0, n).round(1)

    for i in range(1, 10):
        df[f"geo_feature_{i}"] = RNG.normal(0, 1, n).round(4)

    for col, vocab in CAT_VOCABS.items():
        df[col] = RNG.choice(vocab, n)

    return df


def generate_price(quotes: pd.DataFrame, competitor: str) -> np.ndarray:
    n = len(quotes)
    log_price = (
        7.5
        - 0.04  * quotes["num_feature_2"]
        + 0.40  * np.log(np.clip(quotes["num_feature_6"], 50, None))
        - 0.80  * quotes["num_feature_3"]
        + 0.004 * (quotes["num_feature_1"] - 45) ** 2 / 100
        + quotes["cat_feature_1"].map(REGION_EFFECT)
        + COMPETITOR_EFFECT[competitor]
        + RNG.normal(0, 0.18, n)
    )
    return np.exp(log_price).round(2)


# ── Train data (long format) ──────────────────────────────────────────────────

def build_train_data() -> pd.DataFrame:
    quotes = generate_quotes(N_QUOTES)
    rows = []
    for competitor in COMPETITORS:
        chunk = quotes.copy()
        chunk["competitor_id"] = competitor
        chunk["target_price"] = generate_price(quotes, competitor)
        rows.append(chunk)

    df = pd.concat(rows, ignore_index=True)
    id_cols  = ["quote_id", "competitor_id"]
    num_cols = [f"num_feature_{i}" for i in range(1, 13)]
    geo_cols = [f"geo_feature_{i}" for i in range(1, 10)]
    cat_cols = [f"cat_feature_{i}" for i in range(1, 10)]
    df = df[id_cols + num_cols + geo_cols + cat_cols + ["target_price"]]
    return df.sort_values(["quote_id", "competitor_id"]).reset_index(drop=True)


# ── Results data (wide format, one row per quote) ─────────────────────────────

def build_results_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot competitor prices to wide format and add own_price, scoring,
    conversion flag, and model_version for the analysis notebooks.
    """
    price_cols = [f"{c}_price" for c in COMPETITORS]

    # Pivot competitor prices
    wide = (
        train_df.pivot(index="quote_id", columns="competitor_id", values="target_price")
        .rename(columns=lambda c: f"{c}_price")
        .reset_index()
    )

    # Attach quote-level features (one row per quote_id)
    feature_cols = [c for c in train_df.columns if c not in ("competitor_id", "target_price")]
    features = train_df[feature_cols].drop_duplicates("quote_id")
    results = wide.merge(features, on="quote_id")

    # Market statistics
    results["market_min"]    = results[price_cols].min(axis=1).round(2)
    results["market_median"] = results[price_cols].median(axis=1).round(2)
    results["market_max"]    = results[price_cols].max(axis=1).round(2)

    # Own price: centered near market median with some spread
    results["own_price"] = (
        results["market_median"] * RNG.uniform(0.88, 1.12, len(results))
    ).round(2)

    # Scoring: fraction of competitors our own_price beats (0 = most expensive, 1 = cheapest)
    results["scoring"] = results.apply(
        lambda row: round((row["own_price"] < row[price_cols]).mean(), 2), axis=1
    )

    # Conversion: increases as scoring increases (cheaper relative to market = more likely to convert)
    # Three synthetic model versions — version 3 is the best-calibrated
    results["model_version"] = RNG.choice([1, 2, 3], len(results), p=[0.33, 0.34, 0.33])
    version_boost = {1: 0.00, 2: 0.03, 3: 0.06}
    base_prob  = 0.04 + 0.22 * results["scoring"]
    model_bump = results["model_version"].map(version_boost)
    conv_prob  = np.clip(base_prob + model_bump, 0, 1)
    results["converted"] = RNG.binomial(1, conv_prob)

    col_order = (
        ["quote_id"]
        + [f"num_feature_{i}" for i in range(1, 13)]
        + [f"geo_feature_{i}" for i in range(1, 10)]
        + [f"cat_feature_{i}" for i in range(1, 10)]
        + price_cols
        + ["market_min", "market_median", "market_max",
           "own_price", "scoring", "model_version", "converted"]
    )
    return results[col_order].reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating training data...")
    train_df = build_train_data()
    train_path = os.path.join(out_dir, "train_data.csv")
    train_df.to_csv(train_path, index=False)
    print(f"  {len(train_df):,} rows -> {train_path}")

    print("Generating results data...")
    results_df = build_results_data(train_df)
    results_path = os.path.join(out_dir, "results_data.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  {len(results_df):,} rows -> {results_path}")

    print("\ntarget_price (train):")
    print(train_df["target_price"].describe().round(2).to_string())
    print("\nown_price (results):")
    print(results_df["own_price"].describe().round(2).to_string())
    print("\nscoring distribution:")
    print(results_df["scoring"].describe().round(3).to_string())
    print(f"\nconversion rate: {results_df['converted'].mean():.3f}")
