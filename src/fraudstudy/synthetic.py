"""Make a synthetic stand-in for the ULB credit-card dataset, with exactly its columns.

scripts/generate_synthetic_data.py calls make_synthetic() to write data/sample/, which every test
uses so nobody needs the real CSV. It is not a model of the real data; it only has the
properties the pipeline and the tests depend on (see the comments below).
"""

import numpy as np
import pandas as pd

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
COLUMNS = ["Time", *PCA_COLUMNS, "Amount", "Class"]
TIME_MAX_SECONDS = 172_800  # two days, as in the real dataset

# Exact duplicate rows, so deduplication has something to remove (2 of the 20 are fraud).
DUPLICATE_ROWS = 20
DUPLICATE_FRAUDS = 2

# Fraud rows come from two clusters, given in units of each feature's legit standard deviation
# (unlisted features: 0). Both push V14, V12, V10, V17 down and V4, V11 up, the features that
# separate fraud in the real data, but only by 1-2 standard deviations so the classes overlap.
# The clusters sit on opposite sides of V1/V3, so SMOTE points between them land among legit rows.
CLUSTER_SHIFTS = [
    {"V14": -2.0, "V12": -1.6, "V10": -1.2, "V17": -1.6, "V4": 1.2, "V11": 1.3, "V1": 0.8,
     "V3": 0.9},
    {"V14": -1.6, "V12": -0.7, "V10": -1.8, "V17": -0.6, "V4": 1.8, "V11": 0.5, "V1": -1.1,
     "V3": -1.3},
]
FIRST_CLUSTER_SHARE = 0.6
FRAUD_SPREAD = 1.1  # fraud spread inside a cluster, as a multiple of the legit std

# Amount = exp(N(mu, sigma)) per class (0 = legit, 1 = fraud), with a share set to exactly 0.
AMOUNT_LOGNORMAL = {0: (3.0, 1.3), 1: (2.6, 1.7)}
AMOUNT_ZERO_SHARE = {0: 0.01, 1: 0.03}


def feature_scales() -> np.ndarray:
    """Legit standard deviation of V1..V28: from 2.0 down to 0.3, like PCA components."""
    return np.linspace(2.0, 0.3, len(PCA_COLUMNS))


def make_amounts(rng: np.random.Generator, n: int, label: int) -> np.ndarray:
    """Draw `n` lognormal amounts for class `label`, a share of them exactly 0, in cents."""
    mu, sigma = AMOUNT_LOGNORMAL[label]
    amount = rng.lognormal(mean=mu, sigma=sigma, size=n)
    amount[rng.random(n) < AMOUNT_ZERO_SHARE[label]] = 0.0
    return np.round(amount, 2)


def make_fraud_features(rng: np.random.Generator, n: int, scales: np.ndarray) -> np.ndarray:
    """Draw `n` fraud rows of V1..V28: V_j = s_j * (cluster shift_j + FRAUD_SPREAD * z)."""
    first = round(FIRST_CLUSTER_SHARE * n)
    blocks = []
    for shifts, count in zip(CLUSTER_SHIFTS, [first, n - first], strict=True):
        centre = np.array([shifts.get(col, 0.0) for col in PCA_COLUMNS])
        z = rng.standard_normal((count, len(PCA_COLUMNS)))
        blocks.append(scales * (centre + FRAUD_SPREAD * z))
    return np.vstack(blocks)


def make_synthetic(n_rows: int = 6000, fraud_rate: float = 0.015, seed: int = 42) -> pd.DataFrame:
    """Build the synthetic dataset; the same arguments always give the same rows.

    n_rows counts the final rows including the duplicates, and there are round(n_rows *
    fraud_rate) frauds. The default rate is 1.5%, not the real 0.17%: with 6000 rows, 0.17% gives
    about 6 frauds per 5-fold training fold, and SMOTE with k_neighbors=5 needs at least 6.
    Columns are Time, V1..V28, Amount, Class; rows are shuffled.
    """
    n_fraud = round(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud
    legit_duplicates = DUPLICATE_ROWS - DUPLICATE_FRAUDS
    if not 0.0 < fraud_rate < 1.0 or n_fraud < DUPLICATE_FRAUDS + 2 or n_legit <= legit_duplicates:
        raise ValueError(
            f"n_rows={n_rows}, fraud_rate={fraud_rate} gives {n_fraud} frauds and {n_legit} legit "
            f"rows; need a fraud_rate in (0, 1), at least {DUPLICATE_FRAUDS + 2} frauds and more "
            f"than {legit_duplicates} legit rows. Increase n_rows or fraud_rate."
        )

    rng = np.random.default_rng(seed)
    scales = feature_scales()
    # Unique rows first; the duplicates are copies of some of them.
    unique_fraud = n_fraud - DUPLICATE_FRAUDS
    unique_legit = n_legit - legit_duplicates
    v_legit = rng.standard_normal((unique_legit, len(PCA_COLUMNS))) * scales
    v_fraud = make_fraud_features(rng, unique_fraud, scales)
    labels = np.concatenate([np.zeros(unique_legit, dtype=np.int64),
                             np.ones(unique_fraud, dtype=np.int64)])

    # 5 decimals keep the committed CSV well under 2 MB.
    unique = pd.DataFrame(np.round(np.vstack([v_legit, v_fraud]), 5), columns=PCA_COLUMNS)
    unique.insert(0, "Time", np.floor(rng.uniform(0, TIME_MAX_SECONDS, labels.size)))
    unique["Amount"] = np.concatenate([make_amounts(rng, unique_legit, 0),
                                       make_amounts(rng, unique_fraud, 1)])
    unique["Class"] = labels

    copied_legit = rng.choice(unique_legit, legit_duplicates, replace=False)
    copied_fraud = unique_legit + rng.choice(unique_fraud, DUPLICATE_FRAUDS, replace=False)
    duplicates = unique.iloc[np.concatenate([copied_legit, copied_fraud])]

    df = pd.concat([unique, duplicates], ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df[COLUMNS]
