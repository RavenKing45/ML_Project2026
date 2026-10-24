"""Deterministic synthetic stand-in for the ULB credit-card dataset.

The output has exactly the real schema (``Time, V1..V28, Amount, Class``) so every squad can build
and test without the real CSV. It is *not* a model of the real data; it only reproduces the
properties the pipeline and tests depend on:

* Legit ``V_j ~ N(0, s_j)`` with ``s_j`` falling linearly from 2.0 (``V1``) to 0.3 (``V28``),
  mimicking PCA components ordered by explained variance.
* Fraud rows come from two clusters. Both shift ``V14, V12, V10, V17`` negative and ``V4, V11``
  positive (the features known to separate fraud in the real data), but by only ~1-2 standard
  deviations, so classes overlap and the task is not trivially separable. The clusters sit on
  opposite sides of ``V1``/``V3``, so SMOTE interpolating between them lands in legit territory
  (useful for showing SMOTE failure cases later).
* ``Amount`` is lognormal with some exact zeros; ``Time`` is uniform on [0, 172800] seconds.
* ``DUPLICATE_ROWS`` exact duplicate rows (``DUPLICATE_FRAUDS`` of them fraud) exercise
  deduplication.

Why the default fraud rate is 1.5% rather than the real 0.17%: with 6000 rows, 0.17% gives ~10
frauds, i.e. ~7 in a 70% training split and ~5-6 per 5-fold training fold. SMOTE with
``k_neighbors=5`` needs at least 6 minority samples in every training fold and stratified 5-fold
CV needs at least 5 per class, so tests would fail or be flaky. 1.5% gives 90 frauds (~63 train).
"""

import numpy as np
import pandas as pd

from fraudstudy.errors import DataValidationError

N_PCA_FEATURES = 28
PCA_COLUMNS: list[str] = [f"V{i}" for i in range(1, N_PCA_FEATURES + 1)]
COLUMNS: list[str] = ["Time", *PCA_COLUMNS, "Amount", "Class"]

TIME_MAX_SECONDS = 172_800  # two days, as in the real dataset
SCALE_FIRST, SCALE_LAST = 2.0, 0.3  # std of V1 and V28 for legit rows
V_DECIMALS = 5  # keeps the committed CSV well under 2 MB
AMOUNT_DECIMALS = 2

DUPLICATE_ROWS = 20
DUPLICATE_FRAUDS = 2

# Fraud cluster centres, in units of each feature's legit std s_j. Unlisted features: 0.
CLUSTER_SHIFTS: tuple[dict[str, float], ...] = (
    {
        "V14": -2.0,
        "V12": -1.6,
        "V10": -1.2,
        "V17": -1.6,
        "V4": 1.2,
        "V11": 1.3,
        "V1": 0.8,
        "V3": 0.9,
    },
    {
        "V14": -1.6,
        "V12": -0.7,
        "V10": -1.8,
        "V17": -0.6,
        "V4": 1.8,
        "V11": 0.5,
        "V1": -1.1,
        "V3": -1.3,
    },
)
CLUSTER_WEIGHTS: tuple[float, ...] = (0.6, 0.4)
FRAUD_SPREAD = 1.1  # fraud within-cluster std as a multiple of s_j

# Amount ~ exp(N(mu, sigma)); a fraction is set to exactly 0.
AMOUNT_LOGNORMAL = {0: (3.0, 1.3), 1: (2.6, 1.7)}
AMOUNT_ZERO_FRACTION = {0: 0.01, 1: 0.03}


def feature_scales() -> np.ndarray:
    """Return the legit standard deviation ``s_j`` of ``V1..V28``.

    Returns:
        Array of shape (28,) decreasing linearly from ``SCALE_FIRST`` to ``SCALE_LAST``.
    """
    return np.linspace(SCALE_FIRST, SCALE_LAST, N_PCA_FEATURES)


def _amounts(rng: np.random.Generator, n: int, label: int) -> np.ndarray:
    """Draw ``n`` lognormal amounts for class ``label`` with a fraction of exact zeros."""
    mu, sigma = AMOUNT_LOGNORMAL[label]
    amount = rng.lognormal(mean=mu, sigma=sigma, size=n)
    amount[rng.random(n) < AMOUNT_ZERO_FRACTION[label]] = 0.0
    return np.round(amount, AMOUNT_DECIMALS)


def _fraud_features(rng: np.random.Generator, n: int, scales: np.ndarray) -> np.ndarray:
    """Draw ``n`` fraud rows of ``V1..V28`` from the two-cluster mixture.

    Row i in cluster c: ``V_j = s_j * (shift_cj + FRAUD_SPREAD * z)``, ``z ~ N(0, 1)``.
    Every cluster gets at least one row so both always exist.
    """
    counts = np.round(np.asarray(CLUSTER_WEIGHTS) * n).astype(int)
    counts[-1] = n - counts[:-1].sum()
    counts = np.maximum(counts, 1)
    counts[0] = n - counts[1:].sum()
    blocks = []
    for shifts, count in zip(CLUSTER_SHIFTS, counts, strict=True):
        centre = np.array([shifts.get(col, 0.0) for col in PCA_COLUMNS])
        z = rng.standard_normal((count, N_PCA_FEATURES))
        blocks.append(scales * (centre + FRAUD_SPREAD * z))
    return np.vstack(blocks)


def make_synthetic(n_rows: int = 6000, fraud_rate: float = 0.015, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic synthetic dataset with the real ULB schema.

    ``n_rows`` counts the final rows *including* the ``DUPLICATE_ROWS`` duplicates, and the fraud
    count is ``round(n_rows * fraud_rate)`` over the final rows, so the observed fraud rate
    matches ``fraud_rate`` to within rounding.

    Args:
        n_rows: Total number of rows in the output.
        fraud_rate: Fraction of rows with ``Class == 1``.
        seed: Seed for ``numpy.random.default_rng``; same seed gives identical output.

    Returns:
        DataFrame with columns exactly ``Time, V1..V28, Amount, Class`` (``Class`` is int64,
        everything else float64), rows shuffled, index ``0..n_rows-1``.

    Raises:
        DataValidationError: If the arguments cannot produce a valid dataset.
    """
    if not 0.0 < fraud_rate < 1.0:
        raise DataValidationError(f"fraud_rate must be in (0, 1); got {fraud_rate}.")
    n_fraud = round(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud
    min_fraud = DUPLICATE_FRAUDS + len(CLUSTER_SHIFTS)
    if n_fraud < min_fraud or n_legit <= DUPLICATE_ROWS - DUPLICATE_FRAUDS:
        raise DataValidationError(
            f"n_rows={n_rows}, fraud_rate={fraud_rate} gives {n_fraud} frauds and {n_legit} "
            f"legit rows; need at least {min_fraud} frauds and more than "
            f"{DUPLICATE_ROWS - DUPLICATE_FRAUDS} legit rows. Increase n_rows or fraud_rate."
        )

    rng = np.random.default_rng(seed)
    scales = feature_scales()
    # Unique rows first; the duplicates are copies of some of them.
    u_fraud = n_fraud - DUPLICATE_FRAUDS
    u_legit = n_legit - (DUPLICATE_ROWS - DUPLICATE_FRAUDS)

    v_legit = rng.standard_normal((u_legit, N_PCA_FEATURES)) * scales
    v_fraud = _fraud_features(rng, u_fraud, scales)
    labels = np.concatenate([np.zeros(u_legit, dtype=np.int64), np.ones(u_fraud, dtype=np.int64)])
    n_unique = labels.size

    unique = pd.DataFrame(np.round(np.vstack([v_legit, v_fraud]), V_DECIMALS), columns=PCA_COLUMNS)
    unique.insert(0, "Time", np.floor(rng.uniform(0, TIME_MAX_SECONDS, n_unique)))
    unique["Amount"] = np.concatenate([_amounts(rng, u_legit, 0), _amounts(rng, u_fraud, 1)])
    unique["Class"] = labels

    dup_legit = rng.choice(u_legit, DUPLICATE_ROWS - DUPLICATE_FRAUDS, replace=False)
    dup_fraud = u_legit + rng.choice(u_fraud, DUPLICATE_FRAUDS, replace=False)
    duplicates = unique.iloc[np.concatenate([dup_legit, dup_fraud])]

    df = pd.concat([unique, duplicates], ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df[COLUMNS]
