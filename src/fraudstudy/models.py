"""Model builders for the three study models (master guide Part 6, "The 3 models").

Search spaces live in ``configs/models.yaml``; this module only constructs estimators. Every
model gets ``random_state=seed`` (the single seed from ``configs/experiment.yaml``), which a
caller cannot override.
"""

from collections.abc import Callable
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_NAMES: tuple[str, ...] = ("logreg", "random_forest", "lightgbm")
# Models whose fits can use several cores (``n_jobs``); logistic regression (lbfgs, binary)
# is single-threaded.
PARALLEL_MODELS: tuple[str, ...] = ("random_forest", "lightgbm")
DEFAULT_N_JOBS = -1


def _logreg(seed: int, params: dict[str, Any]) -> BaseEstimator:
    """``LogisticRegression(solver="lbfgs")``; ``n_jobs`` is ignored (binary lbfgs)."""
    params = {k: v for k, v in params.items() if k != "n_jobs"}
    return LogisticRegression(solver="lbfgs", random_state=seed, **params)


def _random_forest(seed: int, params: dict[str, Any]) -> BaseEstimator:
    """``RandomForestClassifier`` with ``n_jobs=-1`` unless ``params`` sets it."""
    return RandomForestClassifier(random_state=seed, **{"n_jobs": DEFAULT_N_JOBS, **params})


def _lightgbm(seed: int, params: dict[str, Any]) -> BaseEstimator:
    """``LGBMClassifier(verbose=-1)`` with ``n_jobs=-1`` unless ``params`` sets it."""
    return LGBMClassifier(random_state=seed, verbose=-1, **{"n_jobs": DEFAULT_N_JOBS, **params})


_BUILDERS: dict[str, Callable[[int, dict[str, Any]], BaseEstimator]] = {
    "logreg": _logreg,
    "random_forest": _random_forest,
    "lightgbm": _lightgbm,
}


def get_model(name: str, seed: int, **params: Any) -> BaseEstimator:
    """Build an unfitted classifier with ``random_state=seed``.

    * ``logreg``: ``LogisticRegression(solver="lbfgs")``.
    * ``random_forest``: ``RandomForestClassifier(n_jobs=-1)``.
    * ``lightgbm``: ``LGBMClassifier(verbose=-1, n_jobs=-1)``.

    Args:
        name: One of :data:`MODEL_NAMES`.
        seed: Random seed from ``configs/experiment.yaml``.
        **params: Hyper-parameters (e.g. ``C=0.1``); ``n_jobs`` overrides the default for the
            tree models and is ignored by logistic regression.

    Returns:
        The unfitted estimator.

    Raises:
        ValueError: If ``name`` is unknown or ``params`` sets ``random_state``.
    """
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(MODEL_NAMES)}.")
    if "random_state" in params:
        raise ValueError(
            "Do not pass random_state; every model uses the seed from configs/experiment.yaml."
        )
    return _BUILDERS[name](int(seed), dict(params))
