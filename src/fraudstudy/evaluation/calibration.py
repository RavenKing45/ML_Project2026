"""Optional isotonic calibration of risk scores (master guide Part 6, "Optional: calibration").

Resampled models give inflated scores, so a risk score is not a probability. An isotonic
calibrator maps scores to observed fraud rates with a non-decreasing step function.

Rules:

* Fit it on OUT-OF-FOLD TRAINING scores (each row scored by a fold model that did not train on
  it). Never fit it on test data.
* It never reverses the order of two scores, so it does not change which payments rank above
  which, nor how models/techniques rank on PR-AUC or ROC-AUC. Because it is a step function it
  can merge nearby scores into ties, so PR-AUC / ROC-AUC are always reported on the raw scores.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from fraudstudy.evaluation.metrics import check_inputs

PROBABILITY_RANGE = (0.0, 1.0)


def fit_isotonic(oof_scores: object, y: object) -> IsotonicRegression:
    """Fit an isotonic calibrator on out-of-fold training scores.

    Solves ``min_g sum_i (y_i - g(s_i))^2`` over non-decreasing ``g`` with values in [0, 1];
    scores outside the fitted range are clipped to its ends (``out_of_bounds="clip"``).

    Args:
        oof_scores: Out-of-fold scores of the TRAINING rows (never test scores).
        y: Their binary labels (1 = fraud).

    Returns:
        The fitted ``IsotonicRegression``.

    Raises:
        DataValidationError: If the inputs are invalid.
    """
    labels, s = check_inputs(y, oof_scores)
    low, high = PROBABILITY_RANGE
    calibrator = IsotonicRegression(y_min=low, y_max=high, increasing=True, out_of_bounds="clip")
    return calibrator.fit(s, labels)


def apply_calibrator(cal: IsotonicRegression, scores: object) -> np.ndarray:
    """Map risk scores to calibrated fraud probabilities.

    Args:
        cal: A calibrator from :func:`fit_isotonic`.
        scores: Risk scores.

    Returns:
        Calibrated probabilities in [0, 1], same length as ``scores``.
    """
    low, high = PROBABILITY_RANGE
    return np.clip(cal.predict(np.asarray(scores, dtype=float).ravel()), low, high)
