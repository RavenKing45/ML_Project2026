"""Evaluation toolkit: metrics, threshold choice, curves, bootstrap CIs and calibration.

Independent of the modelling code: every function takes labels and scores only.
"""

from fraudstudy.evaluation.bootstrap import bootstrap_ci, bootstrap_distribution
from fraudstudy.evaluation.calibration import apply_calibrator, fit_isotonic
from fraudstudy.evaluation.curves import downsample_indices, make_curves, threshold_table
from fraudstudy.evaluation.metrics import check_inputs, compute_metrics
from fraudstudy.evaluation.thresholds import best_threshold, fbeta_from_pr

__all__ = [
    "apply_calibrator",
    "best_threshold",
    "bootstrap_ci",
    "bootstrap_distribution",
    "check_inputs",
    "compute_metrics",
    "downsample_indices",
    "fbeta_from_pr",
    "fit_isotonic",
    "make_curves",
    "threshold_table",
]
