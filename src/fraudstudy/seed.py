"""Global seeding. The single seed comes from ``configs/experiment.yaml``."""

import os
import random

import numpy as np


def set_global_seed(seed: int) -> int:
    """Seed Python's ``random``, NumPy's legacy global RNG and ``PYTHONHASHSEED``.

    Note: ``PYTHONHASHSEED`` only affects subprocesses started after this call; estimators and
    samplers should still receive ``random_state=seed`` explicitly.

    Args:
        seed: Non-negative integer seed.

    Returns:
        The seed that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
