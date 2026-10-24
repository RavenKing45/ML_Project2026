"""Shared pytest fixtures."""

import copy
import dataclasses
import os

import pytest

from fraudstudy.experiment import Configs

# Threads per model fit in tests. experiment.yaml uses model_n_jobs = -1 (all cores), which suits
# the raw data but makes the 6,000-row sample several times slower on many-core machines
# (thread start-up dominates: ~16 s per LightGBM quick cell at 32 threads vs ~3 s at 8).
TEST_MODEL_N_JOBS = min(8, os.cpu_count() or 1)


def fast_configs() -> Configs:
    """The project configs with ``parallel.model_n_jobs`` capped for the small test sample.

    Only the thread count changes; seed, protocol, grids and budgets are untouched.

    Returns:
        A :class:`Configs` copy with ``experiment["parallel"]["model_n_jobs"]`` set to
        ``TEST_MODEL_N_JOBS``.
    """
    base = Configs.load()
    experiment = copy.deepcopy(base.experiment)
    experiment["parallel"]["model_n_jobs"] = TEST_MODEL_N_JOBS
    return dataclasses.replace(base, experiment=experiment)


@pytest.fixture(scope="session")
def test_cfgs() -> Configs:
    """Session-wide :func:`fast_configs`."""
    return fast_configs()
