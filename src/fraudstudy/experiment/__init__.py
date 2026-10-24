"""Experiment layer: equal-budget tuning, out-of-fold thresholds, repeated CV, saving results.

Everything here works on the TRAINING split only; the test set is read once, later, by
``scripts/evaluate_test.py``.
"""

from fraudstudy.experiment.results import (
    CellRecord,
    MergeReport,
    cell_id,
    is_cell_done,
    load_cell_record,
    merge_results,
    save_cell,
    write_run_meta,
)
from fraudstudy.experiment.runner import (
    CellResult,
    Configs,
    Protocol,
    build_search_space,
    get_protocol,
    load_training_data,
    run_cell,
    run_experiments,
    space_size,
)

__all__ = [
    "CellRecord",
    "CellResult",
    "Configs",
    "MergeReport",
    "Protocol",
    "build_search_space",
    "cell_id",
    "get_protocol",
    "is_cell_done",
    "load_cell_record",
    "load_training_data",
    "merge_results",
    "run_cell",
    "run_experiments",
    "save_cell",
    "space_size",
    "write_run_meta",
]
