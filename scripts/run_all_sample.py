"""Run every pipeline step that exists today on the synthetic sample, in order, and time each one.

Called by hand or by scripts/golden.py. Each step is the normal script with its sample flags,
started as a separate process, so this file holds no science of its own. Everything is written
under artifacts/sample/ (emptied first), never into artifacts/results/.

Usage::

    python scripts/run_all_sample.py
"""

import shutil
import subprocess
import sys
import time

from fraudstudy.config import resolve

OUT_DIR = resolve("artifacts/sample")
RESULTS = str(OUT_DIR / "results")
FIGURES = str(OUT_DIR / "figures")
MODELS = str(OUT_DIR / "models")
TEST_EVAL = str(OUT_DIR / "test_eval")
TIME_LIMIT_SECONDS = 600  # the whole run must stay under 10 minutes

# Quick mode is meant for the 284k-row real file. On the 6000-row sample the quick subsample
# leaves about 4 frauds per training fold, too few for SMOTE (k_neighbors=5), so the audit and
# the baseline run on the full sample training split; that takes seconds anyway.
# The experiment cells do use --quick: on the sample it only cuts the folds, repeats and
# n_iter (the training subsample applies to raw only), which keeps the cells fast. Random forest
# is left out: with n_jobs=-1 it adds up its trees' votes in a different order on each run, so
# two identical runs differ by about 1e-6 and the golden check could never pass. It uses the
# same pipeline code as the other two models, and the tests cover it. merge_results.py is left
# out too: it only copies cell files, and it fails when a configured model has no cells.
STEPS = [
    ("data summary", ["data_summary.py", "--source", "sample"]),
    ("EDA", ["run_eda.py", "--source", "sample", "--results-dir", RESULTS,
             "--figures-dir", FIGURES]),
    ("resampling audit", ["run_resampling_audit.py", "--source", "sample",
                          "--results-dir", RESULTS]),
    ("baseline CV", ["run_baseline.py", "--source", "sample", "--results-dir", RESULTS]),
    ("experiment cells", ["run_experiments.py", "--source", "sample", "--quick",
                          "--models", "logreg", "lightgbm",
                          "--results-dir", RESULTS, "--models-dir", MODELS]),
    ("test evaluation", ["evaluate_test.py", "--source", "sample", "--results-dir", RESULTS,
                         "--models-dir", MODELS, "--output-dir", TEST_EVAL]),
    ("temporal check", ["run_temporal_check.py", "--source", "sample", "--model", "lightgbm",
                        "--results-dir", RESULTS, "--models-dir", MODELS,
                        "--output-dir", TEST_EVAL]),
    ("stats", ["run_stats.py", "--source", "sample", "--results-dir", RESULTS,
               "--output-dir", TEST_EVAL, "--figures-dir", FIGURES]),
    ("figures", ["make_figures.py", "--source", "sample", "--results-dir", RESULTS,
                 "--output-dir", TEST_EVAL, "--figures-dir", FIGURES]),
    # The default cell is the team's choice on the real data; any LightGBM cell works here.
    ("export", ["export_artifacts.py", "--source", "sample", "--default-cell", "lightgbm__smote",
                "--results-dir", RESULTS, "--models-dir", MODELS,
                "--out-models", str(OUT_DIR / "deploy"), "--out-demo", str(OUT_DIR / "demo"),
                "--out-batch", str(OUT_DIR / "demo" / "demo_batch.csv")]),
]


def main() -> int:
    """Run the steps one after the other; stop at the first one that fails."""
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    total_start = time.perf_counter()
    for name, (script, *args) in STEPS:
        print(f"--- {name}", flush=True)
        start = time.perf_counter()
        # Output is captured so the summary stays readable; it is shown again if the step fails.
        result = subprocess.run(
            [sys.executable, str(resolve("scripts") / script), *args],
            capture_output=True,
            text=True,
        )
        seconds = time.perf_counter() - start
        if result.returncode != 0:
            print(result.stdout + result.stderr)
            print(f"FAILED {name} after {seconds:.1f} s (exit code {result.returncode})")
            return 1
        print(f"    ok in {seconds:.1f} s")
    total = time.perf_counter() - total_start
    print(f"all {len(STEPS)} steps done in {total:.1f} s, output in {OUT_DIR}")
    if total > TIME_LIMIT_SECONDS:
        print(f"too slow: the sample run must finish in under {TIME_LIMIT_SECONDS} s")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
