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

from fraudstudy.paths import resolve

OUT_DIR = resolve("artifacts/sample")
RESULTS = str(OUT_DIR / "results")
FIGURES = str(OUT_DIR / "figures")
TIME_LIMIT_SECONDS = 600  # the whole run must stay under 10 minutes

# Quick mode is meant for the 284k-row real file. On the 6000-row sample the quick subsample
# leaves about 4 frauds per training fold, too few for SMOTE (k_neighbors=5), so the audit and
# the baseline run on the full sample training split; that takes seconds anyway.
# Later steps (experiment cells, merge, test evaluation, temporal check, stats, figures, export)
# are added here as their scripts land.
STEPS = [
    ("data summary", ["data_summary.py", "--source", "sample"]),
    ("EDA", ["run_eda.py", "--source", "sample", "--results-dir", RESULTS,
             "--figures-dir", FIGURES]),
    ("resampling audit", ["run_resampling_audit.py", "--source", "sample",
                          "--results-dir", RESULTS]),
    ("baseline CV", ["run_baseline.py", "--source", "sample", "--results-dir", RESULTS]),
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
