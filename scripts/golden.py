"""Save, then check, a golden copy of every .json and .csv the sample run writes.

`save` runs scripts/run_all_sample.py and copies its .json/.csv output into .golden/. `check` runs
it again and compares every file with .golden/: same files, same keys at every level, numbers
equal within 1e-9. Used to prove that a refactor changed no result.

Usage::

    python scripts/golden.py save
    python scripts/golden.py check
"""

import json
import math
import shutil
import subprocess
import sys

import pandas as pd

from fraudstudy.config import resolve

GOLDEN_DIR = resolve(".golden")
OUT_DIR = resolve("artifacts/sample")
TOLERANCE = 1e-9
# Keys whose values are meant to differ between two identical runs: wall-clock times, run
# durations (including the experiment cells' search, fit and predict times), latency, the git
# commit (changes with every commit) and absolute paths (differ per machine). Their presence is
# still compared, only their values are skipped.
SKIP_KEYS = {
    "generated_at", "created_at", "timestamp", "elapsed", "elapsed_seconds", "duration_seconds",
    "latency_ms", "git_commit", "git_hash", "path", "model_path", "artifacts_dir",
    "started_at", "finished_at", "evaluated_at", "search_seconds", "fit_seconds",
    "predict_seconds", "total_seconds", "mean_fit_time",
}


def run_sample() -> None:
    """Run scripts/run_all_sample.py; exit if it fails."""
    result = subprocess.run([sys.executable, str(resolve("scripts/run_all_sample.py"))])
    if result.returncode != 0:
        sys.exit("run_all_sample.py failed; fix it before saving or checking the golden copy.")


def output_files(folder) -> list[str]:
    """Relative paths of every .json and .csv file below `folder`, sorted."""
    files = [p for p in folder.rglob("*") if p.suffix in (".json", ".csv")]
    return sorted(p.relative_to(folder).as_posix() for p in files)


def read(path) -> object:
    """A JSON file as parsed JSON, a CSV file as a list of row dicts."""
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path).to_dict(orient="records")


def compare(old: object, new: object, where: str, diffs: list[tuple[str, str, str]]) -> None:
    """Walk two values side by side and append (place, golden, now) for every difference."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in old.keys() - new.keys():
            diffs.append((f"{where}.{key}", "key", "missing"))
        for key in new.keys() - old.keys():
            diffs.append((f"{where}.{key}", "missing", "key"))
        for key in old:
            if key in new and key not in SKIP_KEYS:
                compare(old[key], new[key], f"{where}.{key}", diffs)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            diffs.append((where, f"{len(old)} items", f"{len(new)} items"))
            return
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            compare(a, b, f"{where}[{i}]", diffs)
    elif is_number(old) and is_number(new):
        both_nan = math.isnan(old) and math.isnan(new)
        if not both_nan and not math.isclose(old, new, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            diffs.append((where, repr(old), repr(new)))
    elif type(old) is not type(new) or old != new:  # the type check stops True == 1
        diffs.append((where, repr(old), repr(new)))


def is_number(value: object) -> bool:
    """True for int and float, false for bool (True == 1 must not pass as a number)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def save() -> int:
    """Run the sample pipeline and replace .golden/ with its .json and .csv output."""
    run_sample()
    shutil.rmtree(GOLDEN_DIR, ignore_errors=True)
    files = output_files(OUT_DIR)
    for name in files:
        (GOLDEN_DIR / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT_DIR / name, GOLDEN_DIR / name)
    print(f"saved {len(files)} files to {GOLDEN_DIR}")
    return 0


def check() -> int:
    """Run the sample pipeline again and compare its output with .golden/."""
    if not GOLDEN_DIR.is_dir():
        sys.exit("No golden copy yet. Run: python scripts/golden.py save")
    run_sample()
    golden, now = output_files(GOLDEN_DIR), output_files(OUT_DIR)
    diffs = [(name, "file", "missing") for name in golden if name not in now]
    diffs += [(name, "missing", "file") for name in now if name not in golden]
    for name in sorted(set(golden) & set(now)):
        compare(read(GOLDEN_DIR / name), read(OUT_DIR / name), name, diffs)
    if not diffs:
        print(f"golden check passed: {len(golden)} files, no differences")
        return 0
    print(f"{'where':<70} {'golden':>20} {'now':>20}")
    for where, old, new in diffs:
        print(f"{where:<70} {old:>20} {new:>20}")
    print(f"golden check FAILED: {len(diffs)} differences")
    return 1


if __name__ == "__main__":
    commands = {"save": save, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("Usage: python scripts/golden.py save|check")
    raise SystemExit(commands[sys.argv[1]]())
