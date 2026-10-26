"""Export the deployable models, calibrators, feature reference and demo presets for the backend.

Usage::

    python scripts/export_artifacts.py --source raw|sample --default-cell <model>__<technique>
        [--rerun-reason "<why>"] [--no-verify-hash]

The default cell is the TEAM's choice from the cross-validation results; the script refuses to
run without it (``--default-cell`` or ``deploy.default_cell`` in configs/experiment.yaml). The
deployed cells are ``deploy.cells`` (six LightGBM pipelines plus logistic regression for the
default cell's technique). Run it AFTER scripts/evaluate_test.py: the presets and the demo batch
are real test-set rows, so on the raw source the export is recorded in
``artifacts/results/TEST_SET_OPENED.json`` and a repeat needs ``--rerun-reason``.

Writes (raw): ``artifacts/models/{manifest.json, feature_reference.json, deploy/*.joblib}``,
``artifacts/demo/presets.json`` and ``data/sample/demo_batch.csv``. The sample source writes to
``artifacts/models/sample/`` and ``artifacts/demo/sample/`` instead.
"""

import argparse
import sys
from pathlib import Path

from fraudstudy.config import resolve
from fraudstudy.export import BYTES_PER_MB, ExportPaths, export_artifacts, export_paths

RESULTS_DIR = "artifacts/results"
MODELS_DIR = "artifacts/models"


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, export and print what was written.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success, 1 on an error (including a missing default cell or a guard refusal).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("raw", "sample"), required=True)
    parser.add_argument(
        "--default-cell",
        metavar="MODEL__TECHNIQUE",
        help="the team's default model, chosen from the CV results (required unless "
        "deploy.default_cell is set)",
    )
    parser.add_argument("--rerun-reason", help="required if an export already read the test set")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=resolve(RESULTS_DIR))
    parser.add_argument("--models-dir", type=Path, default=resolve(MODELS_DIR))
    parser.add_argument("--out-models", type=Path, help="default: by source (see above)")
    parser.add_argument("--out-demo", type=Path, help="default: by source (see above)")
    parser.add_argument("--out-batch", type=Path, help="default: by source (see above)")
    parser.add_argument("--decisions", type=Path, help="default: docs/decisions.md")
    args = parser.parse_args(argv)

    default = export_paths(args.source)
    out = ExportPaths(
        models=args.out_models or default.models,
        demo=args.out_demo or default.demo,
        batch=args.out_batch or default.batch,
    )
    try:
        report = export_artifacts(
            args.source,
            args.results_dir,
            args.models_dir,
            default_cell=args.default_cell,
            out=out,
            rerun_reason=args.rerun_reason,
            decisions_path=args.decisions,
            verify_hash=not args.no_verify_hash,
            log=lambda line: print(line, flush=True),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    manifest = report.manifest
    print(f"\n{'model id':<26}{'thr F1':>8}{'thr F2':>8}  calibrator  default")
    for m in manifest.models:
        mark = "yes" if m.id == manifest.default_model_id else ""
        cal = "yes" if m.has_calibrator else "no"
        print(f"{m.id:<26}{m.threshold_f1:>8.3f}{m.threshold_f2:>8.3f}  {cal:<10}  {mark}")
    print(f"\npresets ({len(report.presets.presets)}):")
    for p in report.presets.presets:
        print(f"  {p.id:<20} class {p.expected_class}  {p.label}")
    for preset_id in report.skipped_presets:
        print(f"  {preset_id:<20} skipped (no test row matches)")
    for text in report.warnings:
        print(f"warning: {text}")
    for path, size in sorted(report.sizes.items()):
        print(f"wrote      {path} ({size / BYTES_PER_MB:.2f} MB)")
    print(f"total      {report.total_mb:.2f} MB in {len(report.sizes)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
