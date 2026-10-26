"""Tests for fraudstudy.evaluation.stats, fraudstudy.evaluation.figures, scripts/run_stats.py and
scripts/make_figures.py.

Statistics are checked on hand-built fold-score tables; figures are drawn from the contract
example fixtures plus synthetic fold scores (at a low dpi, except one 300-dpi check, to keep the
suite fast). The script test uses one real quick cell on the synthetic sample, cloned for the
other techniques.
"""

import importlib.util
import itertools
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pydantic import ValidationError
from scipy import stats as sps

from fraudstudy.config import resolve
from fraudstudy.contracts import (
    ComparisonTable,
    CurvesFile,
    PairwiseTest,
    ResamplingAudit,
    StatsResult,
)
from fraudstudy.evaluation import figures
from fraudstudy.evaluation import stats as st
from fraudstudy.experiment import Configs, run_experiments
from fraudstudy.experiment.results import (
    CELLS_DIR,
    FOLD_SCORES_FILE,
    FoldScore,
    cell_id,
    fold_frame,
    load_cell_record,
    source_fold_scores,
    upsert_fold_scores,
)
from fraudstudy.experiment.test_eval import COMPARISON_FILE, CURVES_FILE, SAMPLE_EVAL_DIR
from fraudstudy.io import load_json, save_json

MODELS = ["logreg", "random_forest", "lightgbm"]
TECHNIQUES = ["baseline", "ros", "rus", "smote", "smote_rus", "class_weight"]
EXAMPLES = resolve("docs/contracts/examples")
N_REPEATS, N_FOLDS = 3, 5
FAST_DPI = 40


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module (scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(name, resolve(f"scripts/{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fold_scores(
    shift: dict[str, float] | None = None, noise: float = 0.02, seed: int = 0
) -> pd.DataFrame:
    """Synthetic fold scores: 3 models x 6 techniques x 15 (repeat, fold) blocks.

    Each block has its own difficulty (shared by every technique, which is why pairing matters);
    ``shift`` adds a constant to a technique on every block.
    """
    rng = np.random.default_rng(seed)
    shift = shift or {}
    rows = []
    for model in MODELS:
        for repeat, fold in itertools.product(range(N_REPEATS), range(N_FOLDS)):
            level = rng.uniform(0.5, 0.8)
            for technique in TECHNIQUES:
                value = level + shift.get(technique, 0.0) + rng.normal(0, noise)
                rows.append(
                    {
                        "model": model,
                        "technique": technique,
                        "repeat": repeat,
                        "fold": fold,
                        "fold_id": f"{repeat}-{fold}",
                        "pr_auc": float(np.clip(value, 0, 1)),
                        "roc_auc": 0.9,
                    }
                )
    return pd.DataFrame(rows)


def _compute(fold_scores: pd.DataFrame) -> StatsResult:
    return st.compute_stats(
        fold_scores, MODELS, TECHNIQUES, alpha=0.05, n_bootstrap=1000, confidence=0.95, seed=42
    )


# --- Nemenyi CD ---------------------------------------------------------------------------------


def test_nemenyi_cd_matches_demsar_2006() -> None:
    assert st.nemenyi_q(6, 0.05) == pytest.approx(2.850, abs=1e-3)
    assert st.critical_difference(6, 15, 0.05) == pytest.approx(1.95, abs=0.01)


def test_cd_shrinks_with_more_blocks_and_rejects_bad_input() -> None:
    assert st.critical_difference(6, 45, 0.05) < st.critical_difference(6, 15, 0.05)
    with pytest.raises(ValueError):
        st.critical_difference(1, 15, 0.05)


# --- Holm, Wilcoxon, Friedman -------------------------------------------------------------------


def test_holm_adjust_known_values() -> None:
    adjusted = st.holm_adjust([0.01, 0.04, 0.03, 0.2, 0.001])
    np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.2, 0.005])
    assert st.holm_adjust([0.5, 0.9]).max() <= 1.0
    assert st.holm_adjust([]).size == 0


def test_wilcoxon_counts_and_effect_sizes() -> None:
    a = np.array([0.5, 0.6, 0.7, 0.8])
    b = np.array([0.4, 0.6, 0.75, 0.7])
    r = st.wilcoxon_paired(a, b)
    assert (r["wins"], r["ties"], r["losses"], r["n_pairs"]) == (2, 1, 1, 4)
    assert r["median_difference"] == pytest.approx(0.05)
    assert r["mean_difference"] == pytest.approx(0.0375)
    with pytest.raises(ValueError):
        st.wilcoxon_paired([0.1, 0.2], [0.1])


def test_friedman_matches_scipy_with_ties() -> None:
    rng = np.random.default_rng(1)
    x = rng.random((15, 6))
    x[:, 2] = x[:, 3]
    statistic, p = st.friedman(pd.DataFrame(x))
    reference = sps.friedmanchisquare(*x.T)
    assert statistic == pytest.approx(reference.statistic)
    assert p == pytest.approx(reference.pvalue)
    with pytest.raises(ValueError):
        st.friedman(pd.DataFrame(x[:, :2]))


def test_average_ranks_one_is_best() -> None:
    table = pd.DataFrame({"a": [0.9, 0.8], "b": [0.5, 0.5], "c": [0.5, 0.6]})
    ranks = st.average_ranks(table)
    assert ranks["a"] == 1.0
    assert ranks.sum() == pytest.approx(6.0)  # 1 + 2 + 3 per block


# --- acceptance cases ---------------------------------------------------------------------------


def test_technique_better_on_every_fold_is_significant_after_holm() -> None:
    result = _compute(_fold_scores(shift={"smote": 0.05}, noise=0.005))
    for model in MODELS:
        smote = next(t for t in result.pairwise if t.model == model and t.technique_a == "smote")
        assert (smote.wins, smote.ties, smote.losses) == (15, 0, 0)
        assert smote.p_value_adjusted < 0.05 and smote.significant
        assert smote.median_difference == pytest.approx(0.05, abs=0.01)
        friedman = next(f for f in result.friedman if f.scope == model)
        assert min(friedman.average_ranks, key=friedman.average_ranks.get) == "smote"


def test_identical_scores_give_p_one_without_crashing() -> None:
    result = _compute(_fold_scores(noise=0.0))
    assert len(result.pairwise) == len(MODELS) * (len(TECHNIQUES) - 1)
    for t in result.pairwise:
        assert t.p_value == 1.0 and t.p_value_adjusted == 1.0 and not t.significant
        assert (t.wins, t.ties, t.losses) == (0, 15, 0)
        assert t.statistic == 0.0 and t.difference == 0.0
    for f in result.friedman:
        assert f.statistic == 0.0 and f.p_value == 1.0
        assert set(f.average_ranks.values()) == {3.5}


def test_stats_result_shape_and_notes() -> None:
    result = _compute(_fold_scores())
    assert StatsResult.model_validate(result.model_dump(mode="json")) == result
    assert [f.scope for f in result.friedman] == [*MODELS, "all"]
    pooled = result.friedman[-1]
    assert pooled.n_blocks == len(MODELS) * N_REPEATS * N_FOLDS
    assert all(f.n_blocks == 15 for f in result.friedman[:-1])
    assert result.friedman[0].critical_difference == pytest.approx(1.95, abs=0.01)
    assert {(t.technique_a, t.technique_b) for t in result.pairwise} == {
        (t, "baseline") for t in TECHNIQUES[1:]
    }
    assert st.DEPENDENCE_NOTE in result.notes and result.correction == "holm"


def test_partial_models_and_techniques() -> None:
    scores = _fold_scores()
    scores = scores[(scores["model"] == "logreg") & scores["technique"].isin(TECHNIQUES[:3])]
    result = _compute(scores)
    assert [f.scope for f in result.friedman] == ["logreg"]  # no pooled test for one model
    assert len(result.pairwise) == 2


# --- pairing checks -----------------------------------------------------------------------------


def test_paired_scores_rejects_broken_pairing() -> None:
    scores = _fold_scores()
    bad_id = scores.copy()
    bad_id.loc[(bad_id["technique"] == "ros") & (bad_id["fold"] == 0), "fold_id"] = "other"
    with pytest.raises(ValueError, match="different folds"):
        st.paired_scores(bad_id, "logreg", TECHNIQUES)
    missing = scores.drop(scores.index[(scores["technique"] == "rus")][:1])
    with pytest.raises(ValueError, match="lack scores"):
        st.paired_scores(missing, "logreg", TECHNIQUES)
    with pytest.raises(ValueError, match="Duplicate"):
        st.paired_scores(pd.concat([scores, scores.iloc[:1]]), "logreg", TECHNIQUES)
    with pytest.raises(ValueError, match="No fold scores"):
        st.paired_scores(scores, "svm", TECHNIQUES)


def test_missing_baseline_is_an_error() -> None:
    scores = _fold_scores()
    with pytest.raises(ValueError, match="baseline"):
        _compute(scores[scores["technique"] != "baseline"])


def test_pairwise_contract_checks_win_tie_loss_total() -> None:
    base = {
        "model": "logreg",
        "technique_a": "ros",
        "technique_b": "baseline",
        "test": "wilcoxon",
        "metric": "average_precision",
        "difference": 0.0,
        "difference_ci": None,
        "statistic": 0.0,
        "p_value": 1.0,
        "p_value_adjusted": 1.0,
        "significant": False,
    }
    PairwiseTest.model_validate(base)  # 1.3 fields are optional
    PairwiseTest.model_validate({**base, "n_pairs": 3, "wins": 1, "ties": 1, "losses": 1})
    with pytest.raises(ValidationError):
        PairwiseTest.model_validate({**base, "n_pairs": 4, "wins": 1, "ties": 1, "losses": 1})


# --- findings.md --------------------------------------------------------------------------------


def test_findings_has_tables_and_empty_interpretation(tmp_path: Path) -> None:
    result = _compute(_fold_scores())
    comparison = ComparisonTable.model_validate(load_json(EXAMPLES / "comparison.json"))
    path = tmp_path / "findings.md"
    text = st.findings_markdown(result, comparison, [tmp_path / "x.png"], path, "raw")
    assert text.rstrip().endswith(st.INTERPRETATION_HEADING)
    for heading in ("## Test-set results", "## Wilcoxon", "## Friedman", "## Notes"):
        assert heading in text
    assert "(x.png)" in text and st.DEPENDENCE_NOTE in text
    assert text.count("| Logistic regression |") >= 6
    no_test = st.findings_markdown(result, None, [], path, "sample")
    assert "has not been evaluated yet" in no_test


def test_findings_keeps_team_interpretation(tmp_path: Path) -> None:
    path = tmp_path / "findings.md"
    assert st.existing_interpretation(path) == ""
    result = _compute(_fold_scores())
    path.write_text(st.findings_markdown(result, None, [], path, "raw"), encoding="utf-8")
    team_text = "Our reading of the results.\n\n* point one"
    path.write_text(path.read_text(encoding="utf-8") + "\n" + team_text + "\n", encoding="utf-8")
    assert st.existing_interpretation(path) == team_text
    again = st.findings_markdown(
        result, None, [], path, "raw", interpretation=st.existing_interpretation(path)
    )
    assert again.rstrip().endswith(team_text)


# --- figures ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def example_artifacts() -> dict[str, object]:
    comparison = ComparisonTable.model_validate(load_json(EXAMPLES / "comparison.json"))
    return {
        "comparison": comparison,
        "curves": CurvesFile.model_validate(load_json(EXAMPLES / "curves.json")),
        "audit": ResamplingAudit.model_validate(load_json(EXAMPLES / "resampling_audit.json")),
        "fold_scores": _fold_scores(),
        "stats": _compute(_fold_scores()),
    }


def test_cliques_group_ranks_within_cd() -> None:
    assert figures.cliques([1.0, 1.5, 2.0, 4.0, 4.2], cd=1.1) == [(0, 2), (3, 4)]
    assert figures.cliques([1.0, 3.0, 5.0], cd=1.0) == []
    assert figures.cliques([1.0, 1.2, 1.4], cd=2.0) == [(0, 2)]


def test_fraud_rate_from_counts(example_artifacts: dict[str, object]) -> None:
    record = example_artifacts["comparison"].records[0]  # type: ignore[attr-defined]
    m = record.test_at_0_5
    assert figures.fraud_rate(m) == pytest.approx((m.tp + m.fn) / (m.tp + m.fp + m.tn + m.fn))


def test_all_figures_are_written(
    example_artifacts: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(figures, "FIGURE_DPI", FAST_DPI)
    written = figures.make_all_figures(
        example_artifacts["comparison"],  # type: ignore[arg-type]
        example_artifacts["curves"],  # type: ignore[arg-type]
        example_artifacts["stats"],  # type: ignore[arg-type]
        example_artifacts["fold_scores"],  # type: ignore[arg-type]
        example_artifacts["audit"],  # type: ignore[arg-type]
        tmp_path,
    )
    expected = {*figures.FIGURE_NAMES, *(f"cd_diagram_{m}.png" for m in [*MODELS, "all"])}
    assert set(written) == expected
    for path in written.values():
        assert path.is_file() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figures_are_300_dpi(example_artifacts: dict[str, object], tmp_path: Path) -> None:
    stats = example_artifacts["stats"]
    path = figures.plot_cd_diagram(stats.friedman[0], tmp_path / "cd.png")  # type: ignore[attr-defined]
    with Image.open(path) as image:
        assert image.info["dpi"] == pytest.approx((300, 300), abs=0.5)


def test_figures_refuse_empty_inputs(tmp_path: Path) -> None:
    empty = ComparisonTable(dataset_source="synthetic", records=[])
    with pytest.raises(ValueError):
        figures.plot_pr_auc_with_ci(empty, 0.95, tmp_path / "x.png")
    with pytest.raises(ValueError):
        figures.plot_fold_boxplots(_fold_scores().iloc[:0], tmp_path / "x.png")
    audit = ResamplingAudit.model_validate(load_json(EXAMPLES / "resampling_audit.json"))
    audit = audit.model_copy(update={"entries": []})
    with pytest.raises(ValueError):
        figures.resampling_counts_table(audit)


def test_figures_dir_for_keeps_sample_figures_apart() -> None:
    assert figures.figures_dir_for("raw") == resolve("artifacts/figures")
    assert figures.figures_dir_for("sample") == resolve("artifacts/figures/sample")


# --- scripts ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real quick cell on the sample, cloned (with shifted scores) for the other techniques
    of logistic regression, so the scripts see genuine cell records and fold scores."""
    root = tmp_path_factory.mktemp("stats_run")
    results = root / "results"
    run_experiments(
        "sample", True, results, root / "models", ["logreg"], ["baseline"],
        cfgs=Configs.load(), log=lambda _: None,
    )  # fmt: skip
    base = load_cell_record(results / CELLS_DIR / f"{cell_id('logreg', 'baseline')}.json")
    for i, technique in enumerate(TECHNIQUES[1:], start=1):
        folds = [
            FoldScore(**{**f.model_dump(), "pr_auc": min(1.0, f.pr_auc + 0.01 * i)})
            for f in base.fold_scores
        ]
        record = base.model_copy(update={"technique": technique, "fold_scores": folds})
        save_json(record.model_dump(mode="json"), results / CELLS_DIR / f"logreg__{technique}.json")
        upsert_fold_scores(fold_frame(record), results / FOLD_SCORES_FILE)
    return results


def test_source_fold_scores_filters_by_source(sample_results: Path) -> None:
    scores = source_fold_scores(sample_results, "sample")
    assert set(scores["technique"]) == set(TECHNIQUES)
    with pytest.raises(ValueError, match="raw"):
        source_fold_scores(sample_results, "raw")


def test_scripts_write_stats_findings_and_figures(
    sample_results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    shutil.copytree(sample_results, results)
    out = results / SAMPLE_EVAL_DIR
    out.mkdir()
    for name in (COMPARISON_FILE, CURVES_FILE):  # example test results stand in for evaluate_test
        shutil.copy2(EXAMPLES / name, out / name)
    figures_dir = tmp_path / "figures"
    run_stats, make_figures = _load_script("run_stats"), _load_script("make_figures")

    args = ["--source", "sample", "--results-dir", str(results), "--figures-dir", str(figures_dir)]
    assert run_stats.main(args) == 0
    stats = StatsResult.model_validate(load_json(out / "stats.json"))
    assert stats.dataset_source == "synthetic" and not stats.is_example
    assert [f.scope for f in stats.friedman] == ["logreg"]
    assert all(t.wins == t.n_pairs for t in stats.pairwise)  # every clone is shifted upwards

    monkeypatch.setattr(figures, "FIGURE_DPI", FAST_DPI)
    assert make_figures.main([*args, "--audit", str(EXAMPLES / "resampling_audit.json")]) == 0
    assert (figures_dir / "cd_diagram_logreg.png").is_file()
    assert (figures_dir / "resampling_counts.png").is_file()

    assert run_stats.main(args) == 0  # second run links the figures
    findings = (out / "findings.md").read_text(encoding="utf-8")
    assert "synthetic development sample" in findings and "cd_diagram_logreg.png" in findings
    assert findings.rstrip().endswith(st.INTERPRETATION_HEADING)


def test_scripts_report_missing_inputs(tmp_path: Path) -> None:
    run_stats, make_figures = _load_script("run_stats"), _load_script("make_figures")
    args = ["--source", "sample", "--results-dir", str(tmp_path)]
    assert run_stats.main(args) == 1
    assert make_figures.main(args) == 1
