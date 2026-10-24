"""Tests for fraudstudy.eda and scripts/run_eda.py (synthetic sample only)."""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fraudstudy import eda
from fraudstudy import synthetic as syn
from fraudstudy.config import get_dataset_config
from fraudstudy.contracts import HOUR_ASSUMPTION, ClassCounts, EdaSummary
from fraudstudy.data import deduplicate, load_dataset
from fraudstudy.errors import DataValidationError
from fraudstudy.io import load_json
from fraudstudy.paths import PROJECT_ROOT

MIN_FIGURE_BYTES = 10 * 1024

_spec = importlib.util.spec_from_file_location("run_eda", PROJECT_ROOT / "scripts" / "run_eda.py")
assert _spec is not None and _spec.loader is not None
_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_script)


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return get_dataset_config()


@pytest.fixture(scope="module")
def frames(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_dataset("sample", cfg=cfg)
    deduped, _ = deduplicate(raw, target=cfg["target"])
    return raw, deduped


@pytest.fixture(scope="module")
def summary(frames: tuple[pd.DataFrame, pd.DataFrame], cfg: dict[str, Any]) -> EdaSummary:
    raw, deduped = frames
    return eda.compute_eda_summary(
        raw, deduped, cfg, dataset_source="synthetic", dataset_sha256="0" * 64
    )


# --- summary on the synthetic sample -------------------------------------------------------------


def test_summary_round_trips_through_contract(summary: EdaSummary) -> None:
    dumped = summary.model_dump(mode="json")
    assert EdaSummary.model_validate(dumped) == summary


def test_counts_match_generator(summary: EdaSummary) -> None:
    n_fraud = round(6000 * 0.015)
    assert summary.n_rows == 6000 and summary.n_columns == len(syn.COLUMNS)
    assert summary.class_counts == ClassCounts(legit=6000 - n_fraud, fraud=n_fraud)
    assert summary.fraud_rate == pytest.approx(n_fraud / 6000)
    assert summary.n_duplicate_rows == syn.DUPLICATE_ROWS
    assert summary.n_duplicate_frauds == syn.DUPLICATE_FRAUDS
    assert summary.class_counts_dedup == ClassCounts(
        legit=6000 - n_fraud - (syn.DUPLICATE_ROWS - syn.DUPLICATE_FRAUDS),
        fraud=n_fraud - syn.DUPLICATE_FRAUDS,
    )
    assert summary.imbalance_ratio == f"1 : {round((6000 - n_fraud) / n_fraud)}"
    assert set(summary.missing_values) == set(syn.COLUMNS)
    assert sum(summary.missing_values.values()) == 0
    assert summary.training_counts == []


def test_histograms_have_20_bins_and_cover_every_row(summary: EdaSummary) -> None:
    dedup = summary.class_counts_dedup
    assert dedup is not None
    assert len(summary.feature_histograms) == eda.N_TOP_FEATURES + 1
    assert summary.feature_histograms[-1].feature == eda.LOG_AMOUNT
    for hist in [*summary.feature_histograms, summary.amount_histogram]:
        assert len(hist.bin_edges) == eda.N_BINS + 1
        assert len(hist.legit_counts) == len(hist.fraud_counts) == eda.N_BINS
        assert np.all(np.diff(hist.bin_edges) > 0)
        assert sum(hist.legit_counts) == dedup.legit
        assert sum(hist.fraud_counts) == dedup.fraud
    for hist in summary.feature_histograms:
        assert sum(hist.legit_fraction) == pytest.approx(1.0)
        assert sum(hist.fraud_fraction) == pytest.approx(1.0)


def test_top_features_are_the_shifted_ones(summary: EdaSummary) -> None:
    shifted = set().union(*(shifts for shifts in syn.CLUSTER_SHIFTS))
    top = [h.feature for h in summary.feature_histograms[: eda.N_TOP_FEATURES]]
    assert set(top) <= shifted
    assert top == [e.feature for e in summary.effect_sizes if e.feature != "Amount"][:6]
    d = [abs(e.cohens_d) for e in summary.effect_sizes]
    assert d == sorted(d, reverse=True)
    assert len(summary.effect_sizes) == 29  # V1..V28 + Amount
    ks = [f.ks_statistic for f in summary.feature_separation]
    assert ks == sorted(ks, reverse=True)


def test_amount_and_time(summary: EdaSummary, frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, deduped = frames
    stats = summary.amount_summary
    assert stats is not None
    assert stats.raw.max == pytest.approx(deduped["Amount"].max())
    assert stats.log1p.max == pytest.approx(np.log1p(deduped["Amount"].max()))
    assert stats.raw.skewness > stats.log1p.skewness  # log1p reduces right skew
    span = (deduped["Time"].max() - deduped["Time"].min()) / 3600
    assert summary.time_span_hours == pytest.approx(span)
    assert sum(summary.time_histogram.fraud_counts) == summary.class_counts_dedup.fraud


def test_hour_of_day(summary: EdaSummary) -> None:
    hours = summary.hour_of_day
    assert hours is not None
    assert hours.hour_assumption == HOUR_ASSUMPTION
    assert hours.hour_assumption == (
        "assumes Time=0 is midnight; not documented by the data provider"
    )
    assert len(hours.fraud_counts) == 24
    assert sum(hours.fraud_counts) == summary.class_counts_dedup.fraud
    assert sum(hours.legit_counts) == summary.class_counts_dedup.legit


# --- helpers ------------------------------------------------------------------------------------


def test_cohens_d_matches_formula() -> None:
    fraud, legit = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 3.0])
    pooled = np.sqrt((2 * np.var(fraud, ddof=1) + 3 * np.var(legit, ddof=1)) / 5)
    assert eda.cohens_d(fraud, legit) == pytest.approx((2.0 - 1.5) / pooled)
    assert eda.cohens_d(np.ones(3), np.ones(4)) == 0.0


def test_imbalance_ratio() -> None:
    assert eda.imbalance_ratio(ClassCounts(legit=284_315, fraud=492)) == "1 : 578"
    with pytest.raises(DataValidationError):
        eda.imbalance_ratio(ClassCounts(legit=10, fraud=0))


def test_hour_of_day_wraps_after_24_hours() -> None:
    time = pd.Series([0.0, 3599.0, 3600.0, 86_400.0 + 7200.0])
    hours = eda.hour_of_day(time, pd.Series([1, 1, 0, 1]))
    assert hours.fraud_counts[0] == 2 and hours.fraud_counts[2] == 1
    assert hours.legit_counts[1] == 1


def test_feature_histogram_clips_tails_into_outer_bins() -> None:
    legit = np.concatenate([np.zeros(998), [1e6, -1e6]])
    hist = eda.feature_histogram("x", legit, np.array([0.5]), quantiles=(0.01, 0.99))
    assert sum(hist.legit_counts) == 1000
    assert hist.bin_edges[-1] < 1e6


def test_compute_rejects_single_class(
    frames: tuple[pd.DataFrame, pd.DataFrame], cfg: dict[str, Any]
) -> None:
    raw, deduped = frames
    legit_only = deduped[deduped["Class"] == 0]
    with pytest.raises(DataValidationError, match="both classes"):
        eda.compute_eda_summary(raw, legit_only, cfg, dataset_source="synthetic", dataset_sha256="")
    with pytest.raises(DataValidationError, match="deduplicated frame second"):
        eda.compute_eda_summary(deduped, raw, cfg, dataset_source="synthetic", dataset_sha256="")


def test_contract_rejects_unsorted_effect_sizes(summary: EdaSummary) -> None:
    data = summary.model_dump(mode="json")
    data["effect_sizes"] = data["effect_sizes"][::-1]
    with pytest.raises(ValidationError, match="effect_sizes"):
        EdaSummary.model_validate(data)


def test_contract_rejects_bad_hour_of_day(summary: EdaSummary) -> None:
    data = summary.model_dump(mode="json")
    data["hour_of_day"]["fraud_counts"] = data["hour_of_day"]["fraud_counts"][:23]
    with pytest.raises(ValidationError, match="24"):
        EdaSummary.model_validate(data)


# --- script and figures -------------------------------------------------------------------------


def test_run_eda_writes_json_and_figures(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = _script.main(
        ["--source", "sample", "--results-dir", str(tmp_path), "--figures-dir", str(tmp_path)]
    )
    assert code == 0
    parsed = EdaSummary.model_validate(load_json(tmp_path / "eda.json"))
    assert parsed.dataset_source == "synthetic" and len(parsed.dataset_sha256) == 64
    for name in eda.FIGURE_NAMES:
        path = tmp_path / name
        assert path.is_file() and path.stat().st_size > MIN_FIGURE_BYTES, name
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    out = capsys.readouterr().out
    assert "duplicates        20 removed (2 fraud)" in out
    assert "imbalance 1 : 66" in out


def test_contract_1_0_eda_file_still_validates(summary: EdaSummary) -> None:
    data = summary.model_dump(mode="json")
    for field in (
        "imbalance_ratio",
        "class_counts_dedup",
        "amount_summary",
        "time_span_hours",
        "effect_sizes",
        "feature_histograms",
        "hour_of_day",
        "training_counts",
    ):
        del data[field]
    parsed = EdaSummary.model_validate(data | {"contract_version": "1.0"})
    assert parsed.effect_sizes == [] and parsed.hour_of_day is None
