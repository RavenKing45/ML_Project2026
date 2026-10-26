"""The Streamlit presentation app: five pages that walk through the study in demo order.

It only READS saved results and never trains a model: artifacts/metrics/*.json, the figures
and the saved pipelines in artifacts/models/. With --sample, or while the real
artifacts/metrics/test_results.json does not exist yet, it reads artifacts/sample/ instead.

Run it: streamlit run app.py            (real results)
        streamlit run app.py -- --sample (results of the fake sample)
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from smote_from_scratch import SMOTE as ScratchSMOTE

from fraudstudy.data import SEED
from fraudstudy.evaluation import ALPHA, LABELS
from fraudstudy.experiments import MODELS, TECHNIQUES, artifacts_dir

# ---- SETTINGS ------------------------------------------------------------------------------------
PAGES = ["Overview", "SMOTE demo", "Results", "Try a transaction", "Method & limitations"]
PAYMENTS_PER_STORY = 100_000  # "out of 100,000 payments ..." on the Results page
TOP_CONTRIBUTIONS = 6  # bars in the "Why?" chart
TEAM = "Shaurya Agarwal, Pragna Jyoti Mandal, Utkarsh Gupta, Ayush, Aditi."  # as in README.md
# --------------------------------------------------------------------------------------------------

RESEARCH_QUESTION = (
    "How do data-level resampling (random oversampling, random undersampling, SMOTE, SMOTE + "
    "undersampling) and algorithm-level class weighting compare against an unmodified baseline "
    "for highly imbalanced credit-card fraud detection?"
)
TECHNIQUE_NOTES = {
    "baseline": ("trains on the data as it is", "nothing invented, nothing thrown away",
                 "the model sees very few frauds"),
    "ros": ("copies fraud rows at random until frauds = r x legit", "keeps every legit row",
            "exact copies invite overfitting"),
    "rus": ("drops legit rows at random until frauds = r x legit", "small, fast training set",
            "throws away most of the legit information"),
    "smote": ("creates new frauds on lines between a fraud and a nearby fraud",
              "new points instead of copies", "can put fake frauds in legit territory"),
    "smote_rus": ("SMOTE up to a small ratio, then undersampling", "milder than either alone",
                  "two ratios to tune"),
    "class_weight": ("keeps the rows, makes each fraud count more in the loss", "no data changed",
                     "only works for models that accept weights"),
}


@st.cache_data
def load_json(path: str) -> dict | None:
    """The parsed JSON file, or None if it does not exist yet."""
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


@st.cache_resource
def load_model(path: str) -> object:
    """A saved pipeline, loaded once and then shared by every page and every visitor."""
    return joblib.load(path)


def counts_of_all_rows() -> tuple[int, int]:
    """(payments, frauds): from eda_summary.json (written by notebooks/01_eda.ipynb), else
    training counts + test confusion counts, i.e. the rows left after removing duplicates."""
    if EDA:
        return int(EDA["rows"]), int(EDA["frauds"])
    cell = next(iter(RESULTS.values()))
    cell_file = load_json(str(METRICS / "cells" / f"{cell['model']}__{cell['technique']}.json"))
    before, test = cell_file["train_counts_before"], cell["test"]
    frauds = before["fraud"] + test["tp"] + test["fn"]
    return before["legit"] + test["tn"] + test["fp"] + frauds, frauds


def page_overview() -> None:
    """Research question, headline numbers, class counts and why accuracy is misleading."""
    st.title("Class imbalance in credit-card fraud detection")
    st.markdown(f"**Research question.** {RESEARCH_QUESTION}")
    payments, frauds = counts_of_all_rows()
    rate = frauds / payments
    one_in = float(f"{1 / rate:.1g}")  # rounded to one significant figure: 598 -> 600
    left, middle, right = st.columns(3)
    left.metric("Payments", f"{payments:,}")
    middle.metric("Frauds", f"{frauds:,}")
    right.metric("Fraud rate", f"{rate:.3%} = about 1 in {one_in:,.0f}")
    if EDA is None:
        st.caption("eda_summary.json not found (run notebooks/01_eda.ipynb): counts are rows "
                   "after removing duplicates.")

    log_scale = st.checkbox("log scale")
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(["legit", "fraud"], [payments - frauds, frauds], color=["grey", "red"])
    ax.set(xscale="log" if log_scale else "linear", xlabel="Payments")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Why accuracy lies")
    default = RESULTS[PRESETS["default_cell"]]["test"] if PRESETS else None
    left, right = st.columns(2)
    left.markdown('**A model that says "legit" every time**')
    left.metric("Accuracy", f"{(payments - frauds) / payments:.2%}")
    left.metric("Recall (frauds caught)", "0%")
    if default:
        right.markdown(f"**Our default model ({PRESETS['default_cell']}), test set**")
        right.metric("Accuracy", f"{default['accuracy']:.2%}")
        right.metric("Precision", f"{default['precision']:.1%}")
        right.metric("Recall (frauds caught)", f"{default['recall']:.1%}")
    st.write("Both look accurate; only one catches fraud. That is why we compare on PR-AUC.")


def toy_data() -> tuple[np.ndarray, np.ndarray]:
    """300 legit points around (0, 0) and two separate clusters of 10 frauds, always the same."""
    rng = np.random.default_rng(SEED)
    legit = rng.normal(0, 1.3, (300, 2))
    fraud = np.vstack([rng.normal(-3.2, 0.4, (10, 2)), rng.normal(3.2, 0.4, (10, 2))])
    return np.vstack([legit, fraud]), np.array([0] * 300 + [1] * 20)


def page_smote() -> None:
    """Our from-scratch SMOTE on a 2-D toy dataset, with k and the seed as controls."""
    st.title("SMOTE demo")
    st.write("SMOTE picks a fraud, picks one of its k nearest frauds, and puts a new fraud at a "
             "random point on the line between them: x_new = x + gap x (x_neighbour - x).")
    k = st.slider("k (neighbours to pick from)", 1, 10, 5)
    seed = int(st.number_input("seed", value=SEED, step=1))
    show_lines = st.checkbox("show parent-neighbour lines")
    X, y = toy_data()
    smote = ScratchSMOTE(k_neighbors=k, random_state=seed)
    X_new, _ = smote.fit_resample(X, y)
    frauds, synthetic = X[y == 1], X_new[len(X):]

    fig, ax = plt.subplots(figsize=(7, 5))
    if show_lines:
        for parent, neighbour, _ in smote.steps_:
            ends = frauds[[parent, neighbour]]
            ax.plot(ends[:, 0], ends[:, 1], color="orange", linewidth=0.5, alpha=0.4)
    ax.scatter(*X[y == 0].T, c="grey", marker="o", s=15, alpha=0.6, label="legit")
    ax.scatter(*synthetic.T, c="orange", marker="D", s=18, label="synthetic fraud")
    ax.scatter(*frauds.T, c="red", marker="^", s=50, label="real fraud")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.write(f"{len(synthetic)} synthetic frauds. Each cluster has 10 frauds, so with k = 10 "
             "every fraud has a neighbour in the OTHER cluster: synthetic points then appear "
             "between the two clusters, in legit territory.")


def significance(model: str) -> dict:
    """{technique: "better than baseline" / "no significant difference" / "worse"} from stats."""
    tests = STATS["models"].get(model, {}).get("wilcoxon_vs_baseline", []) if STATS else []
    words = {"baseline": "(reference)"}
    for test in tests:
        if not test["p_value_holm"] < ALPHA:
            words[test["technique"]] = "no significant difference"
        else:
            better = test["median_difference"] > 0
            words[test["technique"]] = "better than baseline" if better else "worse"
    return words


def page_results() -> None:
    """Test-set table for one model, the saved figures and the threshold story."""
    st.title("Results on the test set")
    models = [m for m in MODELS if any(r["model"] == m for r in RESULTS.values())]
    model = st.selectbox("Model", models)
    words = significance(model)
    rows = []
    for technique in TECHNIQUES:
        result = RESULTS.get(f"{model}__{technique}")
        if result:
            test, (low, high) = result["test"], result["pr_auc_ci"]
            rows.append({"technique": LABELS[technique], "PR-AUC": test["pr_auc"],
                         "95% CI": f"[{low:.3f}, {high:.3f}]", "ROC-AUC": test["roc_auc"],
                         "precision": test["precision"], "recall": test["recall"],
                         "F1": test["f1"], "F2": test["f2"], "FPR": test["fpr"],
                         "threshold": result["threshold_f1"],
                         "vs baseline (Holm p < 0.05)": words.get(technique, "not tested")})
    table = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
    st.dataframe(table.style.format(precision=3), hide_index=True)
    st.caption("Metrics at each cell's F1 threshold, chosen on out-of-fold TRAINING scores. "
               "Significance: Wilcoxon on the CV fold scores, Holm-adjusted.")

    for name in ["pr_curves.png", "pr_auc_ci.png", "training_counts.png"]:
        if (FIGURES / name).is_file():
            st.image(str(FIGURES / name))

    st.subheader("Moving the threshold")
    technique = st.selectbox("Cell", [t for t in TECHNIQUES if f"{model}__{t}" in RESULTS],
                             format_func=lambda t: f"{model} + {LABELS[t]}")
    curve = RESULTS[f"{model}__{technique}"]["pr_curve"]
    point = st.slider("Point on the PR curve (left = strict, right = flag more)", 0,
                      len(curve["recall"]) - 1, len(curve["recall"]) // 2)
    precision, recall = curve["precision"][point], curve["recall"][point]
    left, right = st.columns(2)
    left.metric("Precision", f"{precision:.1%}")
    right.metric("Recall", f"{recall:.1%}")
    payments, frauds = counts_of_all_rows()
    frauds_in_story = PAYMENTS_PER_STORY * frauds / payments
    caught = recall * frauds_in_story
    # Of the flagged payments a share `precision` is fraud, so innocents = caught x (1/P - 1).
    innocent = caught * (1 / precision - 1) if precision > 0 else float("nan")
    # The curve comes from the test set, whose fraud rate differs a little from the whole
    # dataset's, so at "flag everything" the formula can overshoot the number of innocents.
    innocent = min(innocent, PAYMENTS_PER_STORY - frauds_in_story)
    st.info(f"Out of {PAYMENTS_PER_STORY:,} payments: about {caught:,.0f} frauds caught "
            f"(of {frauds_in_story:,.0f}), {innocent:,.0f} innocent customers flagged.")


def page_try() -> None:
    """Score one saved test payment with a LightGBM model and explain the score."""
    st.title("Try a transaction")
    if not PRESETS or not PRESETS["presets"]:
        st.warning("presets.json not found; run python -m fraudstudy.evaluation --test first.")
        return
    columns = st.columns(len(PRESETS["presets"]))
    for column, preset in zip(columns, PRESETS["presets"], strict=True):
        if column.button(preset["label"]):
            st.session_state["preset"] = preset["id"]
    chosen = st.session_state.get("preset", PRESETS["presets"][0]["id"])
    preset = next(p for p in PRESETS["presets"] if p["id"] == chosen)
    cells = [f"lightgbm__{t}" for t in TECHNIQUES if f"lightgbm__{t}" in RESULTS]
    cell = st.selectbox("LightGBM model", cells, format_func=lambda c: LABELS[c.split("__")[1]])

    pipeline = load_model(str(MODELS_DIR / f"{cell}.joblib"))
    row = pd.DataFrame([preset["features"]])
    score = float(pipeline.predict_proba(row)[:, 1][0])  # predict_proba skips the samplers
    threshold = RESULTS[cell]["threshold_f1"]
    st.write(f"**{preset['label']}**, amount {preset['features']['Amount']:,.2f}")
    left, middle, right = st.columns(3)
    left.metric("Risk score", f"{score:.3f}")
    middle.metric("Threshold", f"{threshold:.3f}")
    right.metric("Model says", "FRAUD" if score >= threshold else "LEGIT")
    st.write(f"True label: **{'FRAUD' if preset['true_class'] == 1 else 'LEGIT'}**")

    st.subheader("Why?")
    # Only the "preprocess" step runs here: the sampler steps are for training and never
    # touch a payment that is being scored.
    X_ready = pipeline.named_steps["preprocess"].transform(row)
    contributions = pipeline.named_steps["model"].predict(X_ready, pred_contrib=True)[0][:-1]
    top = pd.Series(contributions, index=X_ready.columns).sort_values(key=abs)
    top = top.tail(TOP_CONTRIBUTIONS)  # the largest pushes, up or down
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(top.index, top.values, color=["red" if v > 0 else "green" for v in top.values])
    ax.set(xlabel="Contribution to the score (log-odds; > 0 pushes towards fraud)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("V1-V28 are anonymised; their real meaning is unknown.")


def page_method() -> None:
    """The leakage-free pipeline, the six techniques, limitations and the team."""
    st.title("Method & limitations")
    st.subheader("The leakage-free pipeline")
    st.markdown(
        "1. Remove duplicate rows, then split 70/30 (stratified); the 30% test set is locked.\n"
        "2. Cross-validation on the training data only. For each fold:\n"
        "   1. preprocess (median imputer, log1p(Amount), RobustScaler), fitted on the fold;\n"
        "   2. resample the training fold only;\n"
        "   3. train the model;\n"
        "   4. score the untouched validation fold.\n"
        "3. Choose the settings (same tuning budget for every technique, best PR-AUC) and the "
        "F1 threshold from out-of-fold training scores.\n"
        "4. Open the test set once and score every cell."
    )
    st.subheader("The six techniques")
    for technique, (what, pro, con) in TECHNIQUE_NOTES.items():
        st.markdown(f"**{LABELS[technique]}**: {what}. Pro: {pro}. Con: {con}.")
    st.subheader("Limitations")
    st.markdown(
        "- Public, anonymised data: two days of European card payments from 2013.\n"
        "- V1-V28 are PCA components, so the explanations cannot be read as real-world causes.\n"
        "- SMOTE's new frauds are interpolations, not real frauds.\n"
        "- Offline metrics, no real cost of a missed fraud or a blocked customer.\n"
        "- Fold scores share training rows, so the p-values are approximate.\n"
        "- No concept drift and no fraudsters adapting to the model."
    )
    st.subheader("Team")
    st.write(TEAM)


st.set_page_config(page_title="Fraud detection: class imbalance study", layout="wide")
REAL_RESULTS = artifacts_dir("raw") / "metrics" / "test_results.json"
USE_SAMPLE = "--sample" in sys.argv[1:] or not REAL_RESULTS.is_file()
ROOT = artifacts_dir("sample" if USE_SAMPLE else "raw")
METRICS, FIGURES, MODELS_DIR = ROOT / "metrics", ROOT / "figures", ROOT / "models"
RESULTS = load_json(str(METRICS / "test_results.json"))
if RESULTS is None:
    st.error(f"No results in {METRICS}. Run the experiments and the test evaluation first "
             "(steps 5-7 of the README, or its trial run with --source sample --quick).")
    st.stop()
EDA = load_json(str(METRICS / "eda_summary.json"))
STATS = load_json(str(METRICS / "stats.json"))
PRESETS = load_json(str(METRICS / "presets.json"))

page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption("Research demo on anonymised public data. Not a real fraud system.")
if USE_SAMPLE:
    st.warning("Sample results, not the real dataset")
if page == "Overview":
    page_overview()
elif page == "SMOTE demo":
    page_smote()
elif page == "Results":
    page_results()
elif page == "Try a transaction":
    page_try()
else:
    page_method()
