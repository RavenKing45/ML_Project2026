# Credit Card Fraud Detection --- Imbalanced Classification Experiments

A reproducible machine-learning project exploring how preprocessing,
class-imbalance strategies, and classification algorithms affect
credit-card fraud detection.

> **Current status:** The uploaded notebook implements the data audit,
> chronological split, preprocessing/PCA comparisons, resampling
> visualizations, and small-scale model screening. Expanded training,
> ensemble experiments, final threshold selection, final test
> evaluation, and final model serialization are planned next steps, not
> yet implemented in this notebook version.

## Objectives

-   Understand the severe class imbalance in credit-card transaction
    data.
-   Compare practical preprocessing approaches.
-   Compare imbalance-handling approaches without contaminating
    validation or test data.
-   Benchmark a range of classifiers using metrics suitable for rare
    fraud detection.
-   Consider predictive quality alongside training time and
    computational cost.
-   Establish a defensible path from quick screening to final model
    training.

## Dataset

The notebook expects a CSV with columns similar to the widely used
European credit-card transactions dataset:

-   `Time`: elapsed seconds from the first recorded transaction.
-   `V1`--`V28`: anonymized, PCA-transformed transaction features.
-   `Amount`: transaction amount.
-   `Class`: target label (`0` = legitimate, `1` = fraud).

The commonly used public version has approximately 284,807 transactions
and 492 fraud cases. **Verify these figures against your local file**;
the notebook prints the actual dimensions and class distribution. Do not
commit the dataset to this repository unless you have confirmed that
redistribution is permitted.

## Current notebook workflow

### 1. Environment and configuration

The notebook imports the required libraries, checks optional model/GPU
availability, and defines experiment settings in one configuration cell.

Key settings include:

-   `RANDOM_STATE = 42`
-   `QUICK_SAMPLE_SIZE = 30_000`
-   `EXPANDED_SAMPLE_SIZE = 100_000` (configured for future work; the
    expanded phase is not implemented in the current notebook)
-   `CV_FOLDS = 5`
-   `MAX_FINALISTS = 5`
-   `RUN_EXPENSIVE_MODELS`
-   `RUN_PCA_EXPERIMENTS`
-   `RUN_NEURAL_NETWORK`

The GPU setting only enables GPU execution for supported libraries and
models. Many scikit-learn estimators run on the CPU. The notebook should
remain usable on CPU if CUDA is unavailable, provided dependencies are
installed and optional-model errors are handled.

### 2. Data audit and visual exploration

The notebook examines:

-   Dataset shape, data types, missing values, and summary statistics.
-   Class counts and fraud percentage.
-   Exact duplicate rows.
-   Time and amount distributions.
-   Feature distributions and potentially discriminative features.
-   Correlations with the target.

Duplicates and outliers should be investigated rather than automatically
removed: in fraud detection, unusual observations may be informative.

### 3. Chronological train/validation/test split

The notebook sorts records by `Time` and splits them chronologically:

-   First 70%: training pool.
-   Next 15%: validation set.
-   Final 15%: held-out test set.

This split is intended to simulate evaluating on later transactions. The
notebook checks that time ranges do not overlap across adjacent splits.

Because fraud is rare and the dataset covers a limited period, inspect
the number of fraud cases in each split. A chronological split can
produce different class rates across periods. If it leaves too few fraud
cases for stable evaluation, document that limitation and consider a
justified alternative.

**Keep the test set untouched until the final pipeline and decision
threshold have been selected.** Do not use test metrics to select
models, balancing strategies, or thresholds.

### 4. Quick screening sample

The current notebook uses up to 30,000 rows from the training pool for
initial experiments. Its sampling helper is designed to include all
fraud examples from the training pool and sample legitimate transactions
to reach the target size.

Consequently, the screening sample may have a higher fraud proportion
than the original training pool. This is useful for economical
experimentation, but the sampling design must be remembered when
interpreting results.

The validation set remains the full validation split, and the test set
is not used in screening.

### 5. Preprocessing and PCA

The notebook compares minimal preprocessing, standard scaling, and
robust scaling with Logistic Regression as a screening model.

It also optionally compares PCA component counts. Since `V1`--`V28` are
already PCA-transformed in the common dataset, applying PCA again should
only be retained if experiments demonstrate a useful benefit.

Tree-based estimators generally need less feature scaling than linear or
distance-based estimators. Transformations of `Amount` and `Time` should
be assessed empirically and fitted using training data only.

### 6. Imbalance-handling experiments

The current notebook visualizes and screens several approaches,
including:

-   No resampling.
-   Class weighting.
-   Random undersampling.
-   Random oversampling.
-   SMOTE.
-   Borderline-SMOTE.
-   Balanced Random Forest, where available.

Resampling should be applied only to training data, never to validation
or test data. A balanced training set is not automatically better than
the original distribution; compare the effect on fraud detection, false
alarms, and runtime.

SMOTE creates synthetic minority examples. They are not newly observed
real transactions, and the method must be judged by held-out performance
rather than by how balanced the resampled data looks.

### 7. Model screening

The current model configuration includes a range of candidate
estimators, such as:

-   DummyClassifier baseline.
-   Logistic Regression.
-   Linear SVM and RBF SVM.
-   K-Nearest Neighbors.
-   Decision Tree.
-   Random Forest and Extra Trees.
-   Balanced Random Forest.
-   Histogram-based gradient boosting.
-   XGBoost.
-   LightGBM.
-   CatBoost.

Optional models may fail or be skipped if their packages or hardware
backends are unavailable. Check the notebook output and result registry;
a configured model is not proof that its experiment completed
successfully.

The current screening compares baseline, class-weighted, SMOTE, and
selected additional balancing combinations. It records metrics and
training/prediction times where available.

## Evaluation metrics

Accuracy alone is unsuitable as the main measure for a highly imbalanced
fraud dataset. A model can achieve high accuracy while missing most
fraud.

The notebook emphasizes:

-   **Average precision / PR-AUC:** summarizes the precision-recall
    trade-off across score thresholds.
-   **Fraud recall:** fraction of actual fraud cases detected.
-   **Fraud precision:** fraction of flagged transactions that are
    fraud.
-   **F1-score:** harmonic mean of precision and recall.
-   **F2-score:** weights recall more heavily than precision.
-   **ROC-AUC:** complementary threshold-independent ranking metric.
-   **Confusion matrix:** true positives, false positives, true
    negatives, and false negatives.
-   **Training and prediction time:** practical computational cost.

Compare models on the same validation set and interpret metrics
together. Higher recall can mean more false alarms; higher precision can
mean more missed fraud.

## What is implemented and what remains?

### Implemented in the uploaded notebook

-   Environment setup and configuration.
-   Dataset loading and audit.
-   Class-distribution and feature visualizations.
-   Chronological 70/15/15 split.
-   Quick training sample construction.
-   Preprocessing comparison.
-   Optional PCA comparison.
-   Resampling impact and SMOTE visualizations.
-   Small-scale screening across multiple model families and imbalance
    strategies.
-   Screening result tables/plots and saving interim screening results
    to the configured results directory.

### Planned next steps --- not yet implemented in this notebook version

1.  Confirm the leading model/balancing combinations on a larger
    training sample.
2.  Implement the configured expanded-training stage.
3.  Compare a small number of justified voting or stacking ensembles,
    using leakage-safe procedures.
4.  Tune the strongest candidates with a controlled budget.
5.  Select a decision threshold using validation data only.
6.  Retrain the selected pipeline on the complete designated training
    pool.
7.  Evaluate once on the untouched test set.
8.  Save the final preprocessing-plus-model pipeline and metadata.
9.  Document final test results, limitations, and reproducibility
    details.

Do not describe the project as having a finalized best model until these
steps have been run successfully and their results recorded.

## Suggested repository structure

``` text
ML_Project2026/
├── .gitignore
├── README.md
├── requirements.txt
├── Dataset/                         # local only; ignored by Git
│   └── creditcard.csv
├── notebooks/
│   └── fraud_detection_experiments.ipynb
├── models/                           # final artifacts, if appropriate
└── results/                          # generated tables and plots
```

Adjust paths to match the actual repository. Do not create empty folders
merely to match this example.

## Setup

Use a virtual environment in the project root.

### Windows PowerShell

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the appropriate activation
instructions for your environment or run the environment's Python
executable directly.

### Jupyter kernel

``` powershell
python -m ipykernel install --user --name fraud-detection --display-name "Python (Fraud Detection)"
```

Select the `Python (Fraud Detection)` kernel in Jupyter/VS Code.

### Dataset path

The notebook currently expects:

``` text
../Dataset/creditcard.csv
```

This is relative to the notebook's working directory. If your notebook
is in `notebooks/` and the dataset is in the root `Dataset/` folder,
that path is appropriate. Otherwise update `DATA_PATH` in the
configuration cell.

Run cells in order from a fresh kernel. Some cells depend on variables
and functions defined earlier.

## Running the notebook

1.  Place the dataset at the configured path.
2.  Install the notebook dependencies.
3.  Open `notebooks/fraud_detection_experiments.ipynb`.
4.  Verify the configuration and GPU availability output.
5.  Run the notebook from the first cell through the screening summary.
6.  Inspect any failed or skipped experiments as well as the
    leaderboard.
7.  Review generated screening results under the configured `results/`
    directory.

Initial screening is designed to use a fraction of the training pool for
speed. The `EXPANDED_SAMPLE_SIZE` value is only a configuration setting
in this version; it does not mean the expanded experiments are already
implemented.

## GPU notes

The project targets local execution and can use an NVIDIA GPU for
supported libraries. GPU support depends on the installed build and
driver/runtime configuration.

-   PyTorch models can use CUDA if the installed PyTorch build supports
    it.
-   XGBoost GPU usage depends on the installed XGBoost build.
-   CatBoost GPU support depends on a compatible installation and device
    configuration.
-   Many scikit-learn models are CPU-based.
-   A GPU being detected does not mean every model uses it.

If a GPU-specific model fails, verify its backend and use a supported
CPU fallback where feasible. Do not install a different CUDA stack
blindly; first check the installed versions and error message.

## Reproducibility and experiment interpretation

-   A fixed seed improves reproducibility, but some GPU and parallel
    operations may remain nondeterministic.
-   Record library versions and actual split sizes for a final report.
-   Compare balancing methods on the same validation set.
-   Resample training data only.
-   Do not rank failed experiments as if they were valid results.
-   Do not compare scores from different validation populations without
    explaining the difference.
-   Do not treat screening results as final test performance.
-   Reassess shortlisted candidates on a larger training sample before
    selecting the final pipeline.

## Data and model artifacts

The dataset is intentionally excluded from Git. Generated results and
model files may also be excluded to keep the repository lightweight. If
a small artifact needs to be versioned, review its size,
reproducibility, and intended use first.

A final model should be saved as a complete fitted pipeline, including
any transformations needed at inference time. Metadata should record the
feature order, training configuration, chosen threshold, package
versions, and validation/test metrics.

## Limitations

-   The dataset is historical and covers a limited time period; results
    may not generalize to current fraud patterns.
-   Anonymized PCA features limit domain interpretation.
-   A chronological split can have varying fraud rates across periods.
-   Synthetic oversampling does not guarantee more realistic minority
    examples.
-   The best decision threshold depends on operational costs and review
    capacity, which are not specified here.
-   Validation performance is an estimate; it is not a guarantee of
    production performance.
-   This is a research/learning project, not a production-ready bank
    fraud prevention system.

## Project status

**Current milestone:** data exploration and small-scale screening.

**Next milestone:** expanded finalist experiments, leakage-safe ensemble
comparison, threshold selection, final training, held-out test
evaluation, and model export.

The final model and its reported performance should be documented only
after those experiments have actually completed.
