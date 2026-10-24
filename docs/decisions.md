# Architectural and scientific decisions

Changes to any row require team agreement; add a new row rather than editing history.

| date | decision | reason | alternatives |
|------|----------|--------|--------------|
| 2026-10-24 | All resampling runs inside `imblearn.pipeline.Pipeline`, only during `fit`, after the train/test split | Resampling before splitting or outside CV folds leaks synthetic/duplicated minority points into validation data and inflates scores | Resample the full dataset up front (leaky); manual per-fold resampling (error-prone) |
| 2026-10-24 | Primary metric is average precision (PR-AUC) | With ~0.17% positives, accuracy and ROC-AUC are dominated by the majority class; PR-AUC reflects performance on the fraud class | Accuracy; ROC-AUC; F1 at a fixed threshold |
| 2026-10-24 | 70/30 stratified train/test split (`test_size: 0.30`, seed from config) | Keeps enough fraud cases (~150) in the test set for stable estimates while preserving class ratio | Time-based split; 80/20 split; nested CV only |
| 2026-10-24 | Raw `Time` excluded from model features | It is seconds since the first transaction in a 2-day window, so it encodes collection order rather than a generalisable signal | Keep raw `Time`; derive hour-of-day (optional side test only, `use_hour_feature`) |
| 2026-10-24 | Outliers are kept | Fraud cases are often extreme values; removing outliers would remove signal and change the class distribution | IQR / z-score outlier removal |
| 2026-10-24 | `RobustScaler` for scaling, fitted inside the pipeline | `Amount` is heavy-tailed; median/IQR scaling is less sensitive to extreme values than mean/std | `StandardScaler`; `MinMaxScaler`; no scaling |
| 2026-10-24 | Shared ratio grid and equal tuning budget for every technique | A fair comparison requires that no technique gets more search effort or a better-tuned ratio range | Per-technique grids; untuned defaults |
| 2026-10-24 | Test set evaluated exactly once, by `scripts/evaluate_test.py` | Any repeated use of the test set for tuning, thresholds or model choice biases the final estimate | Hold-out validation reused as test; repeated test peeking |
| 2026-10-24 | Models: logistic regression, random forest, LightGBM | Covers a linear model, a bagged tree ensemble and a boosted tree ensemble, which respond differently to resampling and class weights | Add SVM / neural networks (cost, limited added insight); single model only |
