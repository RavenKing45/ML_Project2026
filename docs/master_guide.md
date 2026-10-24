# Fraud Detection Imbalance Study — Master Guide

Oct 24, 2026 · @Shaurya

## How to read this guide (and what it fixes)

We are building two things that fit together: a **fair scientific experiment** that compares ways of handling rare fraud cases, and a **web app** that shows the results and lets someone try the trained model. The experiment is the heart of the grade; the app is how we prove and show it.

This guide assumes you know nothing. Every new word is explained the first time it appears. Read Parts 1 to 3 even if you are only doing the frontend, because the instructor can ask anyone about SMOTE and metrics.

The complete, copy-ready prompts for building everything with Claude are in the second tab: Coding-agent prompts.

**Who should read what**

| You are working on | Must read | Skim |
| --- | --- | --- |
| Everyone | Parts 1, 2, 3, 6, 16 | The rest |
| Data and EDA | Part 5 | Parts 8, 13 |
| Models and experiments | Parts 4, 5, 6, 7 | Part 14 |
| Backend | Parts 7, 9, 11 | Part 8 |
| Frontend | Parts 9, 10 | Part 11 |
| Report and slides | Parts 4, 15, 16, 17 | Everything else |

**What was wrong or missing in Gemini's plan.** Gemini's plan is a strong starting point and most of it is kept here. But a careful review found problems that an examiner could catch, or that would stop you from actually using it:

| # | Problem in Gemini's plan | Why it matters | What this guide does instead |
| --- | --- | --- | --- |
| 1 | Every coding-agent prompt in the PDF is cut off mid-sentence (for example "Exact requi…") | You cannot paste them into Claude; the important instructions are missing | The **Coding-agent prompts** tab has every prompt written out in full |
| 2 | It criticises 1:1 SMOTE for making 99.8% of fraud rows fake, but its own hybrid setting (SMOTE to 5% of normal) turns 331 real frauds into about 9,900, so about 97% are fake too | Self-contradiction is an easy viva attack | SMOTE amount is expressed the way Chawla did (100%–500% extra) and treated as a setting we test, not a fixed choice |
| 3 | It turns `Time` into "hour of day" assuming the data starts at midnight | Nobody has published when the data starts; this is a guess presented as fact | Main experiment leaves raw `Time` out; "hour of day" is an optional side test with the assumption stated |
| 4 | No error bars. The test set has only about 142 fraud cases, so one fraud more or less moves recall by 0.7 points | Two methods can look different purely by luck | Repeated cross-validation, bootstrap confidence intervals and a named statistical test (Part 6) |
| 5 | The "champion model" may be picked using the **test** score | Choosing with the test set is a quiet form of cheating | Choose only with cross-validation; the test set is opened once at the very end |
| 6 | Techniques get different target ratios (ROS and SMOTE to 10%, RUS to 50%) and tuning budgets are not equalised | Then we are not comparing techniques fairly; ratio and budget become hidden variables | Every technique gets the **same** ratio choices and the **same** tuning budget |
| 7 | The app shows "fraud probability" from resampled models as if it were a true probability | Resampling inflates these numbers; an examiner who knows this will ask | The app calls it a **risk score**, explains why, and optionally shows a calibrated version |
| 8 | "Why was this flagged?" uses features like V14 that have no human meaning | Claiming a business explanation would be false | Explanations are presented honestly: "V14 was unusually low, and that pushed the score up" |
| 9 | It plans to load all 18 trained models on a free server with 512 MB of memory | Random Forest files are large; the server can crash | Deploy only a small set of models, or use Hugging Face Spaces (more memory) |
| 10 | One squad owns both evaluation and the entire backend | Workload is uneven; that squad becomes the bottleneck | Rebalanced team plan, with early-finishing squads moving to frontend later (Part 12) |
| 11 | Small factual slips: amounts written in dollars, an unsourced "$30B fraud losses" figure, "90 trainings" while ignoring tuning rounds | Small errors make examiners doubt bigger claims | Amounts are left in the dataset's own unit (believed to be euros); unsourced numbers are removed; compute is estimated honestly |
| 12 | Recommends Hugging Face Spaces as a free 16 GB Docker host | Hugging Face now requires a paid plan to create Docker Spaces on personal accounts, so the backup plan would fail | Render free + small models + a local Docker Compose backup (Part 11) |

Everything that Gemini got right (leakage-free pipelines, PR-AUC as main metric, keeping outliers, RobustScaler, log-transforming Amount, deduplication, presets instead of 30 input boxes) is kept and explained more simply.

## Part 1 — The big picture: what we are building

**Yes, we need a backend and a user interface.** But they come second. First we run a research experiment; then we build a small web app that shows the results and lets the instructor try the model.

### The two halves of the project

**Half 1: the research study (about 60% of the effort, most of the marks).** We take a real dataset of card payments, try 5 to 6 different ways of handling the fact that fraud is extremely rare, train 2 to 3 kinds of models with each way, and measure fairly which combination catches fraud best without blocking too many innocent customers. The output is a results table, graphs, a written report and a conclusion.

**Half 2: the demonstration app (about 40% of the effort).** A website where someone can see the problem, see the results, watch an animation of how SMOTE invents new fraud examples, and test the trained model on a transaction or a whole file of transactions.

Think of it like a science fair. The experiment is the actual science. The app is the poster and the working model on the table that lets judges see it for themselves. A beautiful poster with bad science fails; good science with no poster is hard to show.

### What "frontend", "backend" and "API" mean, from zero

Imagine a restaurant.

- **Frontend** = the dining room and the menu. It is what the customer sees and touches. In our project, it is the website in the browser: buttons, charts, forms. We build it with **React**, a popular tool for building interactive web pages.
- **Backend** = the kitchen. Customers never see it, but it does the real work. In our project, it is a Python program running on a server that holds the trained model and does the calculations. We build it with **FastAPI**, a Python tool for making backends quickly.
- **API** (Application Programming Interface) = the waiter and the order slip. The frontend cannot walk into the kitchen; it writes an order in a fixed format ("here are this transaction's 30 numbers, please score it") and the waiter brings back a reply in a fixed format ("risk score 0.94, flagged as fraud, mostly because V14 was very low"). Each kind of order has an address called an **endpoint**, like `/api/v1/predict`.
- **Model file** = the recipe book the kitchen uses. After the experiment, we save the trained model to a file (a `.joblib` file). The backend loads that file when it starts. The backend does **not** retrain anything; it only uses the finished recipe.
- **JSON** = the format of the order slips. It is just text with labels, like `{"risk_score": 0.94}`. Both Python and the browser can read it.

### How the pieces connect

1. We download the dataset (a CSV file, which is a spreadsheet saved as plain text).
2. Python scripts clean it, split it, run the experiment and produce three kinds of output: trained model files, results files (JSON), and figures (PNG images).
3. The backend loads the model files and the results files.
4. The frontend asks the backend for results and predictions, and draws them nicely.
5. Both are put online ("deployed") so the instructor can open a link.

The key design idea: **training happens offline, once, on our laptops or Google Colab. The website only reads the saved results.** This keeps the website fast, cheap and identical every time we demo it.

### What we will hand in at the end

| Deliverable | What it is | Who mainly makes it |
| --- | --- | --- |
| Code repository | All code on GitHub, runnable with a few commands | Everyone |
| Results | Comparison table, ROC and PR curves, confusion matrices, statistics | Models + Evaluation squads |
| Written report | 10–20 page research report | Report squad, everyone reviews |
| Web app | Live dashboard with 6 pages | Backend + Frontend squads |
| Slides | 15–18 slides | Report squad |
| Viva readiness | Everyone can answer the questions in Part 16 | Everyone |

## Part 2 — The problem from zero

**Fraud is so rare (about 1 in 578 payments) that a model which never flags anything is 99.83% accurate and completely useless.** Everything in this project follows from that one fact.

### Classification

A **classifier** is a program that sorts things into groups. A post-office worker sorts letters into "local" and "outside" by reading the pin code. Our classifier sorts card payments into **legitimate** (class 0) and **fraud** (class 1) by reading 30 numbers about each payment.

- **Feature**: one clue the model reads (the amount, or V14).
- **Label / class**: the right answer we already know for past payments (0 or 1).
- **Training**: showing the model many past payments with answers, so it learns a rule.
- **Testing**: checking the rule on payments it has never seen, to see whether it really learned or just memorised.
- **Model**: the learned rule itself.

### Class imbalance

In our dataset there are 284,315 legitimate payments and only 492 frauds. The common class (legitimate) is called the **majority class**. The rare class (fraud) is the **minority class**. When one class vastly outnumbers the other, the data is **imbalanced**.

Why this hurts: most models learn by trying to make as few total mistakes as possible. If 998 of every 1,000 examples are legitimate, the easiest way to make few mistakes is to say "legitimate" almost always. The model barely learns what fraud looks like, because fraud hardly affects its score.

### The accuracy paradox

**Accuracy** = the percentage of all answers that were right. A "model" that always answers "legitimate" is right 284,315 times out of 284,807 = **99.83% accurate**, and catches **zero** frauds. So in this project accuracy is never used to choose anything. We show it once, on purpose, to demonstrate why it lies.

### The confusion matrix: four kinds of answer

Every prediction falls into one of four boxes. "Positive" means "the thing we are hunting" (fraud), not "good".

|  | Model says FRAUD | Model says LEGITIMATE |
| --- | --- | --- |
| **Really fraud** | **TP** (true positive): thief caught | **FN** (false negative): thief got away, bank loses money |
| **Really legitimate** | **FP** (false positive): innocent customer's card blocked | **TN** (true negative): correctly left alone |

### Every metric, in plain words

| Metric | Formula | Plain meaning | Why it matters for fraud |
| --- | --- | --- | --- |
| Recall (also called TPR, true positive rate) | TP ÷ (TP + FN) | Of all real frauds, how many did we catch? | Low recall = money lost to thieves |
| Precision | TP ÷ (TP + FP) | Of everything we flagged, how much was really fraud? | Low precision = many angry innocent customers and overloaded fraud analysts |
| F1 score | 2 × P × R ÷ (P + R) | One number that is high only when precision AND recall are both high | Good single summary at one threshold |
| F2 score | 5 × P × R ÷ (4P + R) | Like F1 but cares about recall twice as much | Use when missing fraud is costlier than a false alarm |
| FPR (false positive rate) | FP ÷ (FP + TN) | Of all innocent customers, what share did we bother? | Customer friction |
| FNR (false negative rate) | FN ÷ (FN + TP) = 1 − recall | Of all frauds, what share slipped through? | Fraud leakage |
| ROC-AUC | Area under the ROC curve | Chance that a random fraud gets a higher score than a random legitimate payment | Standard, used by the SMOTE paper, but too flattering here (see below) |
| PR-AUC / Average Precision | Area under the precision–recall curve | How well the model keeps precision high as it tries to catch more fraud | **Our main metric**: it is not fooled by the huge number of legitimate payments |
| MCC (Matthews correlation coefficient) | Uses all four boxes | From −1 (always wrong) through 0 (guessing) to +1 (perfect) | An honest single number under imbalance; optional extra |

An analogy for precision and recall: you are fishing for fraud fish in a sea of normal fish. **Recall** asks "how many of the fraud fish in the sea did my net catch?" **Precision** asks "of everything in my net, how much is fraud fish?" A giant net catches every fraud fish (recall 100%) but is mostly full of normal fish (precision tiny). A tiny careful net is clean but misses most fraud.

### The threshold

Most models do not say "fraud" or "legitimate" directly. They output a **score** between 0 and 1. We then choose a **threshold**: above it we call the payment fraud. The default threshold is 0.5, but it can be anything.

- Lower threshold (say 0.1): catch more fraud, bother more innocent people. Recall up, precision down.
- Higher threshold (say 0.9): bother fewer innocent people, miss more fraud. Precision up, recall down.

This matters for our study because **some imbalance techniques mostly just move the scores up**, which has a similar effect to lowering the threshold. A fair study must check whether a technique really helps, or only did something we could have done for free by moving the threshold. That is why we report both threshold-free metrics (PR-AUC, ROC-AUC) and metrics at a chosen threshold.

### ROC-AUC versus PR-AUC: the most important idea to explain in the viva

The **ROC curve** plots recall (catch rate) against FPR (the share of innocent people bothered), for every possible threshold. **ROC-AUC** is the area under it: 1.0 is perfect, 0.5 is coin-flipping.

The trap: FPR divides by the number of legitimate payments, which is enormous. Suppose on the test set our model raises 2,000 false alarms. FPR = 2,000 ÷ 85,000 ≈ 2.4%, which looks tiny, so ROC-AUC still looks excellent. But if it caught 130 frauds, precision = 130 ÷ 2,130 ≈ 6%: **94 out of every 100 alarms are innocent people.**

The **PR curve** plots precision against recall. Precision never uses the giant pile of true negatives, so it cannot hide those 2,000 false alarms. That is why PR-AUC (computed as **average precision** in scikit-learn) is our primary metric. This argument comes from a well-known paper by Davis and Goadrich (2006).

One honest detail: a random guesser's PR-AUC equals the fraud rate (about 0.0017), not 0.5. So PR-AUC numbers like 0.80 are excellent, and you should always state the baseline when you report them.

### Scores are not true probabilities

If we train on data where fraud was artificially boosted from 0.17% to, say, 10%, the model learns that fraud is common and gives everyone higher scores. A score of 0.6 from such a model does **not** mean "60% chance of fraud". In the app we therefore call it a **risk score**. If we want a real probability, we can **calibrate** it (re-map scores to observed fraud rates using validation data). This is an optional extra, covered in Part 6.

## Part 3 — The SMOTE paper explained

**Chawla and colleagues (2002) showed that inventing new minority examples between existing ones, then trimming the majority, beats both copying minority examples and only trimming the majority.** Our project tests whether that still holds on modern fraud data with modern models and metrics.

Full reference: N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique", *Journal of Artificial Intelligence Research* 16 (2002), 321–357.

### The problem they started from

In many real datasets the interesting cases are rare (cancer in mammograms, oil spills in satellite images, fraud). Two simple fixes already existed:

- **Random undersampling (RUS)**: throw away majority examples at random until the classes are closer in size. It helps the model notice the minority, but it throws away information.
- **Random oversampling (ROS)**: copy minority examples at random until the classes are closer in size. Nothing is thrown away, but copies add no new information.

The paper's key observation is about copies. Imagine a decision tree (a model that splits the data with yes/no questions like "is V14 below −5?"). If one fraud point is copied 50 times, the tree can draw a tiny box around that exact spot and get all 50 copies right. It learns "fraud lives exactly here", which is **overfitting**: memorising instead of generalising. The paper showed that ROS makes the tree's fraud regions **smaller and more specific**, and therefore not much better than undersampling.

### SMOTE's idea

Instead of copying, **invent new fraud points that lie between real fraud points**. Then the model learns "fraud lives around this area", a **larger and more general** region. The authors borrowed this from handwriting recognition, where people created extra training images by slightly rotating and stretching real ones. Tabular data cannot be rotated, so SMOTE does its "stretching" directly in **feature space**: the imaginary space where each payment is a point and each of its 30 numbers is one coordinate.

### The algorithm, step by step

1. Take one real fraud example. Call it **x**.
2. Find its **k nearest neighbours among the other fraud examples only** (the paper uses k = 5). "Nearest" means smallest **Euclidean distance**: the straight-line distance, worked out like Pythagoras but over 30 coordinates.
3. Pick one of those k neighbours at random. Call it **x̂** ("x-hat").
4. Pick a random number **λ** (lambda) between 0 and 1.
5. Create the new point:

```latex
x_{new} = x + \lambda \cdot (\hat{x} - x), \qquad \lambda \sim U(0, 1)
```

In words: start at x, walk towards x̂, and stop at a random fraction of the way. The new point always lies on the straight line between two real frauds. If λ = 0.3, it is 30% of the way from x to x̂.

6. Repeat for every fraud example until enough new points exist.

**How much to create (N%).** The paper describes the amount as a percentage of the minority class. N = 200% means "make 2 new points for every real fraud point", so the minority triples in size. They mostly used 100% to 500%.

**Why only fraud neighbours?** If SMOTE drew lines between a fraud point and a legitimate point, it would place new points labelled "fraud" in legitimate territory, teaching the model something false.

**Distance needs comparable scales.** Euclidean distance adds up squared differences across features. If one feature is measured in thousands (Amount) and the others in single digits, the big one decides who counts as "nearest". That is why we **scale features before SMOTE** (Part 5).

### SMOTE plus undersampling

The paper's best method combined two steps: SMOTE the minority by a moderate amount, then randomly undersample the majority. Densify fraud, trim normal.

### How they tested it

| Aspect | What the paper did |
| --- | --- |
| Datasets | 9 datasets from medicine, remote sensing, census and similar, with minority shares from a few percent to about 35% |
| Models | C4.5 (a decision tree), Ripper (a rule learner) and Naive Bayes (a simple probability model) |
| Competitors | Undersampling alone; for Ripper, changing the cost of each error type; for Naive Bayes, changing the class priors |
| Evaluation | 10-fold cross-validation, ROC curves, AUC, and the **ROC convex hull** |

**ROC convex hull, simply.** Every setting of a method gives one point on the ROC chart. Join the outermost points with a stretched rubber band: that is the convex hull. A method whose points sit on the rubber band is the best choice for *some* combination of costs and class ratios. SMOTE + undersampling put more points on the hull than the competitors on most datasets.

### Limitations the paper leaves open (good viva material)

- **It assumes the space between two frauds is also fraud.** If fraud forms two separate clusters with legitimate payments between them, SMOTE can draw a line straight through legitimate territory.
- **It ignores the majority class completely.** It never checks whether legitimate points sit along the line it draws. Later methods fix this: **Borderline-SMOTE** (2005) focuses on fraud points near the border, and **ADASYN** (2008) makes more points around harder fraud cases.
- **Noise gets amplified.** A mislabelled fraud point deep inside legitimate territory becomes the parent of many fake frauds.
- **Distance gets less meaningful in many dimensions.** With 30 features, all points start to look roughly equally far apart, so "nearest neighbour" means less.
- **Its evaluation predates PR curves becoming standard for imbalance.** ROC can look good while precision is poor.

### What we copy, what we change

| Topic | A. What the paper did | B. What we reproduce | C. What we modernise, and why |
| --- | --- | --- | --- |
| Problem | Rare-class learning on 9 small to medium datasets | Same problem | One large, extremely imbalanced dataset (0.17% fraud), much more extreme than the paper's |
| Methods | Undersampling, replication, SMOTE, SMOTE + undersampling | All four, plus a no-handling baseline | Add **class weighting**, the modern "free" alternative an examiner will ask about |
| SMOTE amount | 100%–500% | Test amounts in that spirit | Also test larger ratios, since our imbalance is far more extreme |
| Models | C4.5, Ripper, Naive Bayes | A tree-based model (Random Forest as the modern C4.5 relative) | Logistic Regression as an interpretable baseline, LightGBM as the modern industry standard |
| Metrics | ROC, AUC, ROC convex hull | ROC curves and ROC-AUC | **PR-AUC as primary**, plus precision, recall, F1/F2, FPR, FNR at chosen thresholds |
| Leakage control | Not discussed in modern terms | Resample training data only | Enforced in code with `imblearn` pipelines and automated tests |
| Uncertainty | Mostly single curves per dataset | — | Repeated CV, confidence intervals, statistical tests |
| Threshold | Traced along ROC | — | Tune the threshold fairly on validation data for every method |

## Part 4 — Research question and hypotheses

**Main question: how do different class-imbalance handling techniques change the fraud-catching performance of machine-learning models, once every technique is tuned and judged fairly?**

A **hypothesis** is a guess we write down *before* running the experiment, phrased so the data can prove it wrong. Writing them first stops us from inventing a story after seeing the numbers. Getting a hypothesis rejected is a perfectly good result; it is still a finding.

### Sub-questions

1. **RQ1.** Which technique gives the best PR-AUC for each model?
2. **RQ2.** Does the best technique depend on the model? (A simple linear model may need help that a strong tree model does not.)
3. **RQ3.** Once each method's threshold is tuned fairly, do the differences between techniques remain, or do they shrink?
4. **RQ4.** What does each technique trade off: precision against recall, training time, and how many synthetic or discarded rows it relies on?

### Hypotheses and how each one is decided

| ID | Hypothesis (plain words) | How we decide |
| --- | --- | --- |
| H0 (null) | No technique beats the untouched baseline on PR-AUC once thresholds are tuned | Rejected only if a technique wins with a statistically significant margin (Part 6) |
| H1 | Random **undersampling** raises recall but lowers precision and PR-AUC, because it throws away almost all legitimate examples | Compare RUS with baseline: recall, precision, FPR, PR-AUC |
| H2 | Random **oversampling** helps less than SMOTE for tree models, because copies cause overfitting | Compare ROS vs SMOTE for Random Forest; also compare training vs validation scores |
| H3 | **SMOTE + undersampling** matches or beats SMOTE alone, as in the 2002 paper | Compare those two cells per model |
| H4 | **Class weighting** performs about as well as the best resampling method for Logistic Regression and LightGBM, at far lower cost | Compare class-weighted vs best resampler; also training time |
| H5 | The gain from any technique is **largest for Logistic Regression** and smallest for LightGBM | Size of improvement over baseline, per model |

Why these and not others: each one is tied to a claim from the SMOTE paper (H2, H3), a well-known practical belief (H1, H4), or a modern question the paper could not ask (H5, RQ3). RQ3 is our most original contribution: many student projects compare techniques at the default threshold of 0.5, which unfairly punishes some methods.

## Part 5 — The dataset and every preprocessing step

**We use the public ULB credit-card dataset from Kaggle, remove exact duplicate rows, keep every outlier, log-transform and scale `Amount`, scale the rest, and leave raw `Time` out of the main experiment.** Each choice is justified below. The code reads column names from a config file, so a different dataset can be swapped in later.

### The dataset

The "Credit Card Fraud Detection" dataset on Kaggle was released by the Machine Learning Group of ULB (Université Libre de Bruxelles) with Worldline. It holds card payments made by European cardholders over two days in September 2013. Cite it as Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification", IEEE SSCI 2015, and check the licence shown on the Kaggle page.

The figures below are widely reported for this dataset. **Our EDA must re-check every one of them** and the report should quote our own numbers.

| Column | Type | What it is |
| --- | --- | --- |
| `Time` | number (seconds) | Seconds since the first payment in the file, from 0 to about 172,792 (48 hours) |
| `V1` … `V28` | numbers | 28 anonymised features. The bank transformed the real details with PCA so nobody can identify customers |
| `Amount` | number ≥ 0 | Payment amount, believed to be euros. Median about 22, mean about 88, maximum about 25,691 |
| `Class` | 0 or 1 | The answer: 1 = fraud, 0 = legitimate. This is our **target** |

| Fact to verify | Expected value |
| --- | --- |
| Rows | 284,807 |
| Frauds | 492 (0.173%), about 1 fraud per 578 payments |
| Missing values | None |
| Exact duplicate rows | About 1,081, of which about 19 are fraud |
| After removing duplicates | About 283,726 rows and 473 frauds |

**What is PCA?** Principal Component Analysis takes many original columns and mixes them into new columns that are combinations of the old ones, ordered from "explains the most variation" (V1) to "explains the least" (V28). The new columns hide what the originals were. So we know V14 matters, but nobody outside the bank knows what V14 *means*.

### Step-by-step preprocessing, each step justified

A rule for the whole project: **do nothing just because other people do it.** Every step must answer "what problem does this solve here?"

**1. Validate the schema (check the columns).** Before anything, check that all expected columns exist, are numbers, contain no infinities, and that `Class` contains only 0 and 1. Also record a **SHA-256 hash** of the file: a fingerprint, a long code that changes if even one character in the file changes. It proves everyone used the same data.

**2. Missing values.** The dataset has none. We still put a median **imputer** (a step that fills blanks with the training median) inside the pipeline, so that a user-uploaded CSV with a blank cell does not crash the app. It does nothing to the real data.

**3. Remove exact duplicates, before splitting.** Duplicates are probably logging repeats. If one copy lands in training and its twin in testing, the model is tested on a payment it has literally seen: a small leak. Removing them **before** the split is safe, because it uses no information about which rows go where. Report how many were removed.

**4. Keep every outlier.** An **outlier** is a value far from the usual range. Many tutorials delete them. Here that would be a disaster: fraud payments *are* the unusual ones, with extreme values on features like V14, V12 and V10. Deleting outliers deletes fraud. The Kaggle notebook you saw removes some outliers from fraud rows; we deliberately do not, and we say why in the report.

**5. Transform `Amount` with log(1 + Amount), then scale.** `Amount` is extremely **skewed** (most values small, a few huge). Taking the logarithm squeezes the huge values together, so a €20,000 payment no longer dwarfs everything in distance calculations. We add 1 first because log(0) is undefined and many amounts are exactly 0.

**6. Scale with RobustScaler, inside the pipeline.** **Scaling** puts features on comparable ranges. `RobustScaler` subtracts the median and divides by the **IQR** (interquartile range: the spread of the middle 50% of values). We pick it over the usual `StandardScaler` because extreme fraud values would distort the mean and standard deviation, while the median and IQR barely move. Why scale at all:

- **SMOTE** finds neighbours by distance; unscaled features with big ranges would dominate who counts as "near".
- **Logistic Regression** with regularisation (a penalty on large weights) penalises features unevenly unless they are on similar scales.
- **Tree models** do not need scaling, but scaling does not hurt them, and one shared preprocessing for all models keeps the comparison controlled.

The scaler must **learn its median and IQR from training data only**, which is why it lives inside the pipeline (Part 6).

**About scaling V1–V28.** They are already centred around zero, but their spreads differ (V1 varies roughly six times more than V28). Scaling them equalises their influence on SMOTE's distances. Leaving them unscaled would keep PCA's "importance ordering" in the distance. Both are defensible. We scale them, mainly so Logistic Regression's penalty treats every feature fairly, and we mention the alternative as a limitation.

**7. Leave raw `Time` out of the main experiment.** `Time` only says *where in a two-day window* a payment happened. A model could learn "there was a fraud burst around second 90,000", which would never repeat in the future. Gemini's plan converts it into hour of day, but that assumes second 0 was midnight, and the dataset does not say when it starts. So:

- **Main experiment:** features = V1–V28 + transformed Amount (29 features).
- **Optional side test:** add a cyclical "hour" feature (sin and cos of the hour, so 23:00 and 00:00 count as close), clearly labelled as assuming a midnight start.
- **Time is still used** for one sensitivity check: a **temporal split** (train on the earlier part, test on the later part), which mimics real deployment where you always predict the future.

### Is SMOTE appropriate for PCA features?

Mostly yes, and this is a good viva point.

- All 29 features are continuous numbers, which is what plain SMOTE needs. (If there were categories such as "merchant type", we would need **SMOTE-NC**, a version for mixed data. There are none, so we don't.)
- PCA is a *linear* transformation. Drawing a straight line between two points in PCA space corresponds to drawing a straight line between them in the bank's original (standardised) space. So SMOTE's interpolation is not distorted by PCA itself.
- **But**: if the bank's original data contained categories (very likely, for example merchant category) that were turned into numbers before PCA, then a point halfway between two frauds may correspond to a half-merchant, half-category combination that cannot exist. We cannot check this, so we list it as a limitation: **synthetic frauds are geometrically plausible, not guaranteed to be realistic payments.**

### Temporal considerations

Real fraud changes over time (fraudsters adapt; this is called **concept drift**). Two days of data cannot show this properly. Our main split is a random **stratified** split (both parts keep the same 0.17% fraud rate). The temporal split above is a quick check of whether performance holds when predicting "later" from "earlier".

### Leakage risks to watch for in this dataset

**Data leakage** means information from the test data sneaking into training, which makes scores look better than reality.

| Risk | Where it could happen | Our protection |
| --- | --- | --- |
| Resampling before splitting | SMOTE on the full data | Resampling only inside the training pipeline |
| Scaling on all data | Scaler sees test medians | Scaler inside the pipeline |
| Duplicate twins across splits | Repeated rows | Deduplicate first |
| Picking settings with the test set | Tuning, threshold, champion choice | Use cross-validation only; open the test set once |
| PCA fitted by the bank on all data | Before release | Outside our control; stated as a limitation |
| `Time` memorisation | Fraud bursts at specific seconds | Raw `Time` excluded |

## Part 6 — The experiment design

**We run 6 techniques × 3 models = 18 combinations. Every combination gets the same data split, the same preprocessing, the same choices of resampling ratio, the same tuning budget and the same threshold-tuning rule. The only thing that changes is the technique (and, as a second factor, the model).** That is what makes it a *controlled* experiment.

### The 6 techniques

| ID | Technique | What it does to the training data | In imblearn / sklearn |
| --- | --- | --- | --- |
| A | Baseline | Nothing | no sampler |
| B | Random oversampling (ROS) | Copies fraud rows until fraud reaches ratio *r* of legitimate | `RandomOverSampler` |
| C | Random undersampling (RUS) | Deletes legitimate rows until fraud reaches ratio *r* | `RandomUnderSampler` |
| D | SMOTE | Invents fraud rows between real ones until ratio *r* | `SMOTE` |
| E | SMOTE + RUS | SMOTE a little (to ratio *a*), then delete legitimate rows until ratio *b* | `SMOTE` then `RandomUnderSampler` |
| F | Class weighting | Data unchanged; each fraud mistake counts *w* times more in training | `class_weight` / `scale_pos_weight` |

**Are all six necessary?** A to E are what the topic asks for. F is strongly recommended because the first question a good examiner asks is "why not just use class weights?" **Borderline-SMOTE** and **ADASYN** are optional; add **one** of them (Borderline-SMOTE, since it directly fixes a SMOTE limitation) only if the core 18 are finished early.

**What "ratio r" means.** In imblearn, `sampling_strategy = 0.1` means "after resampling, fraud count = 10% of legitimate count". Our raw ratio is about 0.0017 (1 : 578).

### Fair ratio choices (fixing Gemini's plan)

If SMOTE is tried only at ratio 0.1 and RUS only at 0.5, we are comparing ratios, not techniques. So **every resampler picks from the same ratio list**, and tuning chooses the best one for each:

| Target ratio r | Roughly what it means for SMOTE / ROS | Link to the paper |
| --- | --- | --- |
| 0.005 | Fraud ×3 (N = 200%) | Inside Chawla's 100%–500% |
| 0.01 | Fraud ×6 (N = 500%) | Upper end of Chawla's range |
| 0.05 | Fraud ×29 | Modern, more aggressive |
| 0.1 | Fraud ×58 | Modern |
| 0.5 | Fraud ×290 | Very aggressive |
| 1.0 | Equal classes | The common (often bad) 50/50 default |

For **E (SMOTE + RUS)**: SMOTE amount *a* ∈ {0.005, 0.01, 0.05}, then RUS to *b* ∈ {0.1, 0.5, 1.0}, always with *a* < *b*. This is Chawla's "moderate SMOTE, then trim" idea.

For **F (class weighting)**: the fraud weight is chosen so that it matches the same ratio list (weight = *r* ÷ 0.0017), plus `"balanced"`. Now F is compared like-for-like with the resamplers.

### The 3 models (and why not more)

| Model | Plain description | Why included | Cost | Interpretable? | Needs scaling? |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | Draws one flat dividing surface; output = weighted sum of features turned into a 0–1 score | Simple, explainable baseline; shows how resampling moves a linear boundary | Seconds | Yes: one weight per feature | Yes |
| Random Forest | Hundreds of decision trees, each on a random sample, voting together | Closest modern relative of the paper's C4.5 tree; most sensitive to copying (H2) | Minutes per fit on \~200k rows | Partly | No (harmless) |
| LightGBM | Gradient-boosted trees: each new small tree fixes the previous trees' errors | Industry standard for tabular fraud; fast; tests whether strong models still need help (H5) | Fast | Partly (SHAP works well) | No (harmless) |

Not included, and why: **SVM** gets very slow on 200k rows; **k-nearest neighbours as a classifier** is slow to predict and memory-heavy; **neural networks** usually do not beat boosted trees on 29 tabular features and add tuning complexity. Ten models would make the report longer, not better.

**If time is short (MVP):** Logistic Regression + Random Forest only (12 combinations). Add LightGBM if there is time.

### Splitting the data

1. **Hold out 30% as the test set**, stratified (keeps the 0.17% fraud rate), with a fixed random seed. That is roughly 85,000 payments and about 142 frauds. Lock it away. Nobody looks at its scores until the very end.
2. The other **70% is the training set** (about 198,600 payments, about 331 frauds).
3. Inside the training set, use **stratified 5-fold cross-validation** for tuning and comparison.

**Cross-validation, simply.** Cut the training set into 5 equal slices, each keeping the 0.17% fraud rate. Train on 4 slices, score on the 5th (the **validation fold**). Rotate so each slice is the validation fold once. Average the 5 scores. This gives a much more reliable estimate than a single check, and it never touches the test set.

**Why 70/30 and not 80/20?** With only 473 frauds, an 80/20 split leaves about 95 test frauds. 70/30 leaves about 142, which makes the final test numbers noticeably less jumpy. The training set still keeps about 331 frauds.

### How resampling is kept inside training (leakage prevention)

We use `imblearn.pipeline.Pipeline`. A **pipeline** is a chain of steps run in order. The special imblearn version knows that a sampler should only run during **training** (`fit`) and must be **skipped** during **prediction** (`predict` / `predict_proba`).

```python
Pipeline([
    ("preprocess", preprocessor),   # log(Amount) + RobustScaler
    ("smote", SMOTE(sampling_strategy=a, k_neighbors=5, random_state=SEED)),
    ("rus", RandomUnderSampler(sampling_strategy=b, random_state=SEED)),
    ("model", classifier),
])
```

What happens inside cross-validation, for each of the 5 folds:

1. The preprocessor learns medians and IQRs from the 4 training slices only.
2. SMOTE invents new frauds using only fraud rows from those 4 slices.
3. RUS trims legitimate rows from those 4 slices.
4. The model trains on the result.
5. The 5th slice is scaled with the already-learned numbers, **not resampled**, and scored at its real 0.17% fraud rate.

**Why the order is preprocess → SMOTE → RUS → model.** Scaling must come before SMOTE because SMOTE measures distances. SMOTE only ever looks at fraud rows, so running it before or after RUS creates the same kind of synthetic points. We put SMOTE first for two reasons: its ratio *a* then means the same thing as in technique D (fraud as a share of the *full* legitimate count), which keeps D and E comparable; and it matches the order used in the paper. RUS then sets the final ratio *b*.

**Why this prevents leakage.** If SMOTE ran on all data first, a synthetic training point could be built from a fraud that later lands in the test set. Part of a test answer would be baked into training. The test score would then measure memory, not skill. Inside the pipeline, test and validation rows are never seen by SMOTE, the scaler or the model during training.

&#91;embedded content: leakage-free experiment flow · split, 5-fold CV, one test evaluation\]

Samplers run only while a pipeline is being trained; when it predicts, imblearn skips them, so validation and test rows are always scored at the real fraud rate.

### Tuning, fairly

**Hyperparameters** are settings chosen before training (tree depth, number of trees, regularisation strength, resampling ratio). **Tuning** means trying several and keeping the best.

- Use `RandomizedSearchCV` (tries random combinations from a list) with **the same number of tries for every cell**, for example 20 tries.
- Score with `scoring="average_precision"` (PR-AUC). Not accuracy (useless here), not F1 (depends on the 0.5 threshold, which resampling shifts).
- The search space includes the model's settings **and** the technique's ratio from the shared list above.
- `refit=True` retrains the best combination on the whole 70% training set.

**Nested cross-validation** (a second cross-validation wrapped around the tuning) would give an even less biased estimate, but it multiplies compute by about 5. We don't need it, because our locked test set plays the role of the outer check.

### Choosing a threshold, fairly

For the best configuration of each cell:

1. Get **out-of-fold** scores: each training row is scored by the fold model that did **not** train on it (`cross_val_predict` with `method="predict_proba"`).
2. On those scores, find the threshold that maximises F1 (and separately F2).
3. Freeze that threshold and use it, unchanged, on the test set.

We report test metrics at **both** 0.5 and the tuned threshold. This answers RQ3: does a technique really help, or did it mostly shift scores?

### Measuring uncertainty (fixing Gemini's plan)

With about 142 test frauds, catching one more or one fewer changes recall by 0.7 percentage points. Small differences can be luck. So:

1. **Repeated cross-validation.** Evaluate each cell's best configuration with 5 folds × 3 repeats (different shuffles) = 15 validation scores per cell.
2. **Statistical test.** For each model, compare every technique with the baseline using the **Wilcoxon signed-rank test** on the paired fold scores (paired because each fold is the same data for both techniques), with **Holm correction** because we run several comparisons. Also run a **Friedman test** across all six techniques and show a **critical difference diagram** (a standard picture showing which techniques are statistically tied).
3. **Bootstrap confidence intervals on the test set.** Resample the test set with replacement 1,000 times, recompute PR-AUC each time, and report the middle 95% range, for example "PR-AUC 0.81 (95% CI 0.75–0.86)".

One honest caveat for the report: fold scores are not fully independent (folds share training data), so p-values are approximate. Report effect sizes and confidence intervals, not only p-values.

### Optional: calibration

Resampled models produce inflated scores (Part 2). If the app should show a real "probability", fit an **isotonic** or **sigmoid calibrator** on the out-of-fold scores, which learns to map scores back to real fraud rates. This does not change PR-AUC or ROC-AUC (ranking is unchanged); it only makes the number honest. Optional, but a nice touch in the demo.

### Compute budget (honest estimate)

Per cell: 20 tries × 5 folds + 15 repeated-CV fits + 1 refit ≈ 116 model fits. Over 18 cells ≈ 2,100 fits. Logistic Regression and LightGBM fits take seconds; Random Forest fits on \~200k rows can take one to a few minutes, and more after heavy oversampling. Plan for:

- A **quick mode** for development: 10% of the training data, 3 tries, 3 folds. Runs in minutes.
- The **full run** split by model across 2–3 team laptops or Google Colab sessions, each saving its own results file; a script merges them.
- If Random Forest is too slow: fewer tries (10), `n_estimators=200`, `max_samples=0.5`, and say so in the report.

### What the experiment produces

**Main results table** (one row per technique × model):

| Model | Technique | Best ratio | PR-AUC (95% CI) | ROC-AUC | Threshold | Precision | Recall | F1 | F2 | FPR | FNR | Train time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e.g. LightGBM | SMOTE + RUS | a=0.01, b=0.5 | 0.xx (0.xx–0.xx) | 0.9x | 0.xx | … | … | … | … | … | … | … |

**Figures:** class distribution before and after each resampler; PR curves and ROC curves (one panel per model, one line per technique); confusion matrices at the tuned threshold; grouped bar chart of PR-AUC with error bars; critical difference diagram; precision and recall versus threshold; and a 2D SMOTE illustration for teaching.

**Saved files:** every fold score, every test metric, chosen hyperparameters, threshold, timings, data hash and library versions, so anyone can rebuild every number.

## Part 7 — Explainability: what we can honestly explain

**We can show which features pushed a payment's score up or down, and how unusual each value was. We cannot say what those features mean in the real world, because the bank hid that.** The app's "Why was this flagged?" panel is built around that honesty.

### Two kinds of explanation

- **Global**: what the model relies on in general ("V14, V4 and V12 matter most").
- **Local**: why *this one* payment got *this* score ("V14 = −9.4 pushed it strongly towards fraud").

### What we use, and why (not just because it is popular)

| Model | Local explanation | Why this method |
| --- | --- | --- |
| Logistic Regression | Exact contribution per feature = weight × (scaled value − training average) | The model is a weighted sum, so this is the exact truth, no approximation needed |
| LightGBM / Random Forest | **TreeSHAP** values | SHAP splits a prediction fairly among features (an idea from game theory called Shapley values). For tree models, the TreeSHAP algorithm computes it exactly and quickly, so it is justified here. KernelSHAP (the slow, general version) is not used |

Global explanation: Logistic Regression weights; **permutation importance** on validation data (shuffle one feature and measure how much PR-AUC drops; a reliable, model-agnostic check); and a SHAP summary plot for the tree model. We avoid Random Forest's built-in "impurity importance", which is known to be biased towards features with many distinct values.

### How to word explanations honestly

For each top feature, the app shows three things:

1. **Direction and size**: "pushed the score up by 2.6 units" (for SHAP on LightGBM these units are log-odds, the scale the model adds up internally; say so in a tooltip).
2. **How unusual the value is**: "V14 = −9.4, lower than 99.9% of legitimate training payments". This is the most understandable part for a non-expert, and it is completely truthful.
3. **A fixed disclaimer**: "V1–V28 are anonymised components; their real-world meaning is unknown."

Never write "flagged because the merchant was suspicious" or similar. We do not know that.

### Two traps to avoid

- **Explanations are not causes.** SHAP says what the *model* used, not what *caused* fraud.
- **Moving one slider can create impossible payments.** If the user drags V14 alone to −10 while keeping everything else normal, the payment may look like nothing in the real data. The model will still output a score, but it is extrapolating. The app should show a small note: "Edited transaction; may not resemble real payments." The preset buttons (real test-set payments) are the trustworthy demo path.

## Part 8 — Software architecture, repository and reproducibility

**The project is one GitHub repository with three layers that only talk through files and an API: the research code writes result files, the backend reads them, the frontend asks the backend.** If each layer respects the agreed file formats, ten people can work in parallel without breaking each other's code.

&#91;embedded content: system architecture · offline research, online demo\]

The arrows between frontend and backend carry JSON over HTTPS; the only link from research to website is the set of saved files.

### The repository layout

A **repository** (repo) is the project folder tracked by Git, the tool that records every change and lets many people work on the same code.

```
fraud-imbalance-study/
├── configs/                 # settings files (YAML): dataset columns, seeds, ratio lists, model search spaces
├── data/
│   ├── raw/                 # creditcard.csv goes here; NEVER committed to Git
│   └── sample/              # small synthetic CSVs for tests and the demo (safe to commit)
├── notebooks/               # exploration only: EDA, SMOTE 2D demo; no logic the pipeline depends on
├── src/fraudstudy/          # the research code, as an installable Python package
│   ├── data/                # load, validate, deduplicate, split
│   ├── features/            # Amount log transform, optional hour feature, preprocessor builder
│   ├── sampling.py          # builds the sampler steps for techniques A–F
│   ├── models.py            # model builders (search spaces live in configs/)
│   ├── experiment/          # CV, tuning, thresholds, repeated CV, runner
│   ├── evaluation/          # metrics, bootstrap CIs, statistical tests, plots
│   ├── explain.py           # LR contributions, LightGBM SHAP values, feature percentiles
│   └── config.py, paths.py, seed.py, io.py, contracts.py  # shared helpers and data contracts
├── scripts/                 # command-line entry points: run_eda, run_experiments, export_artifacts
├── artifacts/               # outputs: results JSON/CSV, figures, model files, manifest
├── backend/                 # FastAPI app (reads artifacts/)
├── frontend/                # React + Vite + TypeScript dashboard
├── tests/                   # automated tests, including anti-leakage tests
├── docs/                    # report draft, architecture notes, decisions log, viva prep
├── requirements.txt         # exact Python library versions
├── Dockerfile, docker-compose.yml
└── README.md                # how to install and reproduce everything
```

| Folder | Responsibility | Rule |
| --- | --- | --- |
| `configs/` | Every number that could change | No magic numbers in code |
| `data/raw/` | The real dataset | Listed in `.gitignore`; each person downloads it from Kaggle |
| `src/fraudstudy/` | All research logic | Functions with type hints, docstrings and tests |
| `notebooks/` | Exploring and teaching | May import from `src/`, never the other way round |
| `scripts/` | Run things | Thin: parse arguments, call `src/` functions |
| `artifacts/` | Everything produced | Small JSON and figures committed; large model files via release or Git LFS |
| `backend/` | API | Never trains models; only loads artifacts |
| `frontend/` | Website | Only talks to the backend API |
| `tests/` | Safety net | Must pass before any merge |
| `docs/` | Writing | Includes a **decisions log**: every design choice, date and reason |

### The contract: the most important file in the project

In week 1, the team agrees on the exact shape of the result files the experiment produces and the API sends. Write it down in `docs/contracts.md` with examples. Then the frontend team can build against fake example files while the experiment is still running. Key files:

| File | Contents | Produced by | Read by |
| --- | --- | --- | --- |
| `artifacts/results/comparison.json` | One record per model × technique: best params, CV mean ± std, test metrics at 0.5 and tuned threshold, bootstrap CI, train time | Experiment runner | Backend, report |
| `artifacts/results/fold_scores.csv` | Every fold score of every cell | Experiment runner | Statistics script |
| `artifacts/results/curves.json` | Downsampled ROC and PR curve points per cell | Evaluation | Backend |
| `artifacts/results/stats.json` | Wilcoxon / Friedman results | Statistics script | Backend, report |
| `artifacts/results/eda.json` | Class counts, distributions, before/after resampling counts | EDA script | Backend |
| `artifacts/models/manifest.json` | List of saved models: file name, model, technique, threshold, feature list, data hash, library versions | Exporter | Backend |
| `artifacts/models/*.joblib` | The fitted pipelines | Exporter | Backend |
| `artifacts/demo/presets.json` | 5–6 real test-set payments for one-click demo | Exporter | Backend |

### Reproducibility checklist

**Reproducible** means another student can run our commands and get the same numbers.

- [ ] One seed (`42`) in the config, passed to every split, sampler and model
- [ ] Exact library versions in `requirements.txt` (produce it with `pip freeze` once everything works)
- [ ] Python version written in the README (for example 3.11)
- [ ] SHA-256 hash of `creditcard.csv` checked when loading
- [ ] Every experiment setting read from `configs/`, and a copy of the config saved next to the results
- [ ] README with exact commands: install, download data, run EDA, run quick experiment, run full experiment, start backend, start frontend
- [ ] Results files include the git commit hash, so you know which code produced them

### "MLOps", kept realistic

**MLOps** means the practices for running machine learning reliably: tracking experiments, versioning models, deploying them. For a college project we keep it lightweight:

| Need | Our solution | Why not something bigger |
| --- | --- | --- |
| Track experiments | Results written to JSON/CSV with config and git hash; **MLflow** optional on laptops for browsing runs | A tracking server costs time and money to host |
| Version models | `manifest.json` + file names like `lightgbm__smote_rus.joblib` | A model registry service is overkill |
| Version data | SHA-256 hash in the config | DVC is useful but adds a learning curve |
| Automated checks | GitHub Actions runs `pytest` and a linter (`ruff`) on every pull request | Free and simple |
| Deploy | Docker image for the backend, static site for the frontend | No Kubernetes, no paid cloud |

## Part 9 — Backend (FastAPI)

**The backend is a small Python web server with 8 endpoints. It loads the saved models and result files once at start-up, then answers requests from the frontend in milliseconds. It never trains anything.**

### Words you need

- **Server**: a program that waits for requests and answers them.
- **FastAPI**: a Python library for writing servers. It checks incoming data automatically and creates a test page at `/docs` where you can try every endpoint in the browser.
- **Pydantic**: the library FastAPI uses to describe the exact shape of data ("Amount must be a number ≥ 0"). Bad data is rejected with a clear message before our code even runs.
- **HTTP methods**: `GET` = "give me something", `POST` = "here is data, process it".
- **Status codes**: numbers that say how a request went. 200 = OK, 400 = your request is wrong, 404 = not found, 413 = file too big, 422 = data failed validation, 500 = our bug, 503 = not ready yet.

### The endpoints

All start with `/api/v1` (the `v1` lets us change the design later without breaking old clients).

| Method and path | What it does | Needed? |
| --- | --- | --- |
| `GET /health` | "Are you alive? Are models loaded?" Returns status, number of models, data hash, version | Core (deployment checks use it) |
| `GET /models` | Lists available models (model, technique, threshold, features) and which is the default | Core |
| `GET /results` | The full comparison: metrics, CIs, curves, statistics | Core (replaces Gemini's separate `/metrics`, `/experiments`, `/comparison`) |
| `GET /eda` | Class counts, fraud %, distributions, before/after resampling counts | Core |
| `GET /presets` | 5–6 real test-set payments for one-click demos | Core (solves the "30 input boxes" problem) |
| `POST /predict` | Score one payment, with explanation | Core |
| `POST /predict/batch` | Score an uploaded CSV | Core |
| `POST /smote-demo` | Runs real `imblearn` SMOTE on a tiny 2D toy dataset for the teaching page | Nice to have (frontend has a fallback) |

### `POST /predict`: request and response

Request:

```json
{
  "features": {"V1": -2.31, "V2": 1.95, "...": 0.0, "V28": 0.12, "Amount": 1.0},
  "model_id": "lightgbm__smote_rus",
  "threshold": null
}
```

`model_id` and `threshold` are optional. If missing, the default model and its tuned threshold are used.

Response (improved from Gemini's version):

```json
{
  "label": "FRAUD",
  "risk_score": 0.943,
  "calibrated_probability": 0.71,
  "threshold": 0.62,
  "threshold_source": "tuned_on_validation_F1",
  "model": {"id": "lightgbm__smote_rus", "model": "LightGBM", "technique": "SMOTE + RUS"},
  "explanation": {
    "method": "TreeSHAP",
    "units": "log-odds",
    "top_features": [
      {"feature": "V14", "value": -9.42, "contribution": 3.84, "legit_percentile": 0.1},
      {"feature": "V12", "value": -5.88, "contribution": 1.95, "legit_percentile": 0.4}
    ]
  },
  "warnings": [],
  "disclaimer": "Research demo on anonymised public data. Not a real fraud decision.",
  "latency_ms": 8.4
}
```

Why this is better: it says `risk_score` instead of pretending to be a probability; it includes a calibrated probability only if we built a calibrator (else `null`); it says where the threshold came from; it reports explanation units; and it has a `warnings` list (for example "edited transaction may not resemble real payments").

### Validation and errors

- Every feature the model needs must be present and be a **finite** number (no NaN, no infinity). Unknown extra fields are rejected, so typos are caught.
- `Amount` must be between 0 and a sensible maximum (for example 1,000,000).
- `threshold`, if given, must be between 0 and 1.
- Unknown `model_id` → 404 with the list of valid ids.
- The backend builds the feature table **in the exact column order stored in the manifest**, so the model never gets columns in the wrong order.
- Errors always come back as JSON in one shape: `{"error": {"code": "...", "message": "..."}}`.

### `POST /predict/batch`

1. Accept one uploaded file field named `file`.
2. Reject if larger than 5 MB (413), not ending in `.csv` (400), or more than 10,000 rows (413).
3. Read it **in memory only**; never save it to disk, never execute anything from it.
4. Check the required columns exist; ignore `Time`; if a `Class` column is present, also compute metrics against it (useful for demos).
5. Score all rows at once (vectorised, so 10,000 rows take well under a second).
6. Return a summary (rows, flagged count, flagged %, total flagged amount, and metrics if `Class` existed) plus row-level results. The frontend builds the downloadable CSV from this.

### Start-up and memory

At start-up the backend reads `manifest.json`, loads the models listed there, checks each one's library versions match, and loads the result JSON files. If something is missing, `/health` reports `"degraded"` with the reason instead of crashing.

**Do not load all 18 models on a small server.** A Random Forest file can be tens of megabytes. Deploy a chosen set: the default model plus one model per technique for a single model family (for example the 6 LightGBM pipelines, which are small). The comparison page does not need model files at all; it reads `results` JSON.

### Logging

Log the time, endpoint, model id, label, latency and a request id. **Never log the feature values or uploaded file contents.** Even though our data is anonymised, this is the right habit for anything finance-related.

## Part 10 — Frontend dashboard and the demo flow

**The website has 6 pages, ordered so that clicking through them left to right *is* the 10-minute presentation: problem → SMOTE → results → try it → batch → conclusions.**

### Words you need

- **React**: a JavaScript library for building pages out of reusable pieces called **components** (a metric card, a chart, a button).
- **State**: the data a component remembers right now (which preset is selected, the current threshold).
- **TypeScript**: JavaScript with types, so mistakes like sending text where a number is expected are caught before the page runs.
- **Vite**: the tool that runs the site on your laptop while you develop and bundles it for deployment.
- **Tailwind CSS**: a way to style pages with short class names (`p-4 rounded-xl`) instead of writing separate style files.
- **Recharts**: a React chart library for line, bar and area charts.
- **TanStack Query** (optional but recommended): handles fetching data from the backend, including loading and error states, so every page does not reinvent it.

### The pages

| # | Page | What is on it | Backend calls |
| --- | --- | --- | --- |
| 1 | **Overview** | Title and research question; 3 headline cards (payments, frauds, fraud %); class-imbalance bar chart with a linear/log toggle (on a linear scale the fraud bar is almost invisible, which makes the point); "Why accuracy lies" card comparing an always-legitimate model (99.83% accuracy, 0% recall) with our best model; the headline finding with its confidence interval | `/eda`, `/results` |
| 2 | **Imbalance & SMOTE Lab** | Bar chart of training counts before and after each technique; the SMOTE explainer (below) | `/eda`, `/smote-demo` |
| 3 | **Results** | Model filter; sortable comparison table (best value per column highlighted, CIs shown); PR curves and ROC curves side by side; PR-AUC bar chart with error bars; significance markers; threshold slider showing how precision, recall and FPR change | `/results` |
| 4 | **Try a transaction** | Preset cards (real test payments: everyday legitimate, large legitimate, typical fraud, small-amount fraud, borderline case); model and threshold selectors; result card with a risk gauge; "Why was this flagged?" bar chart; a mini-table showing how each technique's model scores the same payment; an "advanced" drawer to edit any feature or paste JSON | `/presets`, `/models`, `/predict` |
| 5 | **Batch check** | Drag-and-drop CSV upload plus "use sample file" button; summary cards; table sorted by risk with a "flagged only" filter; download results as CSV | `/predict/batch` |
| 6 | **Method & team** | The leakage-free pipeline diagram; one card per technique (rule, pro, con); the A/B/C table from Part 3; limitations; team members and tech stack | none (static) |

### The SMOTE explainer (page 2)

A 2D scatter plot of a small toy dataset (grey = legitimate, red = fraud) with a 4-step stepper:

1. **Imbalance**: few red points among many grey.
2. **Neighbours**: click a red point; dashed lines go to its k nearest red neighbours.
3. **Interpolation**: a new point appears at a random spot on one line, with the formula x\_new = x + λ(x̂ − x) and the actual λ shown.
4. **Result**: all synthetic points (a different colour or shape), plus a simple decision boundary before and after.

Controls: a slider for **k** (1–15) and one for the **amount** of oversampling. When k is large, show a note: "large k can connect distant fraud clusters, creating synthetic points in legitimate territory." Use a toy dataset with two separate fraud clusters on purpose, so this failure is visible.

The frontend should include a built-in copy of the toy data and a simple JavaScript SMOTE, so the page still works if the backend is asleep.

### Design rules (make it look like a product, not a college HTML page)

- **Layout**: a left sidebar with the 6 pages (collapses to a top menu on phones); content in a centred column with cards.
- **Colour**: neutral greys plus one accent colour. Red only for fraud, green only for legitimate. **Never colour alone**: add icons or labels, since some people are colour-blind.
- **Typography**: one clean font (for example Inter); big numbers in cards; short labels.
- **Every data view has three extra states**: **loading** (skeleton placeholders), **empty** ("Upload a CSV to see results"), **error** (a clear message and a retry button).
- **Cold-start banner**: free servers sleep; on page load call `/health` and, if slow, show "Waking up the model server (up to a minute)…".
- **Accessibility**: buttons reachable with the keyboard, visible focus outlines, alt text or text summaries for charts, enough contrast.
- **Numbers**: rates shown with sensible precision (FPR 0.14%, not 0.0013812).

### The 10-minute demo, minute by minute

| Time | Page | What you do | What you say |
| --- | --- | --- | --- |
| 0:00–1:00 | Overview | Show headline cards and the linear-scale chart | "Only 0.17% of payments are fraud." |
| 1:00–2:00 | Overview | Point at the accuracy card | "A model that never flags anything is 99.83% accurate and useless. So we never use accuracy." |
| 2:00–3:30 | SMOTE Lab | Step through 1–4; drag k up | "SMOTE invents new fraud points between real ones in feature space. Large k can bridge clusters." |
| 3:30–4:00 | Method | Show the pipeline diagram | "Resampling only ever touches training folds. The test set was locked away." |
| 4:00–6:00 | Results | Table, then PR vs ROC curves, then threshold slider | "ROC-AUC looks great for everyone; PR-AUC separates them. Here is what changes when thresholds are tuned fairly." |
| 6:00–7:30 | Try a transaction | Click "typical fraud" preset, then "everyday legitimate"; open explanation | "Risk score, threshold, and which features pushed it up, with how unusual they were." |
| 7:30–8:30 | Batch | Load sample file; show summary; download | "The same model on hundreds of payments at once." |
| 8:30–10:00 | Method & team | Limitations and conclusion | "Our answer to the research question, and what we cannot claim." |

Rehearse with the **deployed** site and also with a **local** copy running, in case the internet or the free server fails during the demo. Record a backup screen video.

## Part 11 — Deployment and security

**Put the frontend on Vercel (free) and the backend on Render's free web service as a Docker container, carrying only small models. Keep a local Docker Compose copy as a backup for demo day.**

### Words you need

- **Deploy**: put the app on a computer on the internet so anyone with the link can use it.
- **Docker**: a way to package the backend with its exact Python version and libraries into an **image**, so it runs the same on any computer. A running image is a **container**.
- **Docker Compose**: one command (`docker compose up`) that starts the backend and frontend together on your laptop.
- **Environment variable**: a setting given to a program from outside its code (like the allowed website address), so the same code works in development and production.
- **CORS** (cross-origin resource sharing): a browser safety rule. Our frontend (on vercel.app) and backend (on onrender.com) live at different addresses, so the backend must explicitly say "I accept requests from this frontend address".
- **HTTPS**: encrypted web traffic (the padlock). Vercel and Render provide it automatically.

### Where each part runs

| Part | Host | Cost | Notes |
| --- | --- | --- | --- |
| Frontend (built React site) | Vercel (alternatives: Netlify, Render static site) | Free | Set `VITE_API_BASE_URL` to the backend address |
| Backend (FastAPI in Docker) | Render free web service | Free | 512 MB memory, 0.1 CPU, sleeps after 15 idle minutes, takes about a minute to wake ([Render docs](https://render.com/docs/free)) |
| Model and result files | Inside the backend image | Free | Render's free disk is wiped on restart, so files must be baked into the image, not uploaded at runtime |
| Database | None | — | Nothing needs saving between requests, so a database would only add work |

**Correction to Gemini's plan:** it suggests Hugging Face Spaces as a free 16 GB backup. Hugging Face's own documentation now says Docker and Gradio Spaces require a paid plan to create on personal accounts; only static Spaces are free ([Hugging Face docs](https://huggingface.co/docs/hub/en/spaces-overview)). Don't plan around it unless someone already has a paid account.

### Fitting into 512 MB

- Deploy **Logistic Regression and LightGBM pipelines only** (all small). Random Forest results still appear on the Results page from JSON; its model file just isn't served.
- For LightGBM explanations, use LightGBM's built-in SHAP values (`predict(..., pred_contrib=True)`), which avoids installing the heavy `shap` library on the server. Use `shap` only in notebooks.
- Run one server worker (`uvicorn --workers 1`).
- Test memory locally: `docker stats` while calling `/predict/batch` with 10,000 rows. Stay under about 400 MB.

### Handling the sleeping server

Render's free backend sleeps after 15 minutes without traffic. On demo day:

1. Open the site 5 minutes before presenting so it wakes up.
2. The frontend calls `/health` on load and shows a friendly "waking up" banner if it is slow.
3. Have the local Docker Compose version running on a laptop as a fallback, and a recorded video as a last resort.

### Configuration

| Variable | Where | Example | Purpose |
| --- | --- | --- | --- |
| `ALLOWED_ORIGINS` | Backend | `https://fraud-study.vercel.app,http://localhost:5173` | CORS whitelist |
| `ARTIFACTS_DIR` | Backend | `/app/artifacts` | Where models and results live |
| `DEFAULT_MODEL_ID` | Backend | `lightgbm__smote_rus` | Model used when none is chosen (set after results are in) |
| `MAX_UPLOAD_MB` / `MAX_BATCH_ROWS` | Backend | `5` / `10000` | Upload limits |
| `LOG_LEVEL` | Backend | `INFO` | How chatty logs are |
| `VITE_API_BASE_URL` | Frontend (build time) | `https://fraud-study-api.onrender.com` | Where the frontend sends requests |

Development: Vite forwards `/api` calls to `http://localhost:8000`, so no CORS trouble locally. Production: the variables above. Never hard-code addresses in code.

### Security basics

This is a demo on public, anonymised data, but it is about payments, so we show good habits:

- **No real card data, ever.** The app never asks for card numbers, names, expiry dates or CVV. State this on the site and in the report.
- **Validate every input** (Part 9): types, finite numbers, ranges, no extra fields.
- **Uploads**: CSV only, 5 MB and 10,000 rows maximum, parsed in memory, never saved, never executed.
- **CSV export safety**: when the frontend builds a downloadable CSV, any text cell starting with `=`, `+`, `@` or a tab is prefixed with `'`, so spreadsheet programs don't treat it as a formula ("CSV injection"). Numbers are left alone.
- **Rate limiting**: limit each visitor to, say, 60 predictions and 10 batch uploads per minute (the `slowapi` library does this in a few lines), so one person cannot overload the free server.
- **CORS**: only our frontend's address is allowed in production.
- **Secrets**: we have almost none, but any that appear go in host environment settings, never in Git. Add `.env` to `.gitignore`.
- **Logs**: no feature values, no file contents (Part 9).
- **Dependencies**: pinned versions; GitHub's Dependabot can warn about known vulnerabilities.
- **Honest claims**: the site says "research demo, not for real fraud decisions". Do not claim the model is ready for a bank; Part 17 explains why.

## Part 12 — Team division and Git workflow

**Five pairs, each owning one layer, with squads that finish early moving onto frontend pages and the report. This fixes Gemini's split, where one squad owned both evaluation and the whole backend.** It assumes about 10 people; with fewer, merge squads 1+4 and 2+3.

### The squads

| Squad | Weeks 1–4 (research phase) | Weeks 5–8 (product and writing phase) | Branch prefix |
| --- | --- | --- | --- |
| **S1 Data & story** | Loader, validation, dedup, split, EDA figures, `eda.json` | Overview and Method pages; report chapters 1–3; slides | `s1/` |
| **S2 Preprocessing & sampling** | Preprocessor, sampler builder for A–F, anti-leakage tests, 2D SMOTE toy engine | SMOTE Lab page; before/after resampling chart | `s2/` |
| **S3 Models & experiments** | Model builders, search spaces, CV + tuning runner, quick and full runs | Model export, presets, manifest; reproducibility check on a fresh machine | `s3/` |
| **S4 Evaluation & statistics** | Metrics, thresholds, bootstrap CIs, Wilcoxon/Friedman, curves, explanations | Results page; report results and discussion chapters | `s4/` |
| **S5 Platform** | Contract mocks, FastAPI skeleton, React shell, CI, Docker | Predict and batch endpoints, Try and Batch pages, deployment, demo rehearsal | `s5/` |

Also pick two roles (from any squad):

- **Integration lead**: merges pull requests, keeps `main` working, runs the weekly integration check.
- **Research lead**: guards the methodology (no leakage, fair comparisons), owns the decisions log and the final conclusions.

### Who depends on whom

| Squad | Needs from | Gives to | Interface (the agreed thing) |
| --- | --- | --- | --- |
| S1 | Nothing | Everyone | `load_dataset()`, `split_dataset()`, `eda.json`, synthetic sample CSV |
| S2 | S1 | S3 | `build_preprocessor()`, `build_sampler_steps(technique, params)` |
| S3 | S1, S2 | S4, S5 | `run_cell(model, technique)` → fitted pipeline + out-of-fold scores; saved models + `manifest.json` |
| S4 | S3 | S5, S1 | `evaluate()`, `comparison.json`, `curves.json`, `stats.json`, figures |
| S5 | Contract (week 1); real files later | The demo | API endpoints from Part 9 |

**The trick that lets everyone start in week 1:** S1 writes a small **synthetic** dataset generator on day 2 (fake rows with the same columns and a similar imbalance). Every squad builds and tests against it. S5 builds the website against **example JSON files** written to the contract. When real results arrive, they are just swapped in.

### Git workflow, simply

**Git** records every change. **GitHub** hosts the shared copy. A **branch** is your own parallel copy where you work without disturbing others. A **pull request (PR)** asks to merge your branch into the shared `main` branch, and lets a teammate review it first.

1. `main` is **protected**: nobody pushes to it directly.
2. For each task, create a short-lived branch from `main`: `s2/smote-sampler`.
3. Commit small, clear changes: `Add SMOTE+RUS sampler with ratio validation`.
4. Open a PR. GitHub Actions automatically runs the tests and the linter.
5. One reviewer **from a different squad** approves. This spreads knowledge across the team.
6. Merge, delete the branch, pull the new `main`.

We skip Gemini's extra `develop` branch: with short-lived branches and automatic tests, one `main` branch is simpler and is what most real teams do.

**Never commit:** `creditcard.csv`, `.env` files, large model files (use GitHub Releases or Git LFS), notebook outputs with huge data.

### Making sure everyone understands the whole project

The viva can ask anyone about anything. So:

- **Weekly 30-minute teach-back**: each squad explains what it built in 5 minutes, using this guide's words.
- **Cross-squad reviews**: reviewers must be able to explain the PR they approve.
- **Viva practice** from week 6: pairs quiz each other with Part 16, swapping topics.

## Part 13 — Step-by-step roadmap

**Build in this order: set-up → data → baseline → techniques → full experiment → freeze results → backend → frontend → deploy → write. Each phase ends with a gate: a concrete check that must pass before the next phase relies on it.** The plan assumes 8 weeks; a 4-week version follows.

&#91;embedded content: 8-week roadmap · 8 phases, 7 gates\]

Phases overlap because squads work in parallel against the synthetic data and the contract; report chapters 1–3 can be drafted from week 4 alongside the product work.

### The phases

**Phase 0 — Set-up (week 1, everyone).** Prompts 01–02 in the prompts tab.

1. Everyone installs Python 3.11, Git, VS Code and Node.js 20+, and creates a GitHub and Kaggle account.
2. Integration lead creates the repo from Prompt 01, protects `main`, turns on GitHub Actions.
3. Everyone clones it, creates a virtual environment, installs requirements, downloads `creditcard.csv` into `data/raw/`, and runs `pytest`.
4. All squads agree the contract (`docs/contracts.md`) with example JSON files.
5. S1 adds the synthetic data generator.

**Gate 0:** every member runs the tests green on their own laptop; contract merged.

**Phase 1 — Data and baseline (weeks 1–2).** Prompts 03–05.

1. S1: loading, validation, deduplication, stratified and temporal splits; EDA figures and `eda.json`. Check every number in Part 5's table.
2. S2: preprocessor (log Amount + RobustScaler), as a pipeline step.
3. S3: Logistic Regression baseline inside an imblearn pipeline with 5-fold CV.

**Gate 1:** baseline CV PR-AUC printed with mean ± std; scaler-isolation test passes.

**Phase 2 — All techniques and the runner (weeks 2–3).** Prompts 06–08.

1. S2: samplers for A–F with the shared ratio list; resampling audit (counts before and after).
2. S2: **anti-leakage tests**: SMOTE never sees validation rows; test predictions never change row counts.
3. S3: the experiment runner with quick and full modes and resumable result files.
4. S4: metrics, threshold tuning on out-of-fold scores, curves.

**Gate 2:** quick mode runs all 18 cells end to end in under about 15 minutes and writes `comparison.json`.

**Phase 3 — The real experiment (week 4).** Prompts 09–10.

1. S3 runs the full experiment, split by model across machines. S4 merges results.
2. S4: repeated CV, Wilcoxon + Holm, Friedman, bootstrap CIs, final figures.
3. **Open the test set once**, for all cells together, after every setting is frozen.
4. Research lead writes the findings in plain words in `docs/findings.md`.

**Gate 3: results frozen.** Tag the commit `results-v1`. From now on the numbers do not change unless a bug is found (and if one is, record it in the decisions log and re-run everything).

**Phase 4 — Export and backend (week 5).** Prompts 11–13.

1. S3: export chosen pipelines, `manifest.json` and presets.
2. S5: FastAPI with all endpoints, validation, tests.

**Gate 4:** `/docs` page works locally; every endpoint has a passing test.

**Phase 5 — Frontend (weeks 5–7).** Prompts 14–18.

1. S5: app shell, API client, Try and Batch pages.
2. S1: Overview and Method pages. S2: SMOTE Lab. S4: Results page.

**Gate 5:** the full 10-minute demo can be clicked through locally with real data.

**Phase 6 — Ship (week 7).** Prompts 19–21.

1. Docker, Docker Compose, deployment to Render and Vercel, security tests.

**Gate 6:** deployed site passes the demo click-through from a phone and a laptop.

**Phase 7 — Write and rehearse (week 8).** Prompt 22.

1. Report, slides, README, viva practice, two full demo rehearsals, backup video.

### MVP versus full version

|  | Minimum viable project (about 4 weeks) | Full version (about 8 weeks) |
| --- | --- | --- |
| Techniques | A–E (5) | A–F (6), optional Borderline-SMOTE |
| Models | Logistic Regression, Random Forest | + LightGBM |
| Tuning | Small fixed grid, same for all | Randomised search, equal budget |
| Statistics | Bootstrap CIs on test | + repeated CV, Wilcoxon/Holm, Friedman, CD diagram |
| Threshold | 0.5 and validation-tuned | Same + calibration |
| Extra analyses | — | Temporal split check, hour-of-day side test |
| App | Overview, Results, Try (presets), Batch | + SMOTE Lab with sliders, threshold simulator, technique comparison per payment |
| Deploy | Local Docker Compose + one hosted backend | Hosted frontend and backend, rate limits, cold-start banner |

Do the MVP first, completely, then upgrade. A finished MVP beats an unfinished full version.

## Part 14 — Testing strategy

**Tests protect two things: that the code works, and that the science is valid. The anti-leakage tests are the most important tests in the project, because a leak silently makes every result wrong.**

A **test** is a small program that checks another piece of code does what it should, automatically, every time someone changes something. We use **pytest** for Python and **Vitest** for the frontend.

| Layer | Test | What it proves |
| --- | --- | --- |
| Science | **SMOTE fold isolation**: wrap the sampler in a "spy" that records which rows it receives; assert no validation-fold row ever reaches it | Resampling never touches validation data |
| Science | **Prediction passthrough**: `predict_proba(X_test)` returns exactly one score per test row | Samplers are skipped at prediction time |
| Science | **Scaler isolation**: change test-set values wildly; the fitted scaler's medians and IQRs do not change | Preprocessing learns from training only |
| Science | **No test-set tuning**: the runner's tuning and threshold functions are never given test data (check function inputs) | Choices made without peeking |
| Science | **Reproducibility**: run the quick pipeline twice with seed 42; scores are identical | Results can be rebuilt |
| Science | **Resampling counts**: after ROS/RUS/SMOTE at ratio r, fraud ÷ legitimate ≈ r; ROS creates exact duplicates, SMOTE creates new unique rows on line segments between frauds | Samplers do what we claim |
| Metrics | Hand-computed confusion matrix matches our metric function; always-legitimate model gives 99.83% accuracy and 0 recall on real class ratio | Formulas are right |
| Data | Duplicates removed; split keeps fraud rate within a small tolerance; schema errors raised for missing columns | Data handling is correct |
| API | Every endpoint returns 200 on valid input; 422 for NaN, infinity, negative amount, missing feature, extra field; 404 unknown model; 413 big file; 400 non-CSV | The backend is robust |
| API | Batch of 10,000 rows finishes within a time limit | It is fast enough |
| Frontend | Components render loading, empty and error states; API client handles failures; `npm run build` has zero TypeScript errors | The UI does not break |
| End to end | A script starts the backend, calls each endpoint with the synthetic data and checks the shapes against the contract | The pieces fit |

Rules: every PR must pass all tests in GitHub Actions; tests use the small synthetic dataset so they run in seconds; a bug fix comes with a test that would have caught it.

## Part 15 — Presentation slides

**17 slides, about 15 minutes plus a live demo. Results slides are left as templates on purpose: Gemini's deck announced winners before any experiment was run, which is exactly what a research project must not do.**

| # | Title | Key points | Visual | What the speaker says |
| --- | --- | --- | --- | --- |
| 1 | Title | Topic, team, guide, date; built on Chawla et al. 2002 | Clean title, QR code to the live app | "We tested how to handle rare fraud fairly, and built a tool to show it." |
| 2 | The problem | Card fraud is rare but costly; missed fraud vs blocked customers | 1 red dot among 578 grey dots | "Finding fraud is finding a needle in a haystack." |
| 3 | Why accuracy lies | Always-legitimate model: 99.83% accuracy, 0% recall | Two confusion matrices side by side | "A model that does nothing looks brilliant on accuracy." |
| 4 | Metrics that matter | Precision, recall, F1/F2, FPR; PR-AUC vs ROC-AUC | The 2,000-false-alarms worked example | "ROC hides false alarms behind a huge denominator; PR does not." |
| 5 | Research question | Main question + RQ1–RQ4 | Text only, large | "Which technique helps, for which model, once compared fairly?" |
| 6 | Prior work: SMOTE (2002) | Problem with copying; SMOTE idea; SMOTE + undersampling | Paper title + one figure redrawn by us | "Copies make tiny regions; synthetic points make general ones." |
| 7 | SMOTE step by step | k neighbours, random λ, formula | 4-step drawing | Walk through one synthetic point |
| 8 | Limits of SMOTE | Bridges between clusters, noise, ignores majority, high dimensions | Two-cluster failure picture | "SMOTE assumes the gap between two frauds is fraud too." |
| 9 | Dataset | 284,807 payments, 0.17% fraud, PCA features, duplicates removed | Log-scale bar chart; Amount histogram | "We kept outliers because fraud *is* the outliers." |
| 10 | Method: no leakage | Split first; resample inside training folds only | Pipeline diagram | "The test set was locked away and opened once." |
| 11 | Method: fair comparison | 6 techniques × 3 models; same ratios, same budget, tuned thresholds | 6 × 3 grid | "The only thing that changes is the technique." |
| 12 | Results: overview | **\[Fill from results\]** best technique per model, PR-AUC with CI | Grouped bar chart with error bars | State the main finding in one sentence |
| 13 | Results: PR vs ROC | **\[Fill\]** ROC curves bunched, PR curves spread | Side-by-side curves | "Here is where the difference shows." |
| 14 | Results: thresholds and significance | **\[Fill\]** effect of tuned thresholds; which differences are statistically real | Before/after table; CD diagram | Answer RQ3 honestly, including ties |
| 15 | The application | 6 pages; architecture | Screenshot strip + architecture diagram | Hand over to the live demo |
| 16 | Limitations | Anonymised features, 2 days of data, synthetic ≠ real fraud, drift, cost not modelled | Short list | "What we cannot claim." |
| 17 | Conclusion and future work | Direct answer to the research question; next steps (cost-sensitive thresholds, drift, newer samplers) | 3 takeaway cards | One-sentence answer, then questions |

## Part 16 — Viva questions with answers

**Every team member should be able to answer all of these in their own words.** Short answers first; add detail only if asked.

### The problem and metrics

**1. What is class imbalance?** One class is far rarer than the other. Here, 0.17% of payments are fraud. Models trained to minimise total errors then mostly learn the common class.

**2. Why is accuracy insufficient?** An always-legitimate model scores 99.83% accuracy and catches no fraud. Accuracy is dominated by the true negatives.

**3. Why precision and recall?** Recall measures how much fraud we catch (money saved). Precision measures how many alerts are real (customers not annoyed, analysts not overloaded). Fraud detection needs both.

**4. Why can high recall still be bad?** Flagging everything gives 100% recall and near-zero precision. With 85,000 legitimate test payments, even a 2% false alarm rate means 1,700 blocked innocent customers.

**5. What are false positive and false negative costs?** A false negative lets fraud through: the bank loses the amount plus fees. A false positive blocks a real customer: lost sale, annoyance, analyst time. Usually a missed fraud costs more per case, which is why we also report F2.

**6. ROC-AUC vs PR-AUC?** ROC plots recall against false positive rate; FPR divides by the huge number of legitimate payments, so many false alarms barely move it. PR plots precision against recall and ignores true negatives, so false alarms show up clearly. Random PR-AUC equals the fraud rate (0.0017), not 0.5.

**7. Why PR-AUC as the main metric?** It summarises performance over all thresholds and is sensitive to false alarms on the rare class, which is what matters here.

### SMOTE

**8. Why not just duplicate fraud rows?** Copies add no new information; trees draw tiny regions around the copies and overfit. Chawla et al. showed this makes the minority regions smaller and more specific.

**9. What does SMOTE do exactly?** For a fraud point x, find its k nearest fraud neighbours, pick one (x̂), pick a random λ between 0 and 1, and create x + λ(x̂ − x): a new point on the line between them.

**10. Why nearest neighbours?** Nearby frauds are likely to share the same pattern, so points between them are plausible frauds. Far-apart frauds may be different kinds of fraud.

**11. Why only minority neighbours?** Interpolating towards a legitimate point would create "fraud" points in legitimate territory.

**12. What is k, and what if k is too large?** k is how many neighbours are candidates (default 5). Large k reaches distant frauds, possibly in another cluster, so new points can land among legitimate payments.

**13. What happens near class boundaries?** Synthetic points can spill into overlapping regions, increasing false positives. Borderline-SMOTE and cleaning methods (Tomek links, ENN) were designed for this.

**14. Can SMOTE create unrealistic samples?** Yes. They are geometrically plausible in feature space, not guaranteed to be real payments, especially since the original bank features may have included categories hidden by PCA.

**15. Why SMOTE plus undersampling?** Moderate SMOTE adds minority variety without flooding the data with fakes; undersampling then reduces the majority so the balance shifts further. The 2002 paper found this combination dominated undersampling alone in ROC space.

**16. Is SMOTE valid on PCA features?** Mechanically yes: all features are continuous, and PCA is linear, so lines in PCA space correspond to lines in the original standardised space. Realism is the limitation (Q14).

### Method and leakage

**17. What is data leakage?** Information from evaluation data reaching training, making scores look better than they would be on new data.

**18. Why must SMOTE only touch training data?** If applied before splitting, synthetic training points can be built from test frauds, so the model has partly seen the test answers. Our imblearn pipeline runs samplers only during `fit`, inside each training fold.

**19. Why scale before SMOTE?** SMOTE uses Euclidean distance; without scaling, large-range features decide who is "nearest". The scaler is fitted on training folds only.

**20. Why RobustScaler?** It uses median and IQR, which extreme fraud values barely affect, unlike mean and standard deviation.

**21. Why didn't you remove outliers?** Fraud payments are the outliers. Removing them removes the signal.

**22. Why leave out Time?** It is a position within a 2-day window; a model could memorise fraud bursts that never repeat. Converting it to hour of day requires assuming a start time nobody published.

**23. How did you make the comparison fair?** Same split, same preprocessing, same ratio choices for every resampler, same tuning budget, same threshold rule; only the technique changes.

**24. Why not just use class weights?** We tested exactly that (technique F). It changes how much each error counts instead of changing the data, is cheaper, and often performs similarly. **\[Fill in what our results showed.\]**

**25. Why these models?** Logistic Regression: simple, explainable baseline. Random Forest: modern relative of the paper's decision tree and most sensitive to copying. LightGBM: strong industry standard, testing whether good models still need help.

**26. How do you know the model generalises?** Tuning used only cross-validation inside the training set; the locked test set was evaluated once; we report bootstrap confidence intervals and repeated-CV variability; and we checked a temporal split.

**27. Are your differences significant?** We used Wilcoxon signed-rank tests with Holm correction against the baseline and a Friedman test across techniques. **\[Fill in.\]** We also note fold scores are not fully independent, so p-values are approximate.

**28. Why tune the threshold?** Resampling shifts scores upward, so 0.5 means different things for different techniques. Tuning each threshold on out-of-fold validation scores compares them fairly, and the test set is never used to choose it.

**29. Is the "fraud probability" in your app a real probability?** Not for resampled models: resampling changes the class balance the model sees, inflating scores. We call it a risk score, and calibrate it if we show a probability.

### Limits

**30. What are the limitations?** Public 2-day dataset from 2013; anonymised features with no business meaning; PCA fitted by the provider on all data; synthetic frauds are not real frauds; class overlap; no concept drift or delayed labels; equal cost for every fraud regardless of amount; offline results may not hold in production.

**31. Could a bank use your model?** No. It would need real features (customer history, merchant, device), continuous retraining against drift, cost-based thresholds, monitoring, and regulatory review. Ours is a research comparison and a demo.

## Part 17 — Limitations, open decisions, assumptions and stack

**State limitations openly in the report; it makes the work more credible, not less.** Below are the limits, the decisions to postpone until data and results exist, the assumptions this guide made, and the questions the team must answer this week.

### Limitations and future work

| Limitation | Why it matters | Possible future work |
| --- | --- | --- |
| One public dataset, 2 days, 2013, Europe | Results may not transfer to other banks, years or countries | Repeat on another fraud dataset |
| Anonymised PCA features | No business explanations; no feature engineering such as "payments per card in last hour" | Work with richer (private or synthetic) features |
| PCA fitted by the provider on all data | A small, uncontrollable leak of distribution information | None possible on this dataset; state it |
| Synthetic frauds are not real frauds | SMOTE can create impossible payments | Compare with newer generators; check synthetic realism |
| Class overlap | Some frauds look exactly like normal payments; no method can separate them | Better features |
| Concept drift and delayed labels | Real fraud changes; true labels arrive weeks later via chargebacks | Time-based retraining studies |
| Equal cost per error | A missed €2 fraud counts the same as a missed €5,000 fraud | Amount-aware, cost-sensitive thresholds |
| High-dimensional distances | Nearest neighbours get less meaningful in 29 dimensions | SMOTE in a learned lower-dimensional space |
| Offline benchmark | Lab results may not match live performance | Shadow deployment studies (out of scope) |

### Decisions NOT to make until we see data or results

1. **The default ratio for any resampler**: let tuning pick from the shared list.
2. **The "champion" model for the app**: choose from cross-validation results only, and if two are statistically tied, prefer the simpler or cheaper one and say so.
3. **The app's default threshold**: use the validation-tuned threshold of the chosen model.
4. **Whether calibration is worth showing**: decide after seeing how inflated the scores are.
5. **Whether to add Borderline-SMOTE**: only if the core 18 cells finish early.
6. **The headline sentence of the report**: write it after results are frozen, never before.

### Assumptions this guide made

| Assumption | If wrong, change |
| --- | --- |
| The dataset is the Kaggle/ULB `creditcard.csv` | Update `configs/dataset.yaml`; categorical columns would need SMOTE-NC |
| About 10 team members | Merge squads (Part 12) |
| About 8 weeks | Use the MVP plan (Part 13) |
| Laptops or free Google Colab, no GPU needed | Fine either way; nothing here needs a GPU |
| Free hosting (Vercel + Render free) | Paid hosting removes cold starts and memory limits |
| `Amount` is in euros | Widely assumed, not formally documented; write "amount (believed EUR)" |
| The instructor wants both a report and a live app | Adjust effort split if only one is graded |

### Questions the team must answer this week

- [ ] Exact deadline and what is graded (report, app, viva, slides, code), with weightings
- [ ] Has the instructor fixed the dataset, or can we choose?
- [ ] Any required report format or length?
- [ ] Final team size and who is in each squad
- [ ] Who has a laptop that can run the full experiment, and who will use Colab?
- [ ] Is a live deployment required, or is a local demo acceptable?

### Recommended technology stack

| Layer | Choice | Why this and not something else |
| --- | --- | --- |
| Language | Python 3.11 (research, backend); TypeScript (frontend) | Standard for ML; types catch frontend bugs |
| ML | scikit-learn, imbalanced-learn, LightGBM | imbalanced-learn's pipeline is what makes leakage-free resampling easy; LightGBM is fast and memory-light |
| Statistics | SciPy (Wilcoxon, Friedman), a small CD-diagram helper | Standard, free, well documented |
| Explanations | LightGBM built-in SHAP values on the server; `shap` library in notebooks | Exact and fast without a heavy server dependency |
| Plots | Matplotlib (report figures); Recharts (website) | Matplotlib for print quality, Recharts for interactive web charts |
| Config | YAML files | Human-readable settings outside the code |
| Backend | FastAPI + Pydantic + Uvicorn | Automatic validation and a free test page at `/docs` |
| Frontend | React + Vite + TypeScript + Tailwind + Recharts (+ TanStack Query) | Modern, fast, widely taught, good for dashboards |
| Tests | pytest, Vitest | The standard tools for each language |
| Quality | ruff (Python linter), GitHub Actions | Free automatic checks on every PR |
| Packaging | Docker, Docker Compose | Same environment everywhere |
| Hosting | Vercel (frontend), Render free (backend) | Free; see Part 11 for limits |
| Experiment tracking | JSON/CSV results + git hash; MLflow optional locally | Enough for 18 cells without hosting a server |

### Sources

- N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, [SMOTE: Synthetic Minority Over-sampling Technique](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/chawla2002.html), JAIR 16, 2002
- [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group, Kaggle); cite Dal Pozzolo et al., IEEE SSCI 2015
- [Credit Fraud: Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets) (Kaggle notebook)
- J. Davis and M. Goadrich, "The Relationship Between Precision-Recall and ROC Curves", ICML 2006
- [Render: Deploy for Free](https://render.com/docs/free) (free-tier limits)
- [Hugging Face: Spaces Overview](https://huggingface.co/docs/hub/en/spaces-overview) (paid plan required for Docker Spaces)
