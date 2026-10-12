# Lab 1: Environment and first system measurements

## 1. Goal

Set up a reproducible, version-pinned Python environment, train two baseline classifiers on the
Breast Cancer Wisconsin (Diagnostic) dataset, measure their system cost (training time, single-sample
inference latency, model size, peak memory) and decide on which deployment targets (Cloud, Edge,
Mobile, TinyML) each model could run.

## 2. Method

**Environment.** Python 3.11.8 (CPython) in an isolated `.venv`, packages pinned exactly as required
by the manual. The full list of installed versions with a match check is in
[`results/versions.txt`](results/versions.txt) (0 mismatches). Hardware: AMD64 laptop CPU
(AMD Family 25 Model 116), Windows 11, no GPU used.
The pinned stack contains a conflict (`mlflow==2.14.1` requires `pyarrow<16`, the manual requires
`pyarrow==16.1.0`); it was resolved by installing `pyarrow` separately without dependency checks, as
documented in the root `README.md`.

**Data and models.** `load_breast_cancer(return_X_y=True)`: 569 samples, 30 numeric features,
binary target. Stratified 70/30 split with `random_state=42` (398 train / 171 test). All seeds
(`random`, `numpy`, `torch`, `torch.cuda`) set to 42. Two baselines with the exact settings from the
manual and no preprocessing:

- `LogisticRegression(max_iter=1000, random_state=42)`
- `RandomForestClassifier(n_estimators=100, random_state=42)`

Re-running the training script produces a byte-identical accuracy file.

**Measurements** (all timings with `time.perf_counter`):

| Metric | Protocol |
|---|---|
| Training time | 1 warm-up fit, then 5 fits on a fresh clone of the model; median reported |
| Inference latency | 1 sample (`X_test[:1]`), 1 warm-up, 100 repeats; median reported (min/max kept) |
| Model size | `joblib.dump` of the fitted model; file size in bytes and KB |
| Peak memory | `memory_profiler.memory_usage(max_usage=True)` sampling RSS every 1 ms, each (model, phase) in a **fresh subprocess** so models cannot affect each other |
| Runtime footprint | RSS of a fresh process after each import layer, and peak RSS of a lean serving process (numpy + scikit-learn + joblib only, model loaded from disk, 100 requests) |

Each model is saved to its own file (`results/model_logistic_regression.joblib`,
`results/model_random_forest.joblib`) instead of a single `results/model.joblib`, because there are
two models to measure. Raw timings of every run are in [`results/timings_raw.csv`](results/timings_raw.csv).

**Deployment fit.** Each model is checked against the budgets from the manual on three criteria:
median latency, model file size and peak memory. Memory budgets for Edge and Mobile are ranges of
device RAM, so a model *fits* a class only if its peak memory is within the lower bound (it runs on
every device of the class) and fits *partially* if it only fits under the upper bound. Units are
read as binary (1 MB = 1 MiB). Memory is evaluated in three scenarios:

1. **measured**: the Python process used in this lab (it imports `torch` for the mandatory seeds);
2. **lean Python**: measured peak RSS of the minimal serving process described above;
3. **native (estimate)**: the model ported out of Python, memory approximated by the model file
   size. This is an estimate, not a measurement, and latency is not re-measured for it.

Scripts in `src/`: `print_versions.py`, `train_baselines.py`, `measure.py`,
`runtime_footprint.py`, `deployment_fit.py` (run in this order).

## 3. Results

### Accuracy and system cost

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Test accuracy | **0.9474** | 0.9357 |
| Training time, median of 5 (s) | 0.3377 | **0.1124** |
| Inference latency, median of 100 (ms) | **0.0316** | 1.6019 |
| Inference latency, min / max (ms) | 0.0300 / 0.0638 | 1.5199 / 127.5668 |
| Model size (bytes) | **1 055** | 290 889 |
| Model size (KB) | **1.03** | 284.07 |
| Peak RSS during training (MiB) | 268.25 | 267.91 |
| Peak RSS during inference (MiB) | 266.86 | 268.00 |
| Memory added by training (MiB) | 0.93 | 0.61 |
| Memory added by inference (MiB) | 0.14 | 0.32 |

Sources: [`baseline_accuracy.csv`](results/baseline_accuracy.csv),
[`measurements.csv`](results/measurements.csv).

### Where the memory goes

| Process state | RSS (MiB) |
|---|---:|
| Python interpreter only | 19.2 |
| + numpy, scikit-learn | 102.8 |
| + pandas, joblib | 142.6 |
| + torch | 266.0 |
| Lean serving, Logistic Regression (peak) | 98.4 |
| Lean serving, Random Forest (peak) | 106.5 |

Sources: [`runtime_rss.csv`](results/runtime_rss.csv),
[`lean_serving_rss.csv`](results/lean_serving_rss.csv).

### Deployment fit

Budgets: Cloud ≥1 GB / ≤100 ms / ≤500 MB; Edge 256–1024 MB / ≤50 ms / ≤50 MB;
Mobile 64–256 MB / ≤20 ms / ≤10 MB; TinyML ≤256 KB / ≤10 ms / ≤100 KB.

| Model | Memory scenario | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|---|
| Logistic Regression | measured (with torch) | FITS | PARTIAL | NO | NO |
| Logistic Regression | lean Python | FITS | FITS | PARTIAL | NO |
| Logistic Regression | native (estimate) | FITS | FITS | FITS | FITS |
| Random Forest | measured (with torch) | FITS | PARTIAL | NO | NO |
| Random Forest | lean Python | FITS | FITS | PARTIAL | NO |
| Random Forest | native (estimate) | FITS | FITS | FITS | NO |

Every failure except one is caused by memory. The only other limiting factor is the Random Forest
file size (284 KB) against the TinyML limit of 100 KB. Latency is far below every budget: the slowest
median (1.60 ms) is 6× under the strictest limit (TinyML, 10 ms). Full per-check table:
[`deployment_fit.csv`](results/deployment_fit.csv), summary: [`deployment_fit.md`](results/deployment_fit.md).

**Where each model could run:**

- **Logistic Regression**: Cloud and Edge as a Python service; Mobile only on devices with more
  than ~100 MB of free RAM, or after porting out of Python; TinyML only as a native port (30 weights
  plus an intercept, about 1 KB).
- **Random Forest**: Cloud and Edge as a Python service; Mobile under the same conditions as above;
  **not TinyML**, because the model itself (284 KB) is almost 3× over the size budget, regardless of
  the runtime.

## 4. Conclusions

1. **For a small table dataset, the simple model is the better choice.** Logistic Regression wins
   against Random Forest in almost every test. It is a little more accurate (0.9474 vs 0.9357),
   about 50 times faster when it answers (0.03 ms vs 1.60 ms) and about 275 times smaller
   (1 KB vs 284 KB). Only its training is slower, because the data is not scaled and the solver
   does not finish in 1000 steps.

2. **The limit is the memory of Python and its libraries, not the model.** Both models are fast
   enough for every platform. By size, both fit Cloud, Edge and Mobile, and Logistic Regression
   also fits TinyML; only Random Forest is too big for TinyML (284 KB, the limit is 100 KB). The
   models themselves use less than 1 MB of memory, but the Python process uses about 100 MB with
   only numpy and scikit-learn, and about 268 MB with torch. Because of this, in Python neither
   model fits a small phone (64 MB) or a TinyML device; on Mobile they work only on bigger phones.

3. **We can trust the measurements only if we follow the right method.** This means a warm-up run,
   many repeats, the median instead of one run, and a separate process for each memory test. For
   example, one Random Forest run took 127.6 ms, which is 80 times more than the median. If we used
   only this one run, the model would fail the Edge, Mobile and TinyML budgets.
