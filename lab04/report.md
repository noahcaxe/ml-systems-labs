# Lab 4: Data pipeline and quality

## 1. Problem definition

**Task.** Predict whether a person's annual income is above 50 000 USD from 14 census attributes.
This is binary classification on tabular data with mixed types: 6 numeric features (age, fnlwgt,
education-num, capital-gain, capital-loss, hours-per-week) and 8 categorical ones (workclass,
education, marital-status, occupation, relationship, race, sex, native-country). The dataset is
UCI Adult, 48 842 records from the 1994 US census, loaded with
`fetch_openml(name='adult', version=2, as_frame=True)`.

**Output.** `income ∈ {<=50K, >50K}`. The positive class `>50K` makes up 23.9 % of the rows.

**Success metrics.** F1 of the positive class is the main metric, and accuracy is reported next to
it. With a 76/24 class split, predicting `<=50K` for everyone already gives 76 % accuracy, so
accuracy alone can hide a model that never finds the minority class. F1 combines precision and
recall on `>50K` and is 0 for that trivial model. Both metrics are measured once on a held-out
30 % test split and with 5-fold cross-validation on train.

**Constraints.**
1. *No leakage.* The test split is created before any statistic is learned. Imputation values,
   clip bounds, deduplication and every model fit see train only, and the test rows keep the
   values they had after the split.
2. *Reproducible.* One command rebuilds every artefact from the public source. Every parameter
   lives in `config.yaml`, and randomness is fixed by one seed (42). A SHA256 of the processed data
   proves that a second run produced the same bytes.
3. *Auditable.* Validation reports violations and never drops rows silently. Data problems are
   written to files that can be reviewed.

The goal of this lab is the data pipeline, not the model. LogisticRegression is used only as a
baseline to measure whether the engineered features help.

## 2. Method

**Environment.** The same pinned environment as Labs 1–3: Python 3.11.8, pandas 2.2.2,
NumPy 1.26.4, scikit-learn 1.5.0, joblib 1.4.2, PyYAML 6.0.3 (added to `requirements.txt` in
this lab), Windows 11 laptop.

**One command** (from the repository root):

```bash
python lab04/src/run_pipeline.py          # all steps, writes results/ and data/adult/
python lab04/src/run_pipeline.py --check  # same, and exits with 1 if any dataset hash changed
```

The first run downloads the dataset into `data/openml/`, and every later run reads that cache.
A full run takes about 17 s.

**Pipeline.**

```
fetch_openml ──► ingest ──► validate ──► split 70/30 ──► drop duplicates ──► fit preprocessor ──► transform train, test
  (raw)        (no change,  (rules from   (stratified,    (train only)        (train only:          ──► baseline
               shape, hash) config, all   seed 42)                            median, mode,         ──► label audit
                            rows kept)                                        1/99 pct clip)        ──► SHA256 + run_config.yaml
```

**Code** (`src/`):

| File | Step | Main outputs |
|---|---|---|
| `common.py` | paths, config, raw loading, canonical CSV + SHA256 | |
| `ingest.py` | 2 | `ingestion_log.txt`, `data/adult/raw.csv` |
| `validate.py` | 3 | `validation_report.md`, `validation_checks.csv`, `validation_violations.csv` |
| `features.py` | preprocessing transformers: `PercentileClipper`, `DerivedFeatures`, `build_preprocessor` | |
| `clean.py` | 4 | `preprocessor.joblib`, `cleaning_log.txt`, `fitted_stats.csv`, `data/adult/processed/{train,test}.csv` |
| `baseline.py` | 5 | `baseline.csv`, `baseline.txt` |
| `label_quality.py` | 6 | `label_quality.txt`, `label_conflicts.csv`, `label_audit_sample.csv` |
| `run_pipeline.py` | 7 | `run_config.yaml` |

`config.yaml` holds the schema, ranges, allowed categories, the education → education-num
dictionary, split, cleaning and audit parameters. No step has a hard-coded threshold.
`data_card.md` documents the dataset.

The processed data (`data/adult/`) is not committed because `data/` is in `.gitignore`. It is
rebuilt by the command above, and its hash is stored in `results/run_config.yaml`. Loading
`preprocessor.joblib` needs `lab04/src` on the import path, because the pickle refers to the
classes in `features.py`.

## 3. Results

### 3.1 Ingestion (step 2)

From [`ingestion_log.txt`](results/ingestion_log.txt): **48 842 rows, 15 columns**, the same as the
expected shape in `config.yaml` (an `assert`). The frame is written to `data/adult/raw.csv` exactly
as served. Its canonical SHA256 is computed before and after writing and is the same both times,
`8f970e20…adddc`, so ingestion does not modify the data.

Two points about the OpenML v2 copy that the later steps account for:
- The label column is called **`class`**, not `income`. Validation renames it.
- The `'?'` missing marker of the UCI files is **already decoded to NaN**: workclass 2 799,
  occupation 2 809, native-country 857 cells. To follow the null policy of the manual (`'?'` means
  missing and is an allowed category), validation restores NaN to `'?'`. The preprocessor turns
  `'?'` back into NaN before imputation.

### 3.2 Validation (step 3)

Full report: [`validation_report.md`](results/validation_report.md). Validation runs on all
48 842 rows **before** the split. This does not leak, because every check is a fixed rule from
`config.yaml` (types, ranges, allowed sets, the data dictionary) and nothing is learned from the data.

| Check | Rule | Result |
|---|---|---|
| Columns | the 15 schema columns, no extra | PASS |
| Types | 6 `int` columns are integers with no NaN; 9 `category` columns are strings | PASS (15/15) |
| Allowed values | every category in the UCI list (with `'?'`; the label without `'?'`) | PASS (9/9) |
| Ranges | age 17–90, hours 1–99, capital-gain 0–99 999, capital-loss 0–4 356 | PASS, 0 violations |
| Dictionary | education-num equals the code of education (Preschool = 1 … Doctorate = 16) | PASS |
| Label | income present and not `'?'` | PASS |
| Missing (`'?'`) | allowed, counted | INFO: 6 465 cells in 3 620 rows (7.4 % of rows) |
| Exact duplicates | counted | INFO: 52 rows |

The real data has **0 FAIL and 0 WARN**. The ranges in the manual are exactly the observed min and max
of the dataset, so no row falls outside them. A check that never fires proves little, so the same
checks also run on a corrupted copy of the first 200 rows with 8 injected errors: age 12, hours 0,
capital-loss 5 000, capital-gain −1, age "forty", workclass "Self-employed", a wrong education-num
and a missing label. **Every injected error is caught** (table at the end of the report). Policy:
structural errors (FAIL) stop the pipeline; value errors (WARN) go to
`validation_violations.csv` row by row and are kept. **Validation drops 0 rows.**

### 3.3 Cleaning and transformation (step 4)

From [`cleaning_log.txt`](results/cleaning_log.txt) and [`fitted_stats.csv`](results/fitted_stats.csv).

| | Rows | `>50K` share |
|---|---|---|
| Train after stratified split | 34 189 | 23.94 % |
| − exact duplicates (train only) | −24 | |
| **Train, processed** | **34 165** | 23.94 % |
| **Test, processed** (untouched) | **14 653** | 23.93 % |

Statistics learned on train and applied to both splits:

| Column | Impute | Clip low (1st pct) | Clip high (99th pct) | Values clipped train / test |
|---|---|---|---|---|
| age | median 37 | 17 | 74 | 302 / 143 |
| fnlwgt | median 178 272 | 27 035.2 | 509 629 | 683 / 272 |
| education-num | median 10 | 3 | 16 | 240 / 88 |
| capital-gain | median 0 | 0 | 15 024 | 255 / 136 |
| capital-loss | median 0 | 0 | 1 980 | 340 / 151 |
| hours-per-week | median 40 | 8 | 80 | 537 / 236 |
| workclass / occupation / native-country | mode Private / Prof-specialty / United-States | | | imputed train 1 923 / 1 933 / 590, test 873 / 873 / 267 |

Other categoricals use their train mode too (HS-grad, Married-civ-spouse, Husband, White, Male),
but they have no missing values. The numeric columns have no missing values either, so the median
imputer is a guard for future data and changes nothing here.

**Leakage guards** (asserts in `clean.py`, all pass):
- train and test indices are disjoint, and the split happens before any fit;
- every learned median, mode and percentile equals the value computed directly on the train
  frame;
- the test frame keeps exactly its 14 653 rows in the original order, and nothing is dropped or
  deduplicated;
- after transformation, neither split has a NaN.

**What leakage would have changed.** The log also fits the same preprocessor on all 48 842 rows.
The full-data fit would move the fnlwgt median (178 144 instead of 178 272) and clip bounds
(27 166 / 509 484), and the capital-loss upper bound (2 001 instead of 1 980). The other values stay
the same. The effect on Adult is small because 34 k rows estimate these statistics well, but it is
not zero: with the leaky fit, the test set would partly define its own cleaning rules. On a small
dataset, or with target-dependent steps such as target encoding or feature selection, the same
mistake inflates the test score.

**Duplicates.** Of the 52 exact duplicates in the full data, 24 fell inside train (removed), 5
inside test (kept, as the manual requires), and the rest have one copy in each split. **24 test
rows have an exact twin in train** (0.16 % of the test set). Removing them would mean editing
the test set, so they are kept and reported instead. Their effect on the metrics is at most
±0.16 points.

The fitted transformer (`Pipeline`: `'?'` → NaN → `ColumnTransformer[median + PercentileClipper |
mode]` → `DerivedFeatures`) is saved with `joblib.dump` to **`results/preprocessor.joblib`**. A
reload in a fresh process reproduces the processed test file, with a maximum difference of 1e-17
from the 10-digit CSV format.

### 3.4 Derived features and baseline (step 5)

`capital_net = capital-gain − capital-loss` and `hours_per_age = hours-per-week / (age − 16)`
(age ≥ 17, so the denominator is ≥ 1). Both are computed from clipped values inside the
preprocessor, so they carry no information about the test set. Model: one-hot categoricals +
standardised numerics → `LogisticRegression(max_iter=2000)`, fitted on train. From
[`baseline.csv`](results/baseline.csv):

| Variant | Features | Test accuracy | Test F1 | CV F1 (5-fold, mean ± std) | Paired ΔF1 vs without | Folds better |
|---|---|---|---|---|---|---|
| Without derived | 105 | 0.8540 | 0.6601 | 0.6552 ± 0.0118 | — | — |
| + capital_net | 106 | 0.8542 | 0.6610 | 0.6554 ± 0.0124 | +0.0002 | 2 / 5 |
| + hours_per_age | 106 | 0.8576 | 0.6694 | 0.6615 ± 0.0126 | +0.0063 | **5 / 5** |
| **+ both** | 107 | **0.8578** | **0.6696** | 0.6618 ± 0.0129 | +0.0065 | **5 / 5** |

With both features, test accuracy rises by **+0.38 points** and test F1 by **+0.95 points**.
`hours_per_age` accounts for all of the gain, and `capital_net` adds nothing. The fold-to-fold spread
(±0.012) is larger than the gain, but all variants use the same folds, so the per-fold difference is
the fair comparison. `hours_per_age` improves F1 in all 5 folds, while `capital_net` improves it in
2 of 5, which is noise.

The reason is the model. LogisticRegression is linear in its inputs, and `capital_net` is a linear
combination of two columns it already has. After standardisation, the model can represent
`w·(gain − loss)` exactly with the existing weights, so the new column brings no new information.
`hours_per_age` is a **ratio** (hours worked per year of working age) that no linear combination of
age and hours can express, so it adds a real non-linearity. A tree model can learn such interactions
from age and hours on its own, so it would probably gain less from either feature.

### 3.5 Label quality (step 6)

From [`label_quality.txt`](results/label_quality.txt). Adult's label is objective (reported
income), so the relabel-and-kappa protocol for subjective labels does not apply. The audit has two
parts.

**(a) Duplicate-conflict audit.** Rows that are identical in all 14 features (fnlwgt included) but
have different income labels: **5 groups, 10 rows (0.020 %)** in the full data, and 2 groups (4 rows)
inside train. Each group is one `<=50K` row and one `>50K` row
([`label_conflicts.csv`](results/label_conflicts.csv)). Because fnlwgt is a 6-digit sampling
weight, two different people rarely share all 14 values, so these are most likely the same record
with a label error, or two respondents that the features cannot tell apart. Either way, a noise rate
of 0.02 % is far too small to matter for the model, so the rows are kept and documented.

**(b) 50 random rows against the data dictionary** (`random_state=42`,
[`label_audit_sample.csv`](results/label_audit_sample.csv)). Each row was checked with 14 hard rules
(ranges, allowed values, education-num = code(education)) and 6 plausibility rules (pairs of
allowed values that contradict each other), and then read through by hand.

| Rule | 50-row sample | Full data |
|---|---|---|
| All 14 hard rules (ranges, categories, education code) | 0 | 0 |
| relationship Husband ⇒ sex Male | 0 | **1** |
| relationship Wife ⇒ sex Female | 0 | **3** |
| Husband/Wife ⇒ married | 0 | 0 |
| workclass Never-worked ⇒ 0 hours per week | 0 | **10** |
| Bachelors or higher ⇒ age ≥ 21 | 0 | **5** (18–20-year-old Bachelors/Masters) |
| not both capital-gain > 0 and capital-loss > 0 | 0 | 0 |

**No impossible values are in the 50-row sample, and none of its rows belongs to a conflict group.**
The hand read-through found two rows that are odd but possible: row 21063 (workclass and
occupation `'?'` but 60 hours/week and a capital gain of 9 386), and row 23881 (a 25-year-old who is
Married-civ-spouse and Own-child). On the full data, 19 rows break a plausibility rule (listed in
`label_quality.txt`). The Husband/Female and Wife/Male rows look like data-entry errors in
`relationship` or `sex`, and the Never-worked rows with 4–40 hours look like an error in
`workclass`. These rows are flagged, not deleted, for the same reason as in validation.

### 3.6 Hash, configuration and reproducibility (step 7)

`run_pipeline.py` writes [`results/run_config.yaml`](results/run_config.yaml). It contains
`config.yaml` as used, the SHA256 values below, sizes, baseline metrics and package versions. Each
hash is taken over a canonical CSV: fixed column order, LF line ends, `%.10g` floats, no index.

| Hash | SHA256 |
|---|---|
| raw frame | `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc` |
| processed train | `d85adda24d31f462e37ef41ebd6e5aa6fb6b707442f6dcb37b38fc624a68c1a8` |
| processed test | `7825db706aa7b4c51ab1ec2d08fc12afe2401364f1ca9ac70d53cdc68006b5e0` |
| **processed dataset** (train ‖ test) | **`569ae16cffb25ccdf4471b07bf9747c95618fdfd834f81ac6360250621a9dacf`** |

**Reproducibility test.** I ran the pipeline, then ran `--check`. It reported "same hashes as the
previous run" and exited with 0. Every committed result file except the timestamp line in
`ingestion_log.txt` was also byte-identical between the runs. As a negative test, I changed one
character of the stored train hash, and `--check` reported `CHANGED since the previous run:
['processed_train_sha256']` and exited with 1. The data card is [`data_card.md`](data_card.md):
source, CC BY 4.0 license, size, known biases, privacy.

## 4. Discussion: why the order of the steps matters (Ch. 5–6)

The data-engineering chapters treat the pipeline as part of the system: data is collected, ingested,
validated, processed, labelled and stored in stages, and an error in an early stage spreads silently
into every model trained later. This lab makes each stage leave evidence:

- **Ingestion keeps the raw data immutable** and records its hash. Every later stage can then be
  rerun from a known starting point, and a change in the source (OpenML re-encoding `'?'` as NaN,
  for example) shows up as a different hash, not as a quietly different model.
- **Validation is a gate.** The schema is written down in `config.yaml` (types, ranges, allowed
  values, dictionary rules), and its checks are tested on corrupted data. Clean data passes today,
  and the checks will still work when a future batch breaks the schema. Reporting instead of
  dropping keeps the decision visible.
- **Split first, then learn.** Imputation, clipping and deduplication are learned transformations,
  so they belong to the model, not to the data. Fitting them on train and saving them in one
  `preprocessor.joblib` applies the same transformation to test data and later to new data, so
  training and serving stay consistent.
- **Label quality limits model quality.** Feature cleaning cannot fix wrong labels. The conflict
  audit and dictionary rules estimate how noisy the labels are (≈0.02 % conflicting, a few dozen
  implausible rows), which bounds what a better model could gain from cleaner data.
- **Documentation and hashes make the dataset a versioned artefact.** The data card says what the
  data is and what it should not be used for (§ biases, privacy). The hash says exactly which
  version a model was trained on.

## 5. Limitations

- The manual's ranges equal the observed min/max of Adult, so the range checks cannot fail on this
  dataset. They protect against future data, and the self-test only shows that they work.
- Clipping to the 1st/99th percentile follows the manual, but it is a crude choice for
  capital-gain and capital-loss: more than 90 % of their values are 0, so the 99th percentile cuts
  all large gains to 15 024. A log transform or a "has gain" flag would keep more information.
- Validation and the label audit run on the full dataset. This is safe because they are fixed rules
  and nothing is fitted. If a validation rule were learned from the data (for example, a
  distribution-drift threshold), it would have to be fitted on train only.
- The baseline uses one model and one split. Its default 0.5 threshold is not tuned for F1, so
  higher F1 is possible with the same features.
- The 50-row manual check is small. It finds nothing because the problem rows make up <0.05 % of
  the data, and the full-data rule counts are what show them.
- The SHA256 is reproducible on this pinned environment. Another pandas or scikit-learn version
  could change float formatting or tie-breaking and give a different hash for the same pipeline.
  The versions are stored in `run_config.yaml` for that reason.
