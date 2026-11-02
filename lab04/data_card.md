# Data card: UCI Adult (Census Income), as processed in Lab 4

## Source

| | |
|---|---|
| Dataset | Adult, also known as Census Income |
| Original | UCI Machine Learning Repository, B. Becker and R. Kohavi (1996), DOI [10.24432/C5XW20](https://doi.org/10.24432/C5XW20) |
| Origin of the records | Extracted by Barry Becker from the 1994 US Census Bureau database (Current Population Survey). Records were kept only if age > 16, gross income > 100 USD, hours worked > 0 and final weight > 1 |
| Access used here | `sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True)`, OpenML data id 1590 (<https://www.openml.org/d/1590>), the 32 561 + 16 281 UCI train and test files merged |
| Raw frame SHA256 | `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc` (canonical CSV, see `src/common.py`) |
| Processed SHA256 | train `d85adda2…`, test `7825db70…`, both `569ae16c…`; full values in [`results/run_config.yaml`](results/run_config.yaml) |

## License

The UCI Machine Learning Repository publishes it under **Creative Commons Attribution 4.0 International
(CC BY 4.0)**: anyone may share and adapt it, including commercially, as long as they credit the source
(the citation above). OpenML lists the same data with licence "Public".

## Size and structure

| | Raw | Processed train | Processed test |
|---|---|---|---|
| Rows | 48 842 | 34 165 | 14 653 |
| Columns | 15 (14 features + label) | 18 | 18 |
| Positive class `>50K` | 11 687 (23.93 %) | 23.94 % | 23.93 % |

- **Numeric (6):** age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.
- **Categorical (8):** workclass (8 levels), education (16), marital-status (7), occupation (14),
  relationship (6), race (5), sex (2), native-country (41).
- **Label:** income, `<=50K` / `>50K`. OpenML names it `class`; the pipeline renames it.
- **Missing values:** `'?'` in the original files (OpenML v2 delivers them as NaN). 6 465 cells in
  3 620 rows: workclass 2 799, occupation 2 809, native-country 857. Every row with missing
  workclass also has missing occupation.
- **Duplicates:** 52 exact duplicate rows. 5 groups (10 rows) have identical features but different labels.
- **Processed columns:** `row_id` (row of the raw frame), the 14 features after imputation and
  clipping, `capital_net`, `hours_per_age`, `income`.

## How it was processed

The pipeline is `python lab04/src/run_pipeline.py`. It validates the data, makes a stratified 70/30
split (seed 42), drops exact duplicates from train only (24 rows), and learns the imputation values
(median, mode) and the 1st/99th-percentile clip bounds on train only, then applies them to both
splits. All parameters are in [`config.yaml`](config.yaml), the fitted transformer is in
`results/preprocessor.joblib`, and the evidence is in `report.md`.

## Known biases and limitations

- **Old data.** The data is from 1994, and the 50K USD threshold is in 1994 dollars, not adjusted
  for inflation. Neither the income distribution nor the occupation categories describe today's
  labour market.
- **Unbalanced sensitive groups.** 66.9 % of rows are men and 85.5 % are White. Only 0.8–3.1 % of
  rows belong to each of Amer-Indian-Eskimo, Other and Asian-Pac-Islander. 89.7 % of people were
  born in the United States, and 1.9 % in Mexico, the next largest country.
- **Different base rates.** The share of `>50K` is 30.4 % for men and 10.9 % for women, and 25.4 %
  for White and 12.1 % for Black people. A model trained on this data learns these historical gaps,
  and sex, race and relationship (Husband/Wife) act as direct or proxy features. Accuracy alone will
  hide unequal error rates, so fairness should be measured per group before any use.
- **Population coverage.** The extraction rules leave out people under 17, people with no income and
  people who did not work. `fnlwgt` is a census sampling weight (how many people a row represents),
  not a property of the person, and the pipeline uses it as an ordinary feature.
- **Top-coding and clipping.** Some values are capped in the source: age at 90 (55 rows),
  capital-gain at 99 999 (244 rows), hours-per-week at 99 (137 rows). The pipeline also clips to the
  train 1st/99th percentiles (for example, capital-gain to at most 15 024), which hides the very high
  values.
- **Label noise.** Income is self-reported. 10 rows sit in 5 groups with identical features and
  conflicting labels, and a few rows contradict themselves (1 "Husband" who is Female, 3 "Wife" who
  are Male, 10 "Never-worked" with positive hours, 5 people under 21 with a Bachelor's or higher). These rows are
  kept and listed in `results/label_quality.txt`.
- **Imputation.** Missing categorical values are filled with the train mode (Private,
  Prof-specialty, United-States). This puts people with unknown employment into the biggest group,
  although 63 % of them are under 25 or 60 and older, so they are likely students and retirees.

## Privacy

- There are no direct identifiers (name, address, ID number). The Census Bureau released the records
  in anonymised form, and the dataset is public.
- Quasi-identifiers remain: age, sex, race, native-country, marital-status, relationship,
  occupation and education. Rare combinations (e.g. a 90-year-old from a small country with a rare
  occupation) could narrow down a real person if linked with other data.
- Several columns are special-category or sensitive data under GDPR-style rules: race, sex, marital
  status and income. Use the data for research and teaching. Do not use a model trained on it to make
  decisions about individuals (credit, hiring, insurance).
- The processed files (`data/adult/`) are not committed. They are rebuilt from the public source by
  the pipeline, and the SHA256 in `results/run_config.yaml` proves the rebuilt copy is identical.
