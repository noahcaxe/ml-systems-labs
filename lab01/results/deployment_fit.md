# Deployment fit

## Verdict matrix

| model | scenario | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|---|
| logistic_regression | measured | FITS | PARTIAL | NO | NO |
| logistic_regression | lean_python | FITS | FITS | PARTIAL | NO |
| logistic_regression | native_est | FITS | FITS | FITS | FITS |
| random_forest | measured | FITS | PARTIAL | NO | NO |
| random_forest | lean_python | FITS | FITS | PARTIAL | NO |
| random_forest | native_est | FITS | FITS | FITS | NO |

## Why not FITS

| model | scenario | platform | verdict | reason |
|---|---|---|---|---|
| logistic_regression | measured | Edge | PARTIAL | mem (partial) |
| logistic_regression | measured | Mobile | NO | mem |
| logistic_regression | measured | TinyML | NO | mem |
| logistic_regression | lean_python | Mobile | PARTIAL | mem (partial) |
| logistic_regression | lean_python | TinyML | NO | mem |
| random_forest | measured | Edge | PARTIAL | mem (partial) |
| random_forest | measured | Mobile | NO | mem |
| random_forest | measured | TinyML | NO | size, mem |
| random_forest | lean_python | Mobile | PARTIAL | mem (partial) |
| random_forest | lean_python | TinyML | NO | size, mem |
| random_forest | native_est | TinyML | NO | size, mem |
