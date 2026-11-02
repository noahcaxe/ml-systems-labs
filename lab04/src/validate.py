"""Step 3: schema, type, category, range, missing-value, duplicate and label checks.

Runs on the full raw dataset (the checks use fixed rules from config.yaml, nothing is fitted,
so no information moves from test to train). Every violation is reported row by row; no row is
dropped. A structural failure (missing column, non-integer numeric column) stops the pipeline;
value-level violations are reported and left for the cleaning step to handle.
The same checks are also run on a deliberately corrupted copy to show that each one fires.

Outputs: results/validation_report.md, results/validation_checks.csv,
         results/validation_violations.csv
"""

import numpy as np
import pandas as pd

from common import RESULTS_DIR, fetch_raw, load_config, md_table, standardise, write_text


def run_checks(df, cfg):
    """Return (checks, violations): one row per check, one row per violating cell."""
    checks, violations = [], []
    miss = cfg["missing_marker"]

    def add(check, column, bad, severity, note=""):
        bad = bad if isinstance(bad, pd.Series) else pd.Series(bad, index=df.index)
        idx = df.index[bad.fillna(True).to_numpy(bool)]
        checks.append({"check": check, "column": column, "violations": len(idx),
                       "status": "PASS" if len(idx) == 0 else severity, "note": note})
        for i in idx:
            violations.append({"row": int(i), "check": check, "column": column,
                               "value": df.at[i, column] if column in df else None})

    expected = list(cfg["schema"])
    missing_cols = [c for c in expected if c not in df]
    extra_cols = [c for c in df if c not in expected]
    checks.append({"check": "columns", "column": "*", "violations": len(missing_cols) + len(extra_cols),
                   "status": "PASS" if not missing_cols and not extra_cols else "FAIL",
                   "note": f"missing={missing_cols} extra={extra_cols}"})
    if missing_cols:
        return pd.DataFrame(checks), pd.DataFrame(violations)

    for col, kind in cfg["schema"].items():
        s = df[col]
        if kind == "int":
            as_num = pd.to_numeric(s, errors="coerce")
            not_int = as_num.isna() | (as_num != np.floor(as_num))
            add("type:int", col, not_int, "FAIL", f"dtype={s.dtype}")
        else:
            ok_type = s.map(lambda v: isinstance(v, str))
            add("type:category", col, ~ok_type, "FAIL", f"dtype={s.dtype}")
            allowed = set(cfg["categories"][col])
            add("allowed_values", col, ~s.isin(allowed), "WARN",
                f"{len(allowed)} allowed" + (f" incl. '{miss}'" if miss in allowed else ""))
            if col != cfg["target"]["name"]:
                n = int((s == miss).sum())
                checks.append({"check": "missing ('?')", "column": col, "violations": n,
                               "status": "INFO", "note": f"{n / len(df):.2%} of rows; allowed, imputed later"})

    for col, (lo, hi) in cfg["ranges"].items():
        v = pd.to_numeric(df[col], errors="coerce")
        add("range", col, (v < lo) | (v > hi), "WARN", f"[{lo}, {hi}]")

    edu_code = df["education"].map(cfg["education_num"])
    add("dictionary:education-num", "education-num",
        edu_code.notna() & (edu_code != df["education-num"]), "WARN", "must equal the code of education")

    target = cfg["target"]["name"]
    add("label_not_missing", target, df[target].isna() | (df[target] == miss), "FAIL")

    dup = df.duplicated(keep="first")
    checks.append({"check": "exact_duplicates", "column": "*", "violations": int(dup.sum()),
                   "status": "INFO", "note": "kept here; dropped from train only in clean.py"})
    for i in df.index[dup]:
        violations.append({"row": int(i), "check": "exact_duplicates", "column": "*", "value": None})

    return pd.DataFrame(checks), pd.DataFrame(violations, columns=["row", "check", "column", "value"])


def corrupt(df, cfg):
    """A copy with one known error per check, used to show the checks are not vacuous."""
    bad = df.head(200).copy()
    bad.loc[0, "age"] = 12                      # range
    bad.loc[1, "hours-per-week"] = 0            # range
    bad.loc[2, "capital-loss"] = 5000           # range
    bad.loc[3, "capital-gain"] = -1             # range
    bad["age"] = bad["age"].astype(object)
    bad.loc[4, "age"] = "forty"                 # type:int
    bad.loc[5, "workclass"] = "Self-employed"   # allowed_values
    bad.loc[6, "education-num"] = 3             # dictionary
    bad.loc[7, cfg["target"]["name"]] = cfg["missing_marker"]  # label
    return bad


def validate(raw, cfg):
    df = standardise(raw, cfg)
    checks, violations = run_checks(df, cfg)
    self_test, _ = run_checks(corrupt(df, cfg), cfg)

    RESULTS_DIR.mkdir(exist_ok=True)
    checks.to_csv(RESULTS_DIR / "validation_checks.csv", index=False)
    violations.to_csv(RESULTS_DIR / "validation_violations.csv", index=False)

    failed = checks[checks.status == "FAIL"]
    flagged_rows = violations[violations.check != "exact_duplicates"].row.nunique() if len(violations) else 0
    fired = self_test[self_test.status.isin(["FAIL", "WARN"])]
    n_missing = int(checks[checks.check.str.startswith("missing")].violations.sum())
    rows_missing = int((df == cfg["missing_marker"]).any(axis=1).sum())
    n_dup = int(checks[checks.check == "exact_duplicates"].violations.iloc[0])

    lines = [
        "# Validation report",
        "",
        f"Dataset: OpenML adult v{cfg['source']['version']}, {len(df)} rows x {df.shape[1]} columns "
        f"(label `{cfg['source']['target_source_name']}` renamed to `{cfg['target']['name']}`; "
        f"OpenML's NaN restored to the missing marker `'{cfg['missing_marker']}'`).",
        "",
        "Policy: nothing is dropped here. FAIL = structural error, the pipeline stops. "
        "WARN = value outside the schema, reported and kept. INFO = expected condition, handled by cleaning.",
        "",
        "## Checks",
        "",
        md_table(checks),
        "",
        "## Summary",
        "",
        f"- FAIL checks: {len(failed)}",
        f"- WARN checks: {int((checks.status == 'WARN').sum())}",
        f"- rows with a type, category, range, dictionary or label violation: {flagged_rows}",
        f"- missing ('?') cells: {n_missing} in {rows_missing} rows",
        f"- exact duplicate rows (beyond the first copy): {n_dup}",
        "- rows dropped by validation: 0",
        "",
        "Row-level violations: `validation_violations.csv` (row = position in the raw frame).",
        "",
        "## Self-test on a corrupted copy (first 200 rows, 8 injected errors)",
        "",
        md_table(fired[["check", "column", "violations", "status"]]),
        "",
    ]
    write_text("validation_report.md", "\n".join(lines))

    if len(failed):
        raise ValueError(f"validation failed:\n{failed.to_string(index=False)}")
    return df


if __name__ == "__main__":
    cfg = load_config()
    validate(fetch_raw(cfg), cfg)
