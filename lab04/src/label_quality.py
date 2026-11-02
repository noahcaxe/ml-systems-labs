"""Step 6: label-quality checks for Adult (an objective label, so no relabelling / kappa).

(a) Duplicate-conflict audit: rows with identical values in all 14 features but different
    income labels. On the full raw data and on the train split the model is fitted on.
(b) 50 random rows (seed from config) re-checked against the data dictionary (UCI adult.names):
    hard rules (a value the dictionary does not allow) and plausibility rules (allowed values that
    contradict each other). The rules are also counted on the full data for context.
    The 50 rows are written out with one column per rule for the manual read-through.

Outputs: results/label_conflicts.csv, results/label_audit_sample.csv, results/label_quality.txt
"""

import pandas as pd

from clean import split_and_dedupe
from common import RESULTS_DIR, fetch_raw, load_config, md_table, standardise, write_text


def conflicts(df, cfg):
    """Rows whose 14 features occur more than once with more than one income label."""
    target = cfg["target"]["name"]
    features = [c for c in df if c != target]
    n_labels = df.groupby(features, observed=True, dropna=False)[target].transform("nunique")
    return df[n_labels > 1].sort_values(features + [target])


def rules(df, cfg):
    """One boolean column per rule, True = violation."""
    r = pd.DataFrame(index=df.index)
    for col, (lo, hi) in cfg["ranges"].items():
        r[f"hard: {col} in [{lo}, {hi}]"] = ~df[col].between(lo, hi)
    for col, allowed in cfg["categories"].items():
        r[f"hard: {col} allowed"] = ~df[col].isin(allowed)
    r["hard: education-num = code(education)"] = df["education"].map(cfg["education_num"]) != df["education-num"]
    r["plaus: Husband is Male"] = (df.relationship == "Husband") & (df.sex != "Male")
    r["plaus: Wife is Female"] = (df.relationship == "Wife") & (df.sex != "Female")
    r["plaus: Husband/Wife is married"] = df.relationship.isin(["Husband", "Wife"]) & \
        ~df["marital-status"].str.startswith("Married")
    r["plaus: Never-worked has 0 hours"] = (df.workclass == "Never-worked") & (df["hours-per-week"] > 0)
    r["plaus: Bachelors+ needs age >= 21"] = (df["education-num"] >= 13) & (df.age < 21)
    r["plaus: not both capital gain and loss"] = (df["capital-gain"] > 0) & (df["capital-loss"] > 0)
    return r


def label_quality(df, cfg):
    target = cfg["target"]["name"]
    train, _, _ = split_and_dedupe(df, cfg)

    conf_full, conf_train = conflicts(df, cfg), conflicts(train, cfg)
    features = [c for c in df if c != target]
    conf_full.assign(group=conf_full.groupby(features, observed=True).ngroup()) \
        .rename_axis("row").reset_index().to_csv(RESULTS_DIR / "label_conflicts.csv", index=False)

    n = cfg["label_audit"]["sample_size"]
    sample = df.sample(n=n, random_state=cfg["seed"])
    r_sample, r_full = rules(sample, cfg), rules(df, cfg)
    flagged = r_sample.any(axis=1)
    out = sample.copy()
    out["violations"] = r_sample.apply(lambda row: "; ".join(row.index[row]), axis=1)
    out["in_conflict_group"] = sample.index.isin(conf_full.index)
    out.rename_axis("row").reset_index().to_csv(RESULTS_DIR / "label_audit_sample.csv", index=False)

    counts = pd.DataFrame({"rule": r_full.columns,
                           f"sample ({n})": r_sample.sum().to_numpy(),
                           f"full ({len(df)})": r_full.sum().to_numpy()})
    examples = df.loc[r_full.any(axis=1) & ~r_full.filter(like="hard").any(axis=1),
                      ["age", "workclass", "education", "marital-status", "occupation", "relationship",
                       "sex", "hours-per-week", target]]

    lines = [
        "(a) Duplicate-conflict audit (identical 14 features, different income)",
        f"  full data : {conf_full.groupby(features, observed=True).ngroups} groups, {len(conf_full)} rows "
        f"({len(conf_full) / len(df):.3%} of {len(df)})",
        f"  train     : {conf_train.groupby(features, observed=True).ngroups} groups, {len(conf_train)} rows "
        f"(after exact-duplicate removal; they stay in train)",
        f"  all groups: results/label_conflicts.csv",
        "",
        md_table(conf_full[["age", "workclass", "fnlwgt", "education", "occupation", "sex",
                            "hours-per-week", target]].rename_axis("row").reset_index()),
        "",
        f"(b) {n} random rows (random_state={cfg['seed']}) against the data dictionary",
        f"  rows with any violation : {int(flagged.sum())} / {n}",
        f"  rows in a conflict group: {int(out.in_conflict_group.sum())} / {n}",
        f"  rows: results/label_audit_sample.csv (columns 'violations', 'in_conflict_group')",
        "",
        md_table(counts),
        "",
        "Rows that break a plausibility rule in the full data:",
        md_table(examples.rename_axis("row").reset_index()),
    ]
    write_text("label_quality.txt", "\n".join(lines) + "\n")
    return conf_full, out


if __name__ == "__main__":
    cfg = load_config()
    label_quality(standardise(fetch_raw(cfg), cfg), cfg)
