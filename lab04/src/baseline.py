"""Step 5: LogisticRegression baseline with and without the derived features.

Model: [preprocessor (impute, clip, derived features) -> one-hot for categoricals +
StandardScaler for numerics -> LogisticRegression(max_iter=2000)], everything fitted on the
deduplicated train split only. Four variants differ only in which derived columns the model
sees. Test metrics are one number each, so 5-fold stratified CV on train (the whole pipeline is
refitted inside every fold) is added; all variants use the same folds, so the paired per-fold F1
difference against the variant without derived features shows whether a gain is consistent.

Output: results/baseline.csv, results/baseline.txt
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clean import split_and_dedupe
from common import RESULTS_DIR, columns, fetch_raw, load_config, md_table, standardise, write_text
from features import build_preprocessor

VARIANTS = {
    "without derived": [],
    "+ capital_net": ["capital_net"],
    "+ hours_per_age": ["hours_per_age"],
    "+ both": ["capital_net", "hours_per_age"],
}


def build_model(cfg, derived):
    numeric, categorical = columns(cfg)
    encode = ColumnTransformer([
        ("num", StandardScaler(), numeric + derived),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical, cfg, derived=True)),
        ("encode", encode),
        ("model", LogisticRegression(max_iter=2000, random_state=cfg["seed"])),
    ])


def baseline(df, cfg):
    target, pos = cfg["target"]["name"], cfg["target"]["positive"]
    train, test, _ = split_and_dedupe(df, cfg)
    X_train, y_train = train.drop(columns=target), (train[target] == pos).astype(int)
    X_test, y_test = test.drop(columns=target), (test[target] == pos).astype(int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])

    rows, fold_f1 = [], {}
    for name, derived in VARIANTS.items():
        model = build_model(cfg, derived)
        cv = cross_validate(model, X_train, y_train, cv=folds, scoring=["accuracy", "f1"])
        fold_f1[name] = cv["test_f1"]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "variant": name,
            "n_features": len(model.named_steps["encode"].get_feature_names_out()),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_f1": round(f1_score(y_test, pred), 4),
            "cv_accuracy": round(cv["test_accuracy"].mean(), 4),
            "cv_accuracy_std": round(cv["test_accuracy"].std(), 4),
            "cv_f1": round(cv["test_f1"].mean(), 4),
            "cv_f1_std": round(cv["test_f1"].std(), 4),
            "cv_paired_d_f1": round((cv["test_f1"] - fold_f1["without derived"]).mean(), 4),
            "folds_f1_better": int((cv["test_f1"] > fold_f1["without derived"]).sum()),
        })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "baseline.csv", index=False)
    base = table.iloc[0]
    delta = table.assign(d_test_f1=(table.test_f1 - base.test_f1).round(4),
                         d_test_accuracy=(table.test_accuracy - base.test_accuracy).round(4))
    write_text("baseline.txt", "\n".join([
        f"train {len(train)} rows (deduplicated), test {len(test)} rows; positive class '{pos}'",
        "",
        md_table(table),
        "",
        "Change against 'without derived' on test:",
        md_table(delta[["variant", "d_test_accuracy", "d_test_f1"]]),
    ]) + "\n")
    return table


if __name__ == "__main__":
    cfg = load_config()
    baseline(standardise(fetch_raw(cfg), cfg), cfg)
