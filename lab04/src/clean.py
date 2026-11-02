"""Step 4: split first, then clean with statistics learned from train only.

Order: stratified 70/30 split of the validated raw data -> drop exact duplicates from train
only -> fit the preprocessor (impute, clip) on train -> transform train and test with it.
The test set keeps every row it had after the split. Leakage guards are asserts below; the log
also shows the statistics a full-data fit would have learned, i.e. what leakage would change.

Outputs: results/preprocessor.joblib, results/cleaning_log.txt, results/fitted_stats.csv,
         data/adult/processed/{train,test}.csv
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import ADULT_DIR, RESULTS_DIR, canonical_csv, columns, fetch_raw, load_config, md_table, \
    standardise, write_text
from features import build_preprocessor, fitted_stats

PROCESSED_DIR = ADULT_DIR / "processed"


def split_and_dedupe(df, cfg):
    """(train, test, n_dropped). Rows keep their raw-frame index as a lineage id."""
    target = cfg["target"]["name"]
    train, test = train_test_split(df, test_size=cfg["split"]["test_size"],
                                   stratify=df[cfg["split"]["stratify"]], random_state=cfg["seed"])
    n_before = len(train)
    if cfg["cleaning"]["drop_train_duplicates"]:
        train = train[~train.duplicated(keep="first")]
    assert train.index.intersection(test.index).empty
    assert set(train[target]) == set(test[target])
    return train, test, n_before - len(train)


def processed_frame(pre, part, cfg):
    target = cfg["target"]["name"]
    X = pre.transform(part.drop(columns=target))
    return pd.concat([pd.Series(part.index, index=part.index, name="row_id"), X, part[target]], axis=1)


def clean(df, cfg):
    numeric, categorical = columns(cfg)
    target = cfg["target"]["name"]
    train, test, n_dropped = split_and_dedupe(df, cfg)
    n_test_split = len(test)

    pre = build_preprocessor(numeric, categorical, cfg).fit(train.drop(columns=target))
    stats = fitted_stats(pre)

    # Leakage guards: every learned statistic equals the one computed on train alone.
    miss = cfg["missing_marker"]
    lo_p, hi_p = cfg["cleaning"]["clip_percentiles"]
    for _, r in stats.iterrows():
        col = train[r.column]
        if r.kind == "numeric":
            assert r.impute == col.median()
            assert np.isclose(r.clip_low, np.percentile(col, lo_p))
            assert np.isclose(r.clip_high, np.percentile(col, hi_p))
        else:
            assert r.impute == col[col != miss].mode().sort_values().iloc[0]

    full = fitted_stats(build_preprocessor(numeric, categorical, cfg).fit(df.drop(columns=target)))
    stats["full_data_impute"] = full.impute
    stats["full_data_clip_low"] = full.clip_low
    stats["full_data_clip_high"] = full.clip_high

    train_p, test_p = processed_frame(pre, train, cfg), processed_frame(pre, test, cfg)
    assert len(test_p) == n_test_split and test_p.row_id.tolist() == test.index.tolist()
    assert not train_p.isna().any().any() and not test_p.isna().any().any()

    RESULTS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, RESULTS_DIR / "preprocessor.joblib")
    stats.to_csv(RESULTS_DIR / "fitted_stats.csv", index=False)
    (PROCESSED_DIR / "train.csv").write_bytes(canonical_csv(train_p))
    (PROCESSED_DIR / "test.csv").write_bytes(canonical_csv(test_p))

    def n_clipped(part):
        r = stats[stats.kind == "numeric"].set_index("column")
        v = part[r.index]
        return ((v < r.clip_low) | (v > r.clip_high)).sum()

    clipped = pd.DataFrame({"train": n_clipped(train), "test": n_clipped(test)}).rename_axis("column").reset_index()
    imputed = pd.DataFrame({"train": (train[categorical] == miss).sum(),
                            "test": (test[categorical] == miss).sum()}).rename_axis("column").reset_index()
    pos = cfg["target"]["positive"]
    show = stats.copy()
    for c in ["impute", "clip_low", "clip_high", "full_data_impute", "full_data_clip_low", "full_data_clip_high"]:
        show[c] = show[c].map(lambda v: v if isinstance(v, str) else ("" if pd.isna(v) else f"{v:g}"))

    lines = [
        f"split            : stratified on {target}, test_size={cfg['split']['test_size']}, "
        f"random_state={cfg['seed']}",
        f"train after split: {len(train) + n_dropped}  ({(train[target] == pos).mean():.4f} positive after dedupe)",
        f"test             : {len(test)}  ({(test[target] == pos).mean():.4f} positive), untouched",
        f"exact duplicates dropped from train: {n_dropped}  -> train {len(train)}",
        f"exact duplicates left in test      : {int(test.duplicated().sum())} (by design)",
        "",
        "Learned on train only (asserted); full_data_* = what a fit on all 48842 rows would learn:",
        md_table(show),
        "",
        "Values clipped to the train bounds:",
        md_table(clipped),
        "",
        "Missing ('?') categorical cells imputed with the train mode:",
        md_table(imputed),
        "",
        f"processed train  : {train_p.shape}, columns {', '.join(train_p.columns)}",
        f"processed test   : {test_p.shape}",
        "saved            : results/preprocessor.joblib, data/adult/processed/{train,test}.csv",
    ]
    write_text("cleaning_log.txt", "\n".join(lines) + "\n")
    return train_p, test_p, pre


if __name__ == "__main__":
    cfg = load_config()
    clean(standardise(fetch_raw(cfg), cfg), cfg)
