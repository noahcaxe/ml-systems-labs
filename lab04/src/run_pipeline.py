"""Step 7: the whole Lab 4 pipeline in one command, plus the processed-data hash.

    python lab04/src/run_pipeline.py          # run every step, write results/run_config.yaml
    python lab04/src/run_pipeline.py --check  # run again and fail if any hash differs

Order: ingest -> validate -> clean (split, dedupe train, fit, transform) -> baseline ->
label_quality -> hashes. results/run_config.yaml = config.yaml as used + SHA256 of the raw frame
and of the processed train/test files + sizes, metrics and package versions.
"""

import argparse
import hashlib
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from baseline import baseline
from clean import PROCESSED_DIR, clean
from common import RESULTS_DIR, load_config, sha256_file, sha256_frame
from ingest import ingest
from label_quality import label_quality
from validate import validate

RUN_CONFIG = RESULTS_DIR / "run_config.yaml"


def dataset_hashes(raw):
    files = {name: sha256_file(PROCESSED_DIR / f"{name}.csv") for name in ("train", "test")}
    combined = hashlib.sha256()
    for name in ("train", "test"):
        combined.update((PROCESSED_DIR / f"{name}.csv").read_bytes())
    return {
        "raw_frame_sha256": sha256_frame(raw),
        "processed_train_sha256": files["train"],
        "processed_test_sha256": files["test"],
        "processed_dataset_sha256": combined.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare hashes with the stored run_config.yaml")
    args = parser.parse_args()
    previous = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8")) if RUN_CONFIG.exists() else None
    if args.check and previous is None:
        sys.exit(f"--check needs an earlier run: {RUN_CONFIG} not found")

    cfg = load_config()
    print("== 2. ingest"); raw = ingest(cfg)
    print("== 3. validate"); df = validate(raw, cfg)
    print("== 4. clean"); train, test, _ = clean(df, cfg)
    print("== 5. baseline"); scores = baseline(df, cfg)
    print("== 6. label quality"); conflicts, _ = label_quality(df, cfg)

    hashes = dataset_hashes(raw)
    run = {
        "hashes": hashes,
        "sizes": {"raw": list(raw.shape), "processed_train": list(train.shape),
                  "processed_test": list(test.shape), "label_conflict_rows": len(conflicts)},
        "baseline": scores.set_index("variant")[["test_accuracy", "test_f1"]].to_dict(orient="index"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                        "joblib": joblib.__version__, "pyyaml": yaml.__version__},
        "artifacts": ["results/preprocessor.joblib", "data/adult/processed/train.csv",
                      "data/adult/processed/test.csv"],
        "config": cfg,
    }
    RUN_CONFIG.write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print("== 7. hashes")
    for k, v in hashes.items():
        print(f"{k:26s} {v}")
    if previous is not None:
        diff = [k for k, v in hashes.items() if previous.get("hashes", {}).get(k) != v]
        print("same hashes as the previous run" if not diff else f"CHANGED since the previous run: {diff}")
        if args.check and diff:
            sys.exit(1)


if __name__ == "__main__":
    main()
