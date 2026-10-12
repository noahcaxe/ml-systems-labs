"""Measure how much process RSS the runtime itself costs, separately from the models.

1. Import layers: imports are added one layer at a time in a fresh process and RSS
   is read after each layer -> results/runtime_rss.csv.
2. Lean serving: for each saved model, a fresh process that imports only what
   serving needs (numpy, scikit-learn, joblib -- no torch, no pandas), loads the
   model and answers 100 single-sample requests; peak RSS is recorded
   -> results/lean_serving_rss.csv.

This module deliberately does not import common.py, because common.py imports torch.
Run measure.py first so the .joblib files exist.
"""

import json
import subprocess
import sys
from pathlib import Path

import psutil

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODELS = ["logistic_regression", "random_forest"]
SEED = 42
REQUESTS = 100


def rss_mib(process):
    return process.memory_info().rss / 2**20


def import_layers():
    p = psutil.Process()
    rows = [("python interpreter", rss_mib(p))]

    import numpy  # noqa: F401
    import sklearn.datasets  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    rows.append(("+ numpy, scikit-learn", rss_mib(p)))

    import joblib  # noqa: F401
    import pandas  # noqa: F401
    rows.append(("+ pandas, joblib", rss_mib(p)))

    import torch  # noqa: F401  (only needed for the mandatory torch seeds)
    rows.append(("+ torch", rss_mib(p)))
    return rows


def lean_serving(name):
    """Runs inside a fresh subprocess. Prints JSON with RSS in MiB."""
    import joblib
    from memory_profiler import memory_usage
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True)
    _, X_test, _, _ = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    sample = X_test[:1]
    p = psutil.Process()
    before_load = rss_mib(p)
    model = joblib.load(RESULTS_DIR / f"model_{name}.joblib")

    def serve():
        for _ in range(REQUESTS):
            model.predict(sample)

    peak = memory_usage((serve, ()), interval=0.001, max_usage=True)
    print(json.dumps({"before_load_mib": before_load, "peak_mib": peak}))


def main():
    rows = import_layers()
    lines = ["stage,rss_mib"] + [f"{stage},{value:.1f}" for stage, value in rows]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "runtime_rss.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Import layers:")
    for stage, value in rows:
        print(f"  {stage:<24} {value:7.1f} MiB")

    lines = ["model,runtime_before_load_mib,serving_peak_rss_mib"]
    print("Lean serving (numpy + scikit-learn + joblib only):")
    for name in MODELS:
        out = subprocess.run(
            [sys.executable, "-W", "ignore", __file__, "--lean", name],
            capture_output=True, text=True, check=True,
        )
        res = json.loads(out.stdout.strip().splitlines()[-1])
        lines.append(f"{name},{res['before_load_mib']:.1f},{res['peak_mib']:.1f}")
        print(f"  {name:<24} peak {res['peak_mib']:7.1f} MiB")
    (RESULTS_DIR / "lean_serving_rss.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--lean":
        lean_serving(sys.argv[2])
    else:
        main()
