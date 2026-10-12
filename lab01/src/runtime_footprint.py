"""Measure how much process RSS the runtime itself costs, before any model exists.

Imports are added one layer at a time in a single fresh process and RSS is read
after each layer. Writes results/runtime_rss.csv.
"""

from pathlib import Path

import psutil

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def rss_mib(process):
    return process.memory_info().rss / 2**20


def main():
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

    lines = ["stage,rss_mib"] + [f"{stage},{value:.1f}" for stage, value in rows]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "runtime_rss.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for stage, value in rows:
        print(f"{stage:<24} {value:7.1f} MiB")


if __name__ == "__main__":
    main()
