"""Step 3b: inference latency for batch 1 (median of 100 runs) and batch 128 (median of 20 runs).

Protocol: model.eval() + torch.inference_mode() (the normal inference path, so the Transformer
fast path is on), real test images, 1 s busy loop to bring the CPU to its steady clock, 10
warm-up rounds, then timed rounds. Inside every round each model runs once in a fixed order,
so clock drift affects all models equally. Timer: time.perf_counter_ns. Median reported,
min and max kept. Weights do not change the amount of work, so untrained models are timed.

Outputs: results/latency.csv, results/latency_raw.csv
"""

import time

import numpy as np
import pandas as pd
import torch

from common import RESULTS_DIR, load_datasets, set_seeds
from models import MODELS, build

RUNS = {1: 100, 128: 20}
WARMUP = 10


def spin_up(seconds=1.0):
    A = torch.randn(256, 256)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        A @ A


def main():
    set_seeds()
    _, _, test = load_datasets()
    images = torch.stack([test[i][0] for i in range(max(RUNS))])
    models = {name: build(name).eval() for name in MODELS}
    flops = pd.read_csv(RESULTS_DIR / "flops_compare.csv").set_index("model").hand_flops

    spin_up()
    rows, raw = [], []
    with torch.inference_mode():
        for batch, runs in RUNS.items():
            x = images[:batch]
            for _ in range(WARMUP):
                for m in models.values():
                    m(x)
            samples = {name: [] for name in models}
            for r in range(runs):
                for name, m in models.items():
                    start = time.perf_counter_ns()
                    m(x)
                    samples[name].append(time.perf_counter_ns() - start)
            for name, s in samples.items():
                ms = np.array(s) / 1e6
                raw += [{"model": name, "batch": batch, "run": i, "ms": v} for i, v in enumerate(ms)]
                med = float(np.median(ms))
                rows.append({
                    "model": name, "batch": batch, "runs": runs,
                    "median_ms": round(med, 4), "min_ms": round(ms.min(), 4), "max_ms": round(ms.max(), 4),
                    "ms_per_image": round(med / batch, 5),
                    "images_per_s": round(batch / med * 1e3),
                    "gflop_per_s": round(batch * flops[name] / (med * 1e-3) / 1e9, 2),
                })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "latency.csv", index=False)
    pd.DataFrame(raw).to_csv(RESULTS_DIR / "latency_raw.csv", index=False)
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
