"""Measure system cost of both baselines: training time, single-sample latency,
model size and peak memory. Writes results/measurements.csv and results/timings_raw.csv.

Memory is measured in a fresh subprocess per (model, phase) so that one model's
allocations cannot inflate another's peak. Run without arguments.
"""

import json
import os
import statistics
import subprocess
import sys
import time

import joblib
import pandas as pd
from memory_profiler import memory_usage
from sklearn.base import clone

from common import RESULTS_DIR, load_split, make_models, set_seeds

TRAIN_RUNS = 5
INFER_RUNS = 100
MIB = 1024 * 1024


def time_training(model, X_train, y_train):
    clone(model).fit(X_train, y_train)  # warm-up
    times = []
    for _ in range(TRAIN_RUNS):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        times.append(time.perf_counter() - t0)
    return times


def time_inference(fitted, sample):
    fitted.predict(sample)  # warm-up
    times = []
    for _ in range(INFER_RUNS):
        t0 = time.perf_counter()
        fitted.predict(sample)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def model_path(name):
    return RESULTS_DIR / f"model_{name}.joblib"


def measure_memory(name, phase):
    """Runs inside a fresh subprocess. Prints JSON with RSS in MiB."""
    set_seeds()
    X_train, X_test, y_train, _ = load_split()
    if phase == "train":
        model = make_models()[name]
        target = (model.fit, (X_train, y_train))
    else:
        # Deployment view: load the saved model from disk, then serve single samples.
        model = joblib.load(model_path(name))
        sample = X_test[:1]

        def serve():
            for _ in range(INFER_RUNS):
                model.predict(sample)

        target = (serve, ())
    baseline = memory_usage(-1, interval=0.01, timeout=0.1, max_usage=True)
    peak = memory_usage(target, interval=0.001, max_usage=True, include_children=False)
    print(json.dumps({"baseline_mib": baseline, "peak_mib": peak}))


def run_memory_subprocess(name, phase):
    out = subprocess.run(
        [sys.executable, "-W", "ignore", __file__, "--memory", name, phase],
        capture_output=True, text=True, check=True, env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    set_seeds()
    X_train, X_test, y_train, _ = load_split()
    sample = X_test[:1]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows, raw = [], []
    for name, model in make_models().items():
        print(f"[{name}] timing training ({TRAIN_RUNS} runs + warm-up)...")
        train_times = time_training(model, X_train, y_train)

        fitted = clone(model).fit(X_train, y_train)
        print(f"[{name}] timing single-sample inference ({INFER_RUNS} runs + warm-up)...")
        infer_times = time_inference(fitted, sample)

        path = model_path(name)
        joblib.dump(fitted, path)
        size_bytes = path.stat().st_size

        print(f"[{name}] measuring peak memory in subprocesses...")
        mem_train = run_memory_subprocess(name, "train")
        mem_infer = run_memory_subprocess(name, "infer")

        rows.append({
            "model": name,
            "train_time_median_s": statistics.median(train_times),
            "infer_latency_median_ms": statistics.median(infer_times),
            "infer_latency_min_ms": min(infer_times),
            "infer_latency_max_ms": max(infer_times),
            "model_size_bytes": size_bytes,
            "model_size_kb": size_bytes / 1024,
            "process_baseline_rss_mib": mem_train["baseline_mib"],
            "train_peak_rss_mib": mem_train["peak_mib"],
            "train_peak_delta_mib": mem_train["peak_mib"] - mem_train["baseline_mib"],
            "infer_peak_rss_mib": mem_infer["peak_mib"],
            "infer_peak_delta_mib": mem_infer["peak_mib"] - mem_infer["baseline_mib"],
        })
        raw += [{"model": name, "kind": "train_s", "run": i, "value": t} for i, t in enumerate(train_times)]
        raw += [{"model": name, "kind": "infer_ms", "run": i, "value": t} for i, t in enumerate(infer_times)]

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "measurements.csv", index=False, float_format="%.4f")
    pd.DataFrame(raw).to_csv(RESULTS_DIR / "timings_raw.csv", index=False, float_format="%.6f")
    print()
    print(df.T.to_string(header=False))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--memory":
        measure_memory(sys.argv[2], sys.argv[3])
    else:
        main()
