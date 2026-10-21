"""Step 7: time the forward and the backward pass separately.

For every width and batch size: 20 warm-up rounds, then 100 timed rounds; in each round
every measured function is called once, in a fixed order, so CPU clock drift affects
all of them equally (a 1 s busy loop before the first configuration spins the clock up).
Timer: time.perf_counter_ns. The median is reported, min and max are kept. The backward
timing excludes the forward pass that produces its cache.

To link the timing to matrix multiplication, the same matmuls are also timed alone:
  forward  3 matmuls: X @ W1, A1 @ W2, A2 @ W3
  backward 5 matmuls: A2.T @ dZ3, dZ3 @ W3.T, A1.T @ dZ2, dZ2 @ W2.T, X.T @ dZ1
(dX = dZ1 @ W1.T is not needed, the input has no parameters). Multiply-accumulate counts:
  forward  B * (64W + W^2 + 10W)
  backward B * (64W + W^2 + 10W) + B * (10W + W^2)

Batch 32 is the training batch; 1257 is the whole training set.
Outputs: results/timing.csv, results/timing_raw.csv, results/timing.txt
"""

import os
import platform
import time

import numpy as np
import pandas as pd

from common import RESULTS_DIR, SEED, load_split, set_seeds
from network import N_INPUT, N_OUTPUT, backward, forward, init_params

WIDTHS = (16, 64)
BATCHES = (32, 1257)
WARMUP = 20
REPEATS = 100


def time_interleaved(fns):
    """Time every function REPEATS times, alternating them inside each round so that
    CPU frequency drift hits all of them equally. Returns microseconds per function."""
    for _ in range(WARMUP):
        for fn in fns.values():
            fn()
    samples = {name: [] for name in fns}
    for _ in range(REPEATS):
        for name, fn in fns.items():
            start = time.perf_counter_ns()
            fn()
            samples[name].append(time.perf_counter_ns() - start)
    return {name: np.array(s) / 1e3 for name, s in samples.items()}


def spin_up(seconds=1.0):
    """Busy loop so the CPU reaches its steady clock before the first measurement."""
    A = np.random.default_rng(0).normal(size=(256, 256))
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        A @ A


def macs(batch, width):
    fwd = batch * (N_INPUT * width + width * width + width * N_OUTPUT)
    return fwd, fwd + batch * (width * N_OUTPUT + width * width)


def main():
    set_seeds()
    X_train, _, Y_train, _, _, _ = load_split()

    spin_up()
    rows, raw = [], []
    for width in WIDTHS:
        params = init_params(width, np.random.default_rng(SEED))
        for batch in BATCHES:
            X, Y = X_train[:batch], Y_train[:batch]
            P, cache = forward(params, X)
            dZ3 = (P - Y) / batch
            dZ2, dZ1 = cache["Z2"], cache["Z1"]  # right shapes; values do not affect speed
            A1, A2, W1, W2, W3 = cache["A1"], cache["A2"], params["W1"], params["W2"], params["W3"]

            def fwd_matmuls():
                X @ W1
                A1 @ W2
                A2 @ W3

            def bwd_matmuls():
                A2.T @ dZ3
                dZ3 @ W3.T
                A1.T @ dZ2
                dZ2 @ W2.T
                X.T @ dZ1

            passes = {
                "forward": lambda: forward(params, X),
                "backward": lambda: backward(params, cache, P, Y),
                "forward_matmuls": fwd_matmuls,
                "backward_matmuls": bwd_matmuls,
            }
            medians = {}
            for name, us in time_interleaved(passes).items():
                medians[name] = float(np.median(us))
                raw += [{"width": width, "batch": batch, "pass": name, "run": i, "us": v}
                        for i, v in enumerate(us)]
                rows.append({"width": width, "batch": batch, "pass": name,
                             "median_us": medians[name], "min_us": float(us.min()),
                             "max_us": float(us.max())})
            fwd_macs, bwd_macs = macs(batch, width)
            rows.append({"width": width, "batch": batch, "pass": "_summary",
                         "fwd_macs": fwd_macs, "bwd_macs": bwd_macs,
                         "macs_ratio": bwd_macs / fwd_macs,
                         "time_ratio": medians["backward"] / medians["forward"],
                         "fwd_matmul_share": medians["forward_matmuls"] / medians["forward"],
                         "bwd_matmul_share": medians["backward_matmuls"] / medians["backward"]})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(raw).to_csv(RESULTS_DIR / "timing_raw.csv", index=False)
    df.to_csv(RESULTS_DIR / "timing.csv", index=False)

    lines = [f"Median of {REPEATS} runs after {WARMUP} warm-up, microseconds. "
             f"{platform.processor()}, {os.cpu_count()} logical CPUs, numpy {np.__version__}", ""]
    header = (f"{'W':>3} {'B':>5} | {'forward':>9} {'backward':>9} {'bwd/fwd':>7} | "
              f"{'fwd mm':>8} {'bwd mm':>8} {'mm share f/b':>13} | {'MACs fwd':>9} "
              f"{'MACs bwd':>9} {'ratio':>5}")
    lines += [header, "-" * len(header)]
    for (width, batch), g in df.groupby(["width", "batch"]):
        t = g.set_index("pass")["median_us"]
        s = g[g["pass"] == "_summary"].iloc[0]
        lines.append(
            f"{width:>3} {batch:>5} | {t['forward']:>9.1f} {t['backward']:>9.1f} "
            f"{s.time_ratio:>7.2f} | {t['forward_matmuls']:>8.1f} {t['backward_matmuls']:>8.1f} "
            f"{s.fwd_matmul_share:>6.0%} /{s.bwd_matmul_share:>5.0%} | "
            f"{int(s.fwd_macs):>9} {int(s.bwd_macs):>9} {s.macs_ratio:>5.2f}")
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "timing.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
