"""Steps 1-4: data shapes, parameter shapes, forward-pass checks and initial loss.

Output: results/sanity_checks.txt
"""

import math

import numpy as np

from common import RESULTS_DIR, SEED, load_split, set_seeds
from network import PARAM_NAMES, count_params, cross_entropy, forward, init_params

WIDTHS = (16, 64)


def main():
    set_seeds()
    X_train, X_test, Y_train, Y_test, y_train, y_test = load_split()
    lines = [
        "Data",
        f"  X_train {X_train.shape}  X_test {X_test.shape}",
        f"  Y_train {Y_train.shape}  Y_test {Y_test.shape}",
        f"  X range [{X_train.min():.3f}, {X_train.max():.3f}]",
        f"  train class counts {np.bincount(y_train).tolist()}",
        f"  test  class counts {np.bincount(y_test).tolist()}",
        f"  one-hot rows sum to 1: {bool(np.all(Y_train.sum(axis=1) == 1))}",
        "",
    ]

    for width in WIDTHS:
        params = init_params(width, np.random.default_rng(SEED))
        lines.append(f"W = {width}  ({count_params(params)} parameters)")
        lines += [f"  {n:<3} {params[n].shape}" for n in PARAM_NAMES]

        # forward pass on batches of several sizes: 1 sample, a training batch, all data
        for X in (X_train[:1], X_train[:32], X_train, X_test):
            P, _ = forward(params, X)
            row_err = float(np.abs(P.sum(axis=1) - 1.0).max())
            assert P.shape == (X.shape[0], 10)
            assert row_err < 1e-6
            lines.append(f"  input {str(X.shape):<11} -> output {str(P.shape):<11} "
                         f"max |row sum - 1| = {row_err:.2e}  (< 1e-6: {row_err < 1e-6})")

        P, _ = forward(params, X_train)
        loss = cross_entropy(P, Y_train)
        lines += [
            f"  initial train loss = {loss:.6f}  (ln 10 = {math.log(10):.6f}, "
            f"diff {loss - math.log(10):+.2e})",
            "",
        ]

    text = "\n".join(lines)
    print(text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "sanity_checks.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
