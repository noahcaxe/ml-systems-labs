"""Step 5: compare analytic gradients from backward() with central differences.

For 10 weights drawn at random from W1, W2, W3 (all entries equally likely):
  numerical = (L(w + h) - L(w - h)) / (2h),  h = 1e-5
  rel_error = |analytic - numerical| / max(|analytic|, |numerical|)
Pass criterion from the manual: rel_error < 1e-6 for at least 9 of 10 weights.
The loss is evaluated in float64 on one fixed mini-batch of 32 training samples.

Two weight initialisations are checked, the backward code is the same:
  N(0, 0.01)  the init required by the manual. Its gradients are ~1e-9..1e-5, so the
              float64 rounding error of the central difference, about
              eps * L / (2h) = 2.2e-16 * 2.3 / 2e-5 ~ 2.5e-11, alone gives
              rel_error > 1e-6 for most weights. Reported to show the limit.
  he          N(0, 2 / fan_in), same network. Gradients are large enough for the
              test to resolve 1e-6. This is the evidence that backward() is correct.

Extra columns: grad_zero marks weights whose analytic and numerical gradient are both
exactly 0 (dead ReLU path: trivially equal, not informative); kinks counts ReLU
pre-activations that change sign under the +-h nudge (finite difference invalid there).

Output: results/gradient_check.csv, results/gradient_check.txt
"""

import numpy as np
import pandas as pd

from common import RESULTS_DIR, SEED, load_split, set_seeds
from network import INIT_STD, backward, cross_entropy, forward, init_params

WIDTHS = (16, 64)
INITS = {"N(0, 0.01)": INIT_STD, "he": "he"}
BATCH = 32
N_WEIGHTS = 10
H = 1e-5
TOL = 1e-6
MIN_PASS = 9
WEIGHT_NAMES = ("W1", "W2", "W3")


def loss_and_masks(params, X, Y):
    P, cache = forward(params, X)
    masks = np.concatenate([(cache["Z1"] > 0).ravel(), (cache["Z2"] > 0).ravel()])
    return cross_entropy(P, Y), masks


def sample_weights(params, rng, n):
    """n distinct (name, index) pairs, uniform over all weight entries."""
    sizes = [params[name].size for name in WEIGHT_NAMES]
    flat = rng.choice(sum(sizes), size=n, replace=False)
    picks = []
    for k in flat:
        for name, size in zip(WEIGHT_NAMES, sizes):
            if k < size:
                picks.append((name, np.unravel_index(k, params[name].shape)))
                break
            k -= size
    return picks


def check(init_name, width, X, Y):
    params = init_params(width, np.random.default_rng(SEED), std=INITS[init_name])
    P, cache = forward(params, X)
    grads = backward(params, cache, P, Y)
    _, base_masks = loss_and_masks(params, X, Y)

    rows = []
    for name, idx in sample_weights(params, np.random.default_rng(SEED + width), N_WEIGHTS):
        w = params[name][idx]
        params[name][idx] = w + H
        loss_plus, masks_plus = loss_and_masks(params, X, Y)
        params[name][idx] = w - H
        loss_minus, masks_minus = loss_and_masks(params, X, Y)
        params[name][idx] = w

        numerical = (loss_plus - loss_minus) / (2 * H)
        analytic = grads[name][idx]
        denom = max(abs(analytic), abs(numerical))
        rel_error = abs(analytic - numerical) / denom if denom > 0 else 0.0
        rows.append({
            "init": init_name,
            "width": width,
            "param": name,
            "index": str(tuple(int(i) for i in idx)),
            "analytic": analytic,
            "numerical": numerical,
            "rel_error": rel_error,
            "pass": rel_error < TOL,
            "grad_zero": denom == 0,
            "kinks": int((masks_plus != base_masks).sum() + (masks_minus != base_masks).sum()),
        })
    return rows


def main():
    set_seeds()
    X_train, _, Y_train, _, _, _ = load_split()
    batch = np.random.default_rng(SEED).choice(X_train.shape[0], size=BATCH, replace=False)
    X, Y = X_train[batch], Y_train[batch]

    df = pd.DataFrame([r for init in INITS for w in WIDTHS for r in check(init, w, X, Y)])
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "gradient_check.csv", index=False)

    lines = [f"Gradient check: h = {H}, tolerance {TOL}, need >= {MIN_PASS}/{N_WEIGHTS}, "
             f"batch of {BATCH} training samples, float64", ""]
    for (init, width), group in df.groupby(["init", "width"], sort=False):
        n_pass = int(group["pass"].sum())
        lines.append(f"init {init}, W = {width}: {n_pass}/{N_WEIGHTS} pass "
                     f"({int(group['grad_zero'].sum())} with zero gradient) -> "
                     f"{'PASSED' if n_pass >= MIN_PASS else 'FAILED'}")
        lines.append(group.drop(columns=["init", "width"]).to_string(
            index=False, float_format=lambda v: f"{v:.6e}"))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "gradient_check.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
