"""Step 5: compare analytic gradients from backward() with central differences.

For 10 weights drawn at random from W1, W2, W3:
  numerical = (L(w + h) - L(w - h)) / (2h),  h = 1e-5
  rel_error = |analytic - numerical| / max(|analytic|, |numerical|)
Pass criterion from the manual: rel_error < 1e-6 for at least 9 of 10 weights.
The loss is evaluated in float64 on one fixed mini-batch of 32 training samples.

Weights are drawn only among entries with a non-zero analytic gradient. A zero gradient
(an input pixel that is 0 in the whole batch, or a ReLU unit that is off for every
sample) is matched exactly by the numerical slope and would pass without testing
anything.

The same backward() is checked in three network states:
  trained   init N(0, 0.01) as required, then 50 epochs of SGD with lr 0.1, batch 32
            (the train.py run). This is the evidence that backward() is correct.
  init      the untrained N(0, 0.01) network. Its gradients are ~1e-9..1e-5, so the
            float64 rounding error of the central difference, about
            eps * L / (2h) = 2.2e-16 * 2.3 / 2e-5 ~ 2.5e-11, alone gives
            rel_error > 1e-6 for many weights. Reported to show that limit.
  he        untrained, N(0, 2 / fan_in): the same check at a weight scale where it
            resolves 1e-6 without any training.

kinks counts ReLU pre-activations that change sign under the +-h nudge (the finite
difference is not valid across a kink).

Output: results/gradient_check.csv, results/gradient_check.txt
"""

import numpy as np
import pandas as pd

from common import RESULTS_DIR, SEED, load_split, set_seeds
from network import backward, cross_entropy, forward, init_params
from train import train

WIDTHS = (16, 64)
TRAIN_LR = 0.1
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


def sample_weights(grads, rng, n):
    """n distinct (name, index) pairs, uniform over weight entries with non-zero gradient."""
    candidates = [(name, idx) for name in WEIGHT_NAMES
                  for idx in zip(*np.nonzero(grads[name]))]
    picks = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[k] for k in picks], len(candidates)


def check(state, width, params, X, Y):
    P, cache = forward(params, X)
    grads = backward(params, cache, P, Y)
    _, base_masks = loss_and_masks(params, X, Y)
    picks, n_candidates = sample_weights(grads, np.random.default_rng(SEED + width), N_WEIGHTS)

    rows = []
    for name, idx in picks:
        w = params[name][idx]
        params[name][idx] = w + H
        loss_plus, masks_plus = loss_and_masks(params, X, Y)
        params[name][idx] = w - H
        loss_minus, masks_minus = loss_and_masks(params, X, Y)
        params[name][idx] = w

        numerical = (loss_plus - loss_minus) / (2 * H)
        analytic = grads[name][idx]
        rel_error = abs(analytic - numerical) / max(abs(analytic), abs(numerical))
        rows.append({
            "state": state,
            "width": width,
            "param": name,
            "index": str(tuple(int(i) for i in idx)),
            "analytic": analytic,
            "numerical": numerical,
            "rel_error": rel_error,
            "pass": rel_error < TOL,
            "kinks": int((masks_plus != base_masks).sum() + (masks_minus != base_masks).sum()),
        })
    n_weights = sum(params[name].size for name in WEIGHT_NAMES)
    return rows, f"{n_candidates}/{n_weights} weights have a non-zero gradient"


def main():
    set_seeds()
    data = load_split()
    X_train, Y_train = data[0], data[2]
    batch = np.random.default_rng(SEED).choice(X_train.shape[0], size=BATCH, replace=False)
    X, Y = X_train[batch], Y_train[batch]

    states = {}
    for width in WIDTHS:
        _, _, trained = train(TRAIN_LR, width, data, log=lambda msg: None)
        states[("trained", width)] = trained
    for width in WIDTHS:
        states[("init", width)] = init_params(width, np.random.default_rng(SEED))
    for width in WIDTHS:
        states[("he", width)] = init_params(width, np.random.default_rng(SEED), std="he")

    rows, notes = [], {}
    for (state, width), params in states.items():
        r, notes[(state, width)] = check(state, width, params, X, Y)
        rows += r
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "gradient_check.csv", index=False)

    lines = [f"Gradient check: h = {H}, tolerance {TOL}, need >= {MIN_PASS}/{N_WEIGHTS}, "
             f"batch of {BATCH} training samples, float64, weights with non-zero gradient",
             f"trained = init N(0, 0.01) + 50 epochs SGD lr {TRAIN_LR}; init = untrained "
             f"N(0, 0.01); he = untrained N(0, 2/fan_in)", ""]
    for (state, width), group in df.groupby(["state", "width"], sort=False):
        n_pass = int(group["pass"].sum())
        lines.append(f"{state}, W = {width}: {n_pass}/{N_WEIGHTS} pass -> "
                     f"{'PASSED' if n_pass >= MIN_PASS else 'FAILED'}  "
                     f"(max rel error {group['rel_error'].max():.2e}; "
                     f"{notes[(state, width)]})")
        lines.append(group.drop(columns=["state", "width"]).to_string(
            index=False, float_format=lambda v: f"{v:.6e}"))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "gradient_check.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
