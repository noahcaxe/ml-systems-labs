"""Step 6: mini-batch SGD for every (learning rate, width) pair.

Protocol: init N(0, 0.01) (network.init_params), batch size 32, 50 epochs, the training
set is reshuffled every epoch. Each run uses the same seed for init and shuffling, so
runs differ only in learning rate and width. After every epoch the loss and accuracy
are evaluated on the full training and test sets and logged.

Outputs:
  results/training_log.txt     per-epoch log of every run
  results/training_history.csv one row per (run, epoch)
  results/final_accuracy.csv   final train/test loss and accuracy per run
  results/loss_curves.png      loss curves, one panel per width
"""

import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS_DIR, SEED, load_split, set_seeds  # noqa: E402
from network import accuracy, backward, cross_entropy, forward, init_params  # noqa: E402

LEARNING_RATES = (0.001, 0.01, 0.1)
WIDTHS = (16, 64)
BATCH_SIZE = 32
EPOCHS = 50

# categorical slots 1-3 of the dataviz reference palette, one per learning rate
LR_COLORS = {0.001: "#2a78d6", 0.01: "#eb6834", 0.1: "#1baf7a"}
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def evaluate(params, X, Y, y):
    P, _ = forward(params, X)
    return cross_entropy(P, Y), float((P.argmax(axis=1) == y).mean())


def train(lr, width, data, log):
    X_train, X_test, Y_train, Y_test, y_train, y_test = data
    params = init_params(width, np.random.default_rng(SEED))
    rng = np.random.default_rng(SEED)
    n = X_train.shape[0]

    history = []
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            batch = order[i:i + BATCH_SIZE]
            P, cache = forward(params, X_train[batch])
            grads = backward(params, cache, P, Y_train[batch])
            for name in params:
                params[name] -= lr * grads[name]

        train_loss, train_acc = evaluate(params, X_train, Y_train, y_train)
        test_loss, test_acc = evaluate(params, X_test, Y_test, y_test)
        history.append({
            "lr": lr, "width": width, "epoch": epoch,
            "train_loss": train_loss, "test_loss": test_loss,
            "train_acc": train_acc, "test_acc": test_acc,
        })
        log(f"  epoch {epoch:2d}  train loss {train_loss:.4f}  test loss {test_loss:.4f}  "
            f"train acc {train_acc:.4f}  test acc {test_acc:.4f}")
    seconds = time.perf_counter() - start
    assert accuracy(params, X_test, y_test) == test_acc
    return history, seconds


def direct_labels(ax, ends, min_gap):
    """Label line ends at the right edge; ends closer than min_gap share one label."""
    groups = []
    for lr, y in sorted(ends.items(), key=lambda item: item[1]):
        if groups and y - groups[-1][1] < min_gap:
            groups[-1][0].append(lr)
        else:
            groups.append(([lr], y))
    # push labels apart so neighbouring ones stay at least 11 pt from each other
    to_pt = 72 / ax.figure.dpi
    prev_pt = -np.inf
    for lrs, y in groups:
        y_pt = ax.transData.transform((EPOCHS, y))[1] * to_pt
        shift = max(0.0, prev_pt + 11 - y_pt)
        prev_pt = y_pt + shift
        ax.annotate("lr " + ", ".join(str(lr) for lr in sorted(lrs)), xy=(EPOCHS, y),
                    xytext=(6, shift), textcoords="offset points", va="center",
                    fontsize=9, color=TEXT)


def plot(df, path):
    """Top row: full loss range. Bottom row: zoom on the plateau around ln(10)."""
    fig, axes = plt.subplots(2, len(WIDTHS), figsize=(11, 7.4), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 2]})
    zoom = (2.3010, 2.3030)
    for col, width in enumerate(WIDTHS):
        for row, ax in enumerate(axes[:, col]):
            ax.set_facecolor(SURFACE)
            ends = {}
            for lr in LEARNING_RATES:
                run = df[(df.width == width) & (df.lr == lr)]
                color = LR_COLORS[lr]
                ax.plot(run.epoch, run.train_loss, color=color, linewidth=2,
                        label=f"lr {lr} train")
                ax.plot(run.epoch, run.test_loss, color=color, linewidth=1.5, linestyle="--",
                        label=f"lr {lr} test")
                y_end = run.train_loss.iloc[-1]
                if row == 0 or zoom[0] <= y_end <= zoom[1]:
                    ends[lr] = y_end
            if row == 0:
                ax.set_ylim(0, 2.45)
                ax.set_title(f"W = {width}", color=TEXT, fontsize=11, loc="left")
                direct_labels(ax, ends, min_gap=0.08)
            else:
                ax.axhline(np.log(10), color=TEXT_MUTED, linewidth=0.8, linestyle=":")
                ax.set_ylim(*zoom)
                ax.set_title(f"W = {width}, zoom on the plateau (dotted line = ln 10)",
                             color=TEXT_MUTED, fontsize=10, loc="left")
                ax.set_xlabel("epoch", color=TEXT_MUTED)
                direct_labels(ax, ends, min_gap=0.00004)
            ax.set_xlim(1, EPOCHS + 9)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.tick_params(colors=TEXT_MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
    axes[0, 0].set_ylabel("cross-entropy loss", color=TEXT_MUTED)
    axes[1, 0].set_ylabel("loss (zoom)", color=TEXT_MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Loss per epoch, solid = train, dashed = test (batch 32, init N(0, 0.01))",
                 color=TEXT, fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    set_seeds()
    data = load_split()
    RESULTS_DIR.mkdir(exist_ok=True)

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    rows, finals = [], []
    for width in WIDTHS:
        for lr in LEARNING_RATES:
            log(f"W = {width}, lr = {lr}")
            history, seconds = train(lr, width, data, log)
            rows += history
            last = history[-1]
            finals.append({**{k: last[k] for k in ("width", "lr", "train_loss", "test_loss",
                                                   "train_acc", "test_acc")},
                           "train_seconds": round(seconds, 2)})
            log(f"  final test accuracy {last['test_acc']:.4f}  ({seconds:.1f} s)")
            log("")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "training_history.csv", index=False)
    final = pd.DataFrame(finals)
    final.to_csv(RESULTS_DIR / "final_accuracy.csv", index=False)
    (RESULTS_DIR / "training_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(df, RESULTS_DIR / "loss_curves.png")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
