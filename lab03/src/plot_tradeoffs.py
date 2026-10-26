"""Step 5: test accuracy against parameter count and against inference latency.

Three panels share the accuracy axis: parameters, latency at batch 1, and latency per image
at batch 128 (median ms / 128). The x axes are logarithmic, since the models differ by up to
an order of magnitude. Every model keeps the same color in every panel and is labelled directly.

Inputs: results/test_accuracy.csv, results/flops_compare.csv, results/latency.csv
Output: results/tradeoffs.png
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from common import RESULTS_DIR

COLORS = {"MLP": "#2a78d6", "CNN": "#eb6834", "Attention": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    acc = pd.read_csv(RESULTS_DIR / "test_accuracy.csv").set_index("model")
    cost = pd.read_csv(RESULTS_DIR / "flops_compare.csv").set_index("model")
    lat = pd.read_csv(RESULTS_DIR / "latency.csv")
    b1 = lat[lat.batch == 1].set_index("model").median_ms
    b128 = lat[lat.batch == 128].set_index("model").ms_per_image

    panels = [
        (cost.params_torchinfo, "Parameters"),
        (b1, "Latency, batch 1 (ms)"),
        (b128, "Latency per image, batch 128 (ms)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (x, label) in zip(axes, panels):
        for model, color in COLORS.items():
            ax.scatter(x[model], acc.test_acc[model] * 100, s=70, color=color,
                       edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(model, (x[model], acc.test_acc[model] * 100), xytext=(8, 4),
                        textcoords="offset points", color=INK, fontsize=10)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(label, color=MUTED)
        ax.margins(x=0.25)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel("Test accuracy (%)", color=MUTED)
    axes[0].set_ylim(85, 90.5)
    fig.suptitle("Fashion-MNIST, 5 epochs: accuracy against size and latency (up and left is better)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tradeoffs.png", dpi=150)


if __name__ == "__main__":
    main()
