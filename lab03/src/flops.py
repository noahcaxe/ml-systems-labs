"""Step 3a: parameters (torchinfo), hand-calculated FLOPs per layer, and profiler counts.

Convention: one MAC (multiply-accumulate, a += w * x) = 2 FLOPs. The Ch. 4 formulas count
FLOPs (2 x in x out for Linear); thop and torchinfo report MACs. Hand counts below include
only the multiply-accumulates of matrix products and convolutions; bias adds, ReLU, pooling,
softmax and LayerNorm are O(outputs) and listed separately as "other ops".

Profilers compared on one image (B = 1):
  thop default        : what thop.profile returns out of the box (eval + no_grad)
  thop, no fast path  : torch.backends.mha fast path off, so encoder layers run their submodules
  thop + MHA rule     : the same plus a custom thop rule for nn.MultiheadAttention
  torch FlopCounter   : torch.utils.flop_counter.FlopCounterMode (counts aten ops, in FLOPs)

Outputs: results/torchinfo_summary.txt, results/flops_layers.csv, results/flops_compare.csv,
results/flops.txt
"""

import io
import warnings
from contextlib import redirect_stdout

import pandas as pd
import thop
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torchinfo import summary

from common import RESULTS_DIR, set_seeds
from models import D_FF, D_MODEL, MODELS, N_HEADS, N_LAYERS, N_PATCHES, PATCH, build

warnings.filterwarnings("ignore")
BYTES = 4  # float32


def linear(name, n_in, n_out, rows=1):
    """rows x (n_in -> n_out): MACs = rows * in * out."""
    return dict(layer=name, kind="Linear", formula=f"{rows} x {n_in} x {n_out}" if rows > 1 else f"{n_in} x {n_out}",
                macs=rows * n_in * n_out, params=n_in * n_out + n_out, out_elems=rows * n_out)


def conv(name, k, c_in, c_out, h_out, w_out):
    """MACs = K * K * C_in * C_out * H_out * W_out."""
    return dict(layer=name, kind="Conv2d", formula=f"{k}x{k} x {c_in} x {c_out} x {h_out} x {w_out}",
                macs=k * k * c_in * c_out * h_out * w_out, params=k * k * c_in * c_out + c_out,
                out_elems=c_out * h_out * w_out)


def attention_layer(i, s=N_PATCHES, d=D_MODEL, h=N_HEADS, ff=D_FF):
    """One post-norm Transformer encoder layer on a sequence of s tokens of width d."""
    p = f"enc{i}."
    return [
        dict(layer=p + "qkv_proj", kind="Attention", formula=f"{s} x {d} x 3*{d}",
             macs=s * d * 3 * d, params=d * 3 * d + 3 * d, out_elems=s * 3 * d),
        dict(layer=p + "scores QK^T", kind="Attention", formula=f"{s}^2 x {d}  ({h} heads x {s}^2 x {d // h})",
             macs=s * s * d, params=0, out_elems=h * s * s),
        dict(layer=p + "weights x V", kind="Attention", formula=f"{s}^2 x {d}",
             macs=s * s * d, params=0, out_elems=s * d),
        dict(layer=p + "out_proj", kind="Attention", formula=f"{s} x {d} x {d}",
             macs=s * d * d, params=d * d + d, out_elems=s * d),
        linear(p + "ffn.linear1", d, ff, rows=s),
        linear(p + "ffn.linear2", ff, d, rows=s),
        dict(layer=p + "2 x LayerNorm", kind="Norm", formula="-", macs=0, params=2 * 2 * d,
             out_elems=2 * s * d),
    ]


def hand_layers():
    patch_dim = PATCH * PATCH
    layers = {
        "MLP": [linear("fc1", 784, 256), linear("fc2", 256, 128), linear("fc3", 128, 10)],
        "CNN": [conv("conv1", 3, 1, 16, 28, 28), conv("conv2", 3, 16, 32, 14, 14),
                linear("fc", 32 * 7 * 7, 10)],
        "Attention": [linear("patch_embed", patch_dim, D_MODEL, rows=N_PATCHES),
                      dict(layer="pos_embed", kind="Add", formula="-", macs=0,
                           params=N_PATCHES * D_MODEL, out_elems=N_PATCHES * D_MODEL)]
                     + [row for i in range(1, N_LAYERS + 1) for row in attention_layer(i)]
                     + [linear("head", D_MODEL, 10)],
    }
    rows = []
    for model, ls in layers.items():
        for l in ls:
            rows.append({"model": model, **l, "flops": 2 * l["macs"]})
    return pd.DataFrame(rows)


def count_mha(m: nn.MultiheadAttention, x, y):
    """Custom thop rule: QKV projection + scores + weighted sum of V + output projection (MACs)."""
    b, s, d = x[0].shape  # batch_first
    m.total_ops += torch.DoubleTensor([b * (s * d * 3 * d + 2 * s * s * d + s * d * d)])


def thop_macs(model, x, fastpath=True, custom_ops=None):
    torch.backends.mha.set_fastpath_enabled(fastpath)
    macs, params = thop.profile(model, (x,), custom_ops=custom_ops, verbose=False)
    torch.backends.mha.set_fastpath_enabled(True)
    return int(macs), int(params)


def torch_flops(model, x):
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    torch.backends.mha.set_fastpath_enabled(True)
    return int(fc.get_total_flops())


def main():
    set_seeds()
    RESULTS_DIR.mkdir(exist_ok=True)
    x = torch.randn(1, 1, 28, 28)
    layers = hand_layers()
    layers.to_csv(RESULTS_DIR / "flops_layers.csv", index=False)

    text, rows = io.StringIO(), []
    for name in MODELS:
        model = build(name).eval()
        info = summary(model, input_size=(1, 1, 28, 28), verbose=0, depth=4,
                       col_names=("output_size", "num_params"))
        text.write(f"{name}\n{info}\n\n")

        hand = layers[layers.model == name]
        assert hand.params.sum() == info.total_params, (name, hand.params.sum(), info.total_params)
        macs_default, params_thop = thop_macs(model, x)
        rows.append({
            "model": name,
            "params_torchinfo": info.total_params,
            "params_thop": params_thop,
            "hand_macs": int(hand.macs.sum()),
            "hand_flops": int(hand.flops.sum()),
            "thop_default": macs_default,
            "thop_no_fastpath": thop_macs(model, x, fastpath=False)[0],
            "thop_mha_rule": thop_macs(model, x, fastpath=False,
                                       custom_ops={nn.MultiheadAttention: count_mha})[0],
            "torch_flopcounter_flops": torch_flops(model, x),
            "weight_bytes": info.total_params * BYTES,
            "activation_bytes": int(hand.out_elems.sum()) * BYTES,
        })

    (RESULTS_DIR / "torchinfo_summary.txt").write_text(text.getvalue(), encoding="utf-8")
    compare = pd.DataFrame(rows)
    # Arithmetic intensity = FLOPs / bytes touched (weights read once per batch + activations per image).
    for b in (1, 128):
        compare[f"intensity_b{b}"] = (b * compare.hand_flops /
                                      (compare.weight_bytes + b * compare.activation_bytes)).round(2)
    compare.to_csv(RESULTS_DIR / "flops_compare.csv", index=False)

    with redirect_stdout(io.StringIO()) as out:
        pd.set_option("display.width", 200)
        print("Hand-calculated cost per layer, one image (MACs; FLOPs = 2 x MACs)\n")
        print(layers.drop(columns="out_elems").to_string(index=False))
        print("\nModel totals vs profilers (thop columns are MACs, FlopCounter is FLOPs)\n")
        print(compare.T.to_string(header=False))
    (RESULTS_DIR / "flops.txt").write_text(out.getvalue(), encoding="utf-8")
    print(out.getvalue())


if __name__ == "__main__":
    main()
