"""Check both models against the Cloud / Edge / Mobile / TinyML budgets from the lab manual.

Inputs (run measure.py and runtime_footprint.py first):
  results/measurements.csv, results/lean_serving_rss.csv
Outputs:
  results/deployment_fit.csv  -- one row per (model, scenario, platform) with every check
  results/deployment_fit.md   -- summary matrix for the report

Units: budgets are read as binary units (1 MB = 1 MiB, 1 KB = 1 KiB) to match the
measurements.

Memory budgets for Edge and Mobile are ranges of device RAM. A model "fits" a class
only if it runs on the smallest device of that class (peak <= lower bound); if it
needs more than the lower bound but no more than the upper bound it fits "partial"ly,
i.e. only on the larger devices of that class.

Memory scenarios:
  measured      peak RSS of the Python process used in this lab (includes torch)
  lean_python   peak RSS of a process with only numpy + scikit-learn + joblib
  native_est    ESTIMATE for a model ported out of Python (C / firmware): memory is
                taken as the serialized model size; latency is not re-measured
"""

import math

import pandas as pd

from common import RESULTS_DIR

KIB = 1 / 1024  # in MiB

# memory range in MiB (min, max), latency in ms, model size in MiB
BUDGETS = {
    "Cloud": {"mem": (1024, math.inf), "latency_ms": 100, "size_mib": 500},
    "Edge": {"mem": (256, 1024), "latency_ms": 50, "size_mib": 50},
    "Mobile": {"mem": (64, 256), "latency_ms": 20, "size_mib": 10},
    "TinyML": {"mem": (256 * KIB, 256 * KIB), "latency_ms": 10, "size_mib": 100 * KIB},
}


def memory_check(peak_mib, mem_range):
    lo, hi = mem_range
    if peak_mib <= lo:
        return "yes"
    if peak_mib <= hi:
        return "partial"
    return "no"


def verdict(checks):
    if "no" in checks:
        return "NO"
    if "partial" in checks:
        return "PARTIAL"
    return "FITS"


def to_markdown(df):
    """Minimal markdown table; avoids the optional `tabulate` dependency outside the pinned stack."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def main():
    meas = pd.read_csv(RESULTS_DIR / "measurements.csv").set_index("model")
    lean = pd.read_csv(RESULTS_DIR / "lean_serving_rss.csv").set_index("model")

    rows = []
    for model, m in meas.iterrows():
        size_mib = m["model_size_bytes"] / 2**20
        scenarios = {
            "measured": max(m["infer_peak_rss_mib"], m["train_peak_rss_mib"]),
            "lean_python": lean.loc[model, "serving_peak_rss_mib"],
            "native_est": size_mib,
        }
        for scenario, peak_mib in scenarios.items():
            for platform, b in BUDGETS.items():
                lat_ok = "yes" if m["infer_latency_median_ms"] <= b["latency_ms"] else "no"
                size_ok = "yes" if size_mib <= b["size_mib"] else "no"
                mem_ok = memory_check(peak_mib, b["mem"])
                rows.append({
                    "model": model,
                    "scenario": scenario,
                    "platform": platform,
                    "latency_ms": round(m["infer_latency_median_ms"], 4),
                    "latency_budget_ms": b["latency_ms"],
                    "latency_ok": lat_ok,
                    "size_kib": round(m["model_size_bytes"] / 1024, 2),
                    "size_budget_kib": round(b["size_mib"] * 1024, 2),
                    "size_ok": size_ok,
                    "peak_mem_mib": round(peak_mib, 3),
                    "mem_budget_mib": f"{b['mem'][0]:g}-{b['mem'][1]:g}",
                    "mem_ok": mem_ok,
                    "verdict": verdict([lat_ok, size_ok, mem_ok]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "deployment_fit.csv", index=False)

    matrix = df.pivot_table(index=["model", "scenario"], columns="platform", values="verdict",
                            aggfunc="first", sort=False)[list(BUDGETS)]
    failing = (
        df[df["verdict"] != "FITS"]
        .assign(reason=lambda d: d.apply(
            lambda r: ", ".join(k.split("_")[0] + (f" ({r[k]})" if r[k] == "partial" else "")
                                for k in ("latency_ok", "size_ok", "mem_ok") if r[k] != "yes"), axis=1))
        .loc[:, ["model", "scenario", "platform", "verdict", "reason"]]
    )
    md = ["# Deployment fit", "", "## Verdict matrix", "", to_markdown(matrix.reset_index()),
          "", "## Why not FITS", "", to_markdown(failing), ""]
    (RESULTS_DIR / "deployment_fit.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
