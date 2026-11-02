"""Shared setup for Lab 4: paths, config, raw data loading and dataset hashing."""

import hashlib
from pathlib import Path

import yaml
from sklearn.datasets import fetch_openml

LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
DATA_DIR = LAB_DIR.parent / "data"
ADULT_DIR = DATA_DIR / "adult"
CONFIG_PATH = LAB_DIR / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_raw(cfg):
    """The raw frame exactly as OpenML serves it (cached in data/openml after the first run)."""
    src = cfg["source"]
    return fetch_openml(name=src["name"], version=src["version"], as_frame=True,
                        data_home=DATA_DIR, parser="auto").frame


def standardise(raw, cfg):
    """A copy with the label named as in the schema, categories as plain strings and the
    missing marker restored: OpenML decodes '?' to NaN, the schema keeps it as '?'."""
    df = raw.rename(columns={cfg["source"]["target_source_name"]: cfg["target"]["name"]})
    for col, kind in cfg["schema"].items():
        if kind == "category" and col in df:
            df[col] = df[col].astype(object).where(df[col].notna(), cfg["missing_marker"])
    return df


def columns(cfg):
    """(numeric, categorical) feature names in schema order, without the label."""
    target = cfg["target"]["name"]
    num = [c for c, k in cfg["schema"].items() if k == "int"]
    cat = [c for c, k in cfg["schema"].items() if k == "category" and c != target]
    return num, cat


def canonical_csv(df):
    """Deterministic text form of a frame: fixed column order, LF line ends, fixed float format."""
    return df.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode("utf-8")


def sha256_frame(df):
    return hashlib.sha256(canonical_csv(df)).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(name, text):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / name).write_text(text, encoding="utf-8")
    print(text)

