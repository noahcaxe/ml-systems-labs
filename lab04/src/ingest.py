"""Step 2: ingestion. Read UCI Adult from OpenML, keep it unchanged, log its size and hash.

The frame is written as served (NaN = OpenML's decoded '?') to data/adult/raw.csv as an
immutable snapshot; the in-memory frame is checked against that file after writing.

Outputs: results/ingestion_log.txt, data/adult/raw.csv
"""

import platform
from datetime import datetime, timezone

import pandas as pd
import sklearn

from common import ADULT_DIR, fetch_raw, load_config, sha256_file, sha256_frame, write_text


def ingest(cfg):
    raw = fetch_raw(cfg)
    before = sha256_frame(raw)

    ADULT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = ADULT_DIR / "raw.csv"
    raw.to_csv(snapshot, index=False, lineterminator="\n")
    assert sha256_frame(raw) == before, "ingestion must not modify the frame"

    rows, cols = raw.shape
    expected = tuple(cfg["expected_shape"])
    assert (rows, cols) == expected, f"expected {expected}, got {(rows, cols)}"

    lines = [
        f"time (UTC)        : {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
        f"source            : fetch_openml(name='{cfg['source']['name']}', "
        f"version={cfg['source']['version']}, as_frame=True)",
        f"python / pandas / sklearn : {platform.python_version()} / {pd.__version__} / {sklearn.__version__}",
        f"rows              : {rows}",
        f"columns           : {cols}",
        f"column names      : {', '.join(raw.columns)}",
        f"dtypes            : {', '.join(f'{c}={t}' for c, t in raw.dtypes.items())}",
        f"NaN cells         : {int(raw.isna().sum().sum())} "
        f"({', '.join(f'{c}={n}' for c, n in raw.isna().sum().items() if n)})",
        f"sha256 (frame)    : {before}",
        f"sha256 (raw.csv)  : {sha256_file(snapshot)}",
        f"snapshot          : data/adult/raw.csv (written as served, not modified)",
    ]
    write_text("ingestion_log.txt", "\n".join(lines) + "\n")
    return raw


if __name__ == "__main__":
    ingest(load_config())
