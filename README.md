# ML Systems Labs

Laboratory work for the course based on Vijay Janapa Reddi, *Introduction to Machine Learning Systems*.

## Setup

Python 3.11.8 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Known conflict in the pinned stack.** `mlflow==2.14.1` declares `pyarrow<16`, while the course requires
`pyarrow==16.1.0`, so `pip install -r requirements.txt` fails with `ResolutionImpossible`. To get the exact
required versions, install in two steps:

```bash
grep -v '^pyarrow' requirements.txt > req_no_pyarrow.txt   # PowerShell: Get-Content requirements.txt | ? { $_ -notmatch '^pyarrow' } > req_no_pyarrow.txt
pip install -r req_no_pyarrow.txt
pip install --no-deps pyarrow==16.1.0
```

`pip check` then reports only this mlflow/pyarrow mismatch; mlflow still imports and works.

## Layout

| Folder | Content |
|---|---|
| `lab01/` | Environment and first system measurements |
| `lab02/` | A neural network from scratch in NumPy: backpropagation, gradient check, training, timing |
| `data/` | Datasets, downloaded by scripts on first run (not tracked) |

Each lab folder contains `report.md`, `src/` (code) and `results/` (generated outputs).
