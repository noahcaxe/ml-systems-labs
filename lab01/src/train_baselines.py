"""Train the two baselines and record test accuracy to results/baseline_accuracy.csv."""

import pandas as pd
from sklearn.metrics import accuracy_score

from common import RESULTS_DIR, load_split, make_models, set_seeds


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    print(f"Train: {X_train.shape}, test: {X_test.shape}")

    rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": round(acc, 4)})
        print(f"{name}: {acc:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "baseline_accuracy.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
