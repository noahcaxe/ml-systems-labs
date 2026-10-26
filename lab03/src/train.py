"""Step 4: train every model for 5 epochs with Adam (lr 1e-3), batch 128.

Every model starts from the same seed and sees the training batches in the same order.
After every epoch: mean training loss, validation accuracy, wall time. After the last
epoch: test accuracy. The weights are saved so later steps can reuse them.

Outputs: results/training_history.csv, results/test_accuracy.csv, results/<model>.pt
"""

import time

import pandas as pd
import torch
from torch import nn

from common import RESULTS_DIR, make_loaders, set_seeds
from models import MODELS, build

EPOCHS = 5
LR = 1e-3


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = sum((model(x).argmax(1) == y).sum().item() for x, y in loader)
    return correct / len(loader.dataset)


def train_one(name):
    set_seeds()
    train_loader, val_loader, test_loader = make_loaders()
    model = build(name)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start, total = time.perf_counter(), 0.0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        seconds = time.perf_counter() - start
        val = accuracy(model, val_loader)
        history.append({"model": name, "epoch": epoch, "train_loss": total / len(train_loader.dataset),
                        "val_acc": val, "epoch_s": seconds})
        print(f"{name:9s} epoch {epoch}  loss {history[-1]['train_loss']:.4f}  val {val:.4f}  {seconds:.1f}s")
    torch.save(model.state_dict(), RESULTS_DIR / f"{name.lower()}.pt")
    return history, accuracy(model, test_loader)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    history, final = [], []
    for name in MODELS:
        h, test_acc = train_one(name)
        history += h
        final.append({"model": name, "val_acc": h[-1]["val_acc"], "test_acc": test_acc,
                      "train_s": sum(r["epoch_s"] for r in h)})
        print(f"{name}: test accuracy {test_acc:.4f}\n")
    pd.DataFrame(history).round(5).to_csv(RESULTS_DIR / "training_history.csv", index=False)
    pd.DataFrame(final).round(4).to_csv(RESULTS_DIR / "test_accuracy.csv", index=False)


if __name__ == "__main__":
    main()
