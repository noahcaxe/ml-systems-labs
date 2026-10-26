"""Shared setup for Lab 3: seeds, Fashion-MNIST loaders and paths."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 128
N_TRAIN, N_VAL = 55_000, 5_000

LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
DATA_DIR = LAB_DIR.parent / "data"

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_datasets():
    """Official 60k train split into 55k train / 5k validation (fixed seed); official 10k test."""
    full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=TRANSFORM)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=TRANSFORM)
    train, val = random_split(full_train, [N_TRAIN, N_VAL], generator=torch.Generator().manual_seed(SEED))
    return train, val, test


def make_loaders(batch_size=BATCH_SIZE):
    train, val, test = load_datasets()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(SEED)),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )
