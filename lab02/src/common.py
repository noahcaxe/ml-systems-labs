"""Shared setup for Lab 2: seeds, digits data split and paths."""

import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.30
N_CLASSES = 10

LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def one_hot(y, n_classes=N_CLASSES):
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def load_split():
    """Digits 8x8, pixels scaled to [0, 1], one-hot labels, 70/30 stratified split.

    Returns X_train, X_test, Y_train, Y_test (one-hot) and y_train, y_test (int labels).
    """
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    return X_train, X_test, one_hot(y_train), one_hot(y_test), y_train, y_test
