"""Shared setup for Lab 1: seeds, data split, model definitions and paths."""

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.30

LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split():
    """Breast Cancer Wisconsin, 70/30 stratified split."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def make_models():
    """Fresh, unfitted baselines exactly as specified in the lab manual."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }
