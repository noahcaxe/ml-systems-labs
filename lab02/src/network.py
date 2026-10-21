"""Two-hidden-layer MLP in plain NumPy: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax.

Shapes (B = batch size):
  X  (B, 64)
  W1 (64, W)  b1 (W,)    Z1 = X  @ W1 + b1   A1 = relu(Z1)   (B, W)
  W2 (W, W)   b2 (W,)    Z2 = A1 @ W2 + b2   A2 = relu(Z2)   (B, W)
  W3 (W, 10)  b3 (10,)   Z3 = A2 @ W3 + b3   P  = softmax(Z3) (B, 10)
"""

import numpy as np

N_INPUT = 64
N_OUTPUT = 10
INIT_STD = 0.01
PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")


def init_params(width, rng, std=INIT_STD):
    """Weights from N(0, std^2), biases zero.

    std="he" uses sqrt(2 / fan_in) per layer instead of one fixed value.
    """
    shapes = {"W1": (N_INPUT, width), "W2": (width, width), "W3": (width, N_OUTPUT)}
    params = {}
    for i, (name, shape) in enumerate(shapes.items(), start=1):
        s = np.sqrt(2.0 / shape[0]) if std == "he" else std
        params[name] = rng.normal(0.0, s, size=shape)
        params[f"b{i}"] = np.zeros(shape[1])
    return params


def count_params(params):
    return sum(p.size for p in params.values())


def relu(Z):
    return np.maximum(Z, 0.0)


def softmax(Z):
    # subtract row max for numerical stability; result is unchanged
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def forward(params, X):
    """Return class probabilities P (B, 10) and the cache needed by backward."""
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)
    Z3 = A2 @ params["W3"] + params["b3"]
    P = softmax(Z3)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return P, cache


def cross_entropy(P, Y, eps=1e-12):
    """Mean over the batch of -sum_k Y_k log P_k."""
    return float(-np.sum(Y * np.log(P + eps)) / Y.shape[0])


def backward(params, cache, P, Y):
    """Gradients of the mean cross-entropy w.r.t. every parameter (chain rule).

    Softmax + cross-entropy together give dL/dZ3 = (P - Y) / B. Each linear layer
    Z = A @ W + b passes back dW = A.T @ dZ, db = sum_rows(dZ), dA = dZ @ W.T, and
    ReLU passes the gradient only where its input was positive.
    """
    B = Y.shape[0]
    dZ3 = (P - Y) / B
    dW3 = cache["A2"].T @ dZ3
    db3 = dZ3.sum(axis=0)

    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (cache["Z2"] > 0)
    dW2 = cache["A1"].T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (cache["Z1"] > 0)
    dW1 = cache["X"].T @ dZ1
    db1 = dZ1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def predict(params, X):
    P, _ = forward(params, X)
    return P.argmax(axis=1)


def accuracy(params, X, y):
    return float((predict(params, X) == y).mean())
