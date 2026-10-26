"""The three architectures compared in Lab 3. Input: (B, 1, 28, 28), output: (B, 10) logits.

MLP:       Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10
CNN:       Conv(1,16,3,p=1) -> ReLU -> MaxPool2 -> Conv(16,32,3,p=1) -> ReLU -> MaxPool2
           -> Flatten -> Linear(32*7*7, 10)
Attention: 16 patches of 7x7 -> Linear 49->64 (+ learned position embedding)
           -> 2 x TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128)
           -> mean over patches -> Linear 64->10
"""

import torch
from torch import nn

PATCH = 7
N_PATCHES = (28 // PATCH) ** 2  # 16
D_MODEL, N_HEADS, D_FF, N_LAYERS = 64, 4, 128, 2


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 x 14 x 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 x 7 x 7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


def to_patches(x):
    """(B, 1, 28, 28) -> (B, 16, 49): row-major 4x4 grid of 7x7 patches, each flattened."""
    B = x.shape[0]
    x = x.reshape(B, 28 // PATCH, PATCH, 28 // PATCH, PATCH)  # (B, row, i, col, j)
    return x.permute(0, 1, 3, 2, 4).reshape(B, N_PATCHES, PATCH * PATCH)


class PatchTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(PATCH * PATCH, D_MODEL)
        # Without a position signal, self-attention + mean pooling cannot tell where a
        # patch came from (the model would be invariant to shuffling the patches).
        self.pos = nn.Parameter(torch.zeros(1, N_PATCHES, D_MODEL))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
        self.head = nn.Linear(D_MODEL, 10)

    def forward(self, x):
        h = self.embed(to_patches(x)) + self.pos  # (B, 16, 64)
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": PatchTransformer}


def build(name):
    return MODELS[name]()
