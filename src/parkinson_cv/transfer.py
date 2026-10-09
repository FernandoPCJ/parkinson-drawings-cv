"""Rung 4: transfer learning. A ResNet18 pre-trained on ImageNet, fine-tuned on the same ink maps.

Same protocol as `cnn.py` so the numbers are comparable: same 128x128 paper-relative ink map, same
light augmentation, class imbalance handled with `pos_weight`, a FIXED number of epochs (no looking at
test data to stop), and the same `fit_predict(train_df, test_df)` interface.

What changes:
  * the 1-channel ink map is repeated into 3 channels and normalised with the ImageNet mean / std,
    because the pre-trained filters expect that kind of input;
  * the last layer (1000 classes) is replaced by one logit;
  * the whole network is fine-tuned with a smaller learning rate (the pre-trained weights are a good
    starting point, not something to destroy in the first steps).

`pretrained=False` keeps the same architecture with random weights. It exists for tests (no download) and
as a control: if random-init ResNet18 does as well as the pre-trained one, the gain is not transfer.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import random_affine
from .cv import labels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """ResNet18 with a single-logit head. Pre-trained weights are downloaded on first use."""
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


class InkResNet(nn.Module):
    """Takes the (B,1,H,W) ink map in 0..1 like SmallCNN, returns one logit per image."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = build_resnet18(pretrained)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(-1, 3, -1, -1)
        return self.net((x - self.mean) / self.std).squeeze(1)


def train_resnet(X: np.ndarray, train_df, epochs: int = 8, batch: int = 64, lr: float = 3e-4,
                 seed: int = 0, augment: bool = True, pretrained: bool = True, device: str | None = None):
    """Fine-tune ONE model on the rows of `train_df` (column "idx" points into X). Returns (model, device)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr = torch.from_numpy(X[train_df["idx"].to_numpy()]).unsqueeze(1).to(dev)  # uint8
    ytr = torch.from_numpy(labels(train_df)).float().to(dev)
    n_pos = float(ytr.sum())
    pos_weight = torch.tensor((len(ytr) - n_pos) / max(n_pos, 1.0), device=dev)

    model = InkResNet(pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(ytr) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(ytr), device=dev)
        for s in range(0, len(perm), batch):
            ids = perm[s:s + batch]
            if len(ids) < 2:  # BatchNorm needs more than one sample
                continue
            xb = xtr[ids].float() / 255.0
            if augment:
                xb = random_affine(xb)
            loss = F.binary_cross_entropy_with_logits(model(xb), ytr[ids], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model, dev


def predict_proba(model: InkResNet, X: np.ndarray, rows, dev) -> np.ndarray:
    """Probability of parkinson for X[rows]."""
    xte = torch.from_numpy(X[np.asarray(rows)]).unsqueeze(1).to(dev)
    with torch.no_grad():
        return torch.cat([torch.sigmoid(model(xte[s:s + 256].float() / 255.0))
                          for s in range(0, len(xte), 256)]).cpu().numpy()


def resnet_fit_predict(X: np.ndarray, epochs: int = 8, batch: int = 64, lr: float = 3e-4, seed: int = 0,
                       augment: bool = True, pretrained: bool = True, device: str | None = None):
    """Return `fit_predict(train_df, test_df)`, same interface as `cnn.cnn_fit_predict`."""

    def fit_predict(train_df, test_df):
        model, dev = train_resnet(X, train_df, epochs, batch, lr, seed, augment, pretrained, device)
        proba = predict_proba(model, X, test_df["idx"].to_numpy(), dev)
        return (proba >= 0.5).astype(int), proba

    return fit_predict
