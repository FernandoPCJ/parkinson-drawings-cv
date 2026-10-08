"""Rung 3: a small CNN trained from scratch, plugged into the same CV loop as the baselines.

Input: 1-channel "ink map" (0 = paper, larger = ink) from scripts/build_cnn_cache.py.
Every fold trains a NEW model for a FIXED number of epochs. There is no early stopping on the
test fold: picking the best epoch by looking at test data would leak information and inflate
the result.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cv import labels


class SmallCNN(nn.Module):
    """4 blocks of Conv3x3 -> BatchNorm -> ReLU -> MaxPool, then global average pooling.

    Global average pooling means the head sees one number per channel ("how much of this
    pattern is present anywhere"), so the model has few parameters (~40k at width=16) and
    cannot just memorize pixel positions. Output: one logit per image (>0 means parkinson).
    """

    def __init__(self, width: int = 16, dropout: float = 0.3):
        super().__init__()
        channels = [width, 2 * width, 4 * width, 4 * width]
        layers, c_in = [], 1
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                       nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c_in, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


def random_affine(x: torch.Tensor, max_deg: float = 10.0, max_shift: float = 0.04,
                  scale: tuple[float, float] = (0.95, 1.05)) -> torch.Tensor:
    """Small random rotation / shift / zoom plus a mild brightness change, per image.

    Paper is 0, so the empty corners that the transform creates look like normal paper.
    No flips (a mirrored spiral turns the other way) and no blur (it would erase the fine
    stroke detail that may carry the signal).
    """
    b, dev = x.shape[0], x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * max_deg * math.pi / 180
    zoom = torch.empty(b, device=dev).uniform_(*scale)
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2  # grid spans [-1, 1]
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(ang) / zoom, torch.sin(ang) / zoom
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    gain = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    return (out * gain).clamp(0, 1)


def train_cnn(X: np.ndarray, train_df, epochs: int = 20, batch: int = 64, lr: float = 1e-3,
              width: int = 16, seed: int = 0, augment: bool = True, device: str | None = None):
    """Train ONE model on the rows of `train_df` (column "idx" points into X). Returns (model, device).

    Class imbalance is handled with `pos_weight` in the loss (like class_weight="balanced").
    The number of epochs is fixed: no looking at test data to decide when to stop.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr = torch.from_numpy(X[train_df["idx"].to_numpy()]).unsqueeze(1).to(dev)  # uint8
    ytr = torch.from_numpy(labels(train_df)).float().to(dev)
    n_pos = float(ytr.sum())
    pos_weight = torch.tensor((len(ytr) - n_pos) / max(n_pos, 1.0), device=dev)

    model = SmallCNN(width=width).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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


def predict_proba(model: SmallCNN, X: np.ndarray, rows, dev) -> np.ndarray:
    """Probability of parkinson for X[rows]."""
    xte = torch.from_numpy(X[np.asarray(rows)]).unsqueeze(1).to(dev)
    with torch.no_grad():
        return torch.cat([torch.sigmoid(model(xte[s:s + 256].float() / 255.0))
                          for s in range(0, len(xte), 256)]).cpu().numpy()


def cnn_fit_predict(X: np.ndarray, epochs: int = 20, batch: int = 64, lr: float = 1e-3,
                    width: int = 16, seed: int = 0, augment: bool = True, device: str | None = None):
    """Return `fit_predict(train_df, test_df)` for `cv.oof_predictions`.

    The DataFrames must have an integer column "idx" pointing to the row of X.
    """

    def fit_predict(train_df, test_df):
        model, dev = train_cnn(X, train_df, epochs, batch, lr, width, seed, augment, device)
        proba = predict_proba(model, X, test_df["idx"].to_numpy(), dev)
        return (proba >= 0.5).astype(int), proba

    return fit_predict


CAM_LAYER = 14  # model.features[14] = ReLU of the last conv block (8x8 map for a 128px input)


def grad_cam(model: SmallCNN, x: torch.Tensor):
    """Grad-CAM for one image x (1,1,H,W, float in 0..1).

    Returns (heat map HxW in 0..1, probability of parkinson). The map shows where the evidence
    FOR THE PREDICTED CLASS comes from: for a "parkinson" answer, regions that push the logit up;
    for a "healthy" answer, regions that push it down.
    """
    store = {}
    handle = model.features[CAM_LAYER].register_forward_hook(lambda m, i, o: store.update(a=o))
    logit = model(x)
    handle.remove()
    sign = 1.0 if float(logit.detach()) > 0 else -1.0
    act = store["a"]
    grad = torch.autograd.grad(sign * logit.sum(), act)[0]
    cam = F.relu((grad.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
    return cam, float(torch.sigmoid(logit.detach()))
