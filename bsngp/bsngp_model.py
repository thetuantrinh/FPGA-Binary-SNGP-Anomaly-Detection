#!/usr/bin/env python3
"""
Binary-SNGP (B-SNGP) training model and its export to the integer accelerator spec (golden.py).

Backbone: 3 binary 1D convolutions (1-bit weights, BN folded into per-channel integer
thresholds, sign activations) + global sum pooling.
Head: Gaussian-process output layer approximated with D random Fourier features
(SNGP, Liu et al. 2020), made hardware-friendly:
  * random projection weights are +/-1 and the phase is quantised to 1/1024 of a turn,
    so the projection is add/subtract only and cos() is a 1024-entry ROM;
  * the posterior covariance (lambda*I + Phi^T Phi)^-1 is replaced by a rank-R eigen
    form, which can only over-estimate the variance (conservative);
  * the output decision needs no sigmoid/sqrt/divide (L > TL, V > TV).

Note: classic SNGP also spectral-normalises the backbone. With sign activations the
backbone is scale-invariant, so spectral normalisation has no effect here; distance
awareness comes from the GP head only. This is a deliberate, documented difference.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anomaly_detection.bnn_models import binarize, BinaryConv1d

import golden

LUT_SIZE = golden.LUT_SIZE


class BinarySNGP(nn.Module):
    def __init__(self, c1=16, c2=32, c3=32, num_rff=128, ksh=2, seed=0, dc_remove=False):
        super().__init__()
        self.c1, self.c2, self.c3, self.num_rff, self.ksh = c1, c2, c3, num_rff, ksh
        self.dc_remove = dc_remove
        self.conv1 = BinaryConv1d(golden.N_IN, c1, 3, stride=1, padding=1, bias=False, binarize_input=False)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = BinaryConv1d(c1, c2, 3, stride=2, padding=1, bias=False, binarize_input=False)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = BinaryConv1d(c2, c3, 3, stride=2, padding=1, bias=False, binarize_input=False)
        self.bn3 = nn.BatchNorm1d(c3)

        # Fixed (untrained) random features: +/-1 projection, integer phase offsets.
        g = torch.Generator().manual_seed(10_000 + seed)
        wr = torch.randint(0, 2, (num_rff, c3), generator=g).float() * 2 - 1
        br = torch.randint(0, LUT_SIZE, (num_rff,), generator=g).float()
        self.register_buffer("wr", wr)
        self.register_buffer("br", br)
        self.out = nn.Linear(num_rff, 1)

    def features(self, x_int):
        """x_int: (N, 6, 64) float tensor holding integer input codes."""
        if self.dc_remove:  # same integer op as golden.remove_dc
            x_int = x_int - torch.div(x_int.sum(dim=-1, keepdim=True), 64, rounding_mode="floor")
        x = x_int / golden.INPUT_SCALE
        b1 = binarize(self.bn1(self.conv1(x)))
        b2 = binarize(self.bn2(self.conv2(b1)))
        b3 = binarize(self.bn3(self.conv3(b2)))
        h = b3.sum(dim=-1)                                     # (N, C3), integers in [-16, 16]
        return h, (b1, b2, b3)

    def rff(self, h):
        s = h @ self.wr.t()
        angle = (2.0 * math.pi / LUT_SIZE) * (s * (1 << self.ksh) + self.br)
        return torch.cos(angle)

    def forward(self, x_int):
        h, _ = self.features(x_int)
        return self.out(self.rff(h)).squeeze(-1)


def _fold_bn_threshold(bn: nn.BatchNorm1d, acc_scale: float):
    """
    bit = sign(gamma * (acc/acc_scale - mu) / sqrt(var + eps) + beta) >= 0, acc integer.
    Returns integer threshold and direction (+1: acc >= th, -1: acc <= th).
    """
    gamma = bn.weight.detach().cpu().double().numpy()
    beta = bn.bias.detach().cpu().double().numpy()
    mu = bn.running_mean.detach().cpu().double().numpy()
    sd = np.sqrt(bn.running_var.detach().cpu().double().numpy() + bn.eps)
    th = np.zeros(len(gamma), dtype=np.int64)
    dirn = np.ones(len(gamma), dtype=np.int64)
    big = 2 ** 14
    for c in range(len(gamma)):
        if gamma[c] == 0.0:
            th[c] = -big if beta[c] >= 0 else big
            dirn[c] = 1
            continue
        t = (mu[c] - beta[c] * sd[c] / gamma[c]) * acc_scale
        if gamma[c] > 0:
            th[c], dirn[c] = math.ceil(t), 1
        else:
            th[c], dirn[c] = math.floor(t), -1
        th[c] = int(np.clip(th[c], -big, big))
    return th, dirn


def _wbits(conv):
    return (conv.weight.detach().cpu().numpy() >= 0).astype(np.int64)  # 1 -> +1 (sign(0)=+1)


@torch.no_grad()
def export_integer_params(model: BinarySNGP) -> dict:
    """Backbone + head mean. Covariance (u) and thresholds (tl, tv) are added by fit_covariance()."""
    model.eval()
    th1, d1 = _fold_bn_threshold(model.bn1, golden.INPUT_SCALE)
    th2, d2 = _fold_bn_threshold(model.bn2, 1.0)
    th3, d3 = _fold_bn_threshold(model.bn3, 1.0)

    beta = model.out.weight.detach().cpu().double().numpy()[0]
    b0 = float(model.out.bias.detach().cpu().double().numpy()[0])
    s_beta = np.abs(beta).max() / 127.0
    beta_int = np.clip(np.rint(beta / s_beta), -127, 127).astype(np.int64)
    logit_scale = 127.0 / s_beta  # L_int ~= logit_float * logit_scale
    beta0_int = int(np.rint(b0 * logit_scale))

    return {
        "w1": _wbits(model.conv1), "th1": th1, "dir1": d1,
        "w2": _wbits(model.conv2), "th2": th2, "dir2": d2,
        "w3": _wbits(model.conv3), "th3": th3, "dir3": d3,
        "wr": (model.wr.cpu().numpy() > 0).astype(np.int64),
        "br": model.br.cpu().numpy().astype(np.int64),
        "ksh": np.int64(model.ksh),
        "cos_lut": golden.cos_lut(),
        "beta": beta_int, "beta0": np.int64(beta0_int),
        "logit_scale": np.float64(logit_scale),
        "qsh": np.int64(7),
        "dc_remove": np.int64(1 if model.dc_remove else 0),
    }


def fit_covariance(params: dict, phi_train_int: np.ndarray, rank: int, ridge: float = 1.0):
    """
    Laplace posterior of the RFF-GP output layer: Sigma = (ridge*I + Phi^T Phi)^-1, Phi = phi_int/127.
    ridge*var(phi) = ||phi||^2 - sum_k e_k/(ridge+e_k) (v_k^T phi)^2   (exact over all k).
    Keeping the top-`rank` eigen-directions drops only subtracted terms -> V over-estimates.
    U[k] = round(127 * sqrt(e_k/(ridge+e_k)) * v_k).
    """
    Phi = phi_train_int.astype(np.float64) / 127.0
    G = Phi.T @ Phi
    e, v = np.linalg.eigh(G)
    order = np.argsort(e)[::-1]
    e, v = e[order], v[:, order]
    w = e / (ridge + e)
    U = np.sqrt(np.clip(w[:rank], 0, None))[:, None] * v[:, :rank].T
    out = dict(params)
    out["u"] = np.clip(np.rint(127.0 * U), -127, 127).astype(np.int64)
    out["ridge"] = np.float64(ridge)
    out["eig"] = e
    return out


def full_variance(phi_train_int: np.ndarray, phi_int: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Reference: exact float posterior variance (full rank, no integer rounding)."""
    Phi = phi_train_int.astype(np.float64) / 127.0
    A = ridge * np.eye(Phi.shape[1]) + Phi.T @ Phi
    X = phi_int.astype(np.float64) / 127.0
    return np.einsum("nd,nd->n", X, np.linalg.solve(A, X.T).T)
