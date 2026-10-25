#!/usr/bin/env python3
"""
Bit-exact integer reference ("golden") model of the Binary-SNGP accelerator.

This file is the specification of the hardware. bsngp_core.vhd must reproduce
every integer below exactly; sim/compare_rtl.py checks that it does.

Datapath for one window x[t][i] (t = 0..63 oldest->newest, i = 0..5, int8 in [-127, 127]):

  DC  (if DC_REMOVE) x[t][i] -= floor(sum_t x[t][i] / 64)   per-window mean removal (gravity/offset
      invariance; makes the model independent of sensor mounting orientation's static part)
  L1  acc1[c][t] = sum_{k,i} (+/-) x[t+k-1][i]          binary weights, zero padding
      b1[c][t]   = acc1 >= TH1[c]  (DIR1=+1)  or  acc1 <= TH1[c]  (DIR1=-1)     (BN+sign folded)
  L2  acc2[c][t] = sum_{k,i} w2[c][i][k] * b1[i][2t+k-1]  (+/-1 x +/-1, zero padding)  -> b2
  L3  same with stride 2 on b2 -> b3[c][t], t = 0..15
  GAP h[c]       = sum_t b3[c][t]                         in [-16, 16]
  RFF s[j]       = sum_c WR[j][c] * h[c]                  WR in {+1,-1}
      idx[j]     = (s[j] * 2^KSH + BR[j]) mod 1024
      phi[j]     = COS_LUT[idx[j]]                        round(127*cos(2*pi*idx/1024))
  GP  L          = sum_j BETA[j] * phi[j] + BETA0          (logit, scale = LOGIT_SCALE)
      S          = sum_j phi[j]^2
      P[k]       = sum_j U[k][j] * phi[j]                 k = 0..R-1
      Q[k]       = P[k] >> QSH                            arithmetic shift (floor)
      V          = S - sum_k Q[k]^2                       ~ 127^2 * lambda * posterior variance
  out abnormal   = L > TL
      unknown    = V > TV
"""

import numpy as np

WIN = 64
STRIDE = 32
N_IN = 6
LUT_BITS = 10
LUT_SIZE = 1 << LUT_BITS
X_MAX = 127
INPUT_SCALE = 16.0  # x_int = clip(round(z * 16), -127, 127), z = z-score normalized sample


def quantize_input(z: np.ndarray) -> np.ndarray:
    """Quantize z-score normalized samples to the int8 code used on the wire and in hardware."""
    return np.clip(np.rint(z * INPUT_SCALE), -X_MAX, X_MAX).astype(np.int64)


def cos_lut() -> np.ndarray:
    n = np.arange(LUT_SIZE)
    return np.rint(127.0 * np.cos(2.0 * np.pi * n / LUT_SIZE)).astype(np.int64)


def remove_dc(x: np.ndarray) -> np.ndarray:
    """x: (N, 6, 64) int codes. Subtract floor(window mean) per channel (exact integer op)."""
    x = np.asarray(x, dtype=np.int64)
    return x - (x.sum(axis=-1, keepdims=True) >> 6)


def _conv_pm(x: np.ndarray, w_pm: np.ndarray, stride: int) -> np.ndarray:
    """
    x: (N, Cin, Lin) int, w_pm: (Cout, Cin, 3) in {+1,-1}; kernel 3, padding 1.
    out[n, c, t] = sum_{i,k} w[c,i,k] * x[n, i, stride*t + k - 1]
    """
    n, cin, lin = x.shape
    lout = (lin + 2 - 3) // stride + 1
    xp = np.zeros((n, cin, lin + 2), dtype=np.int64)
    xp[:, :, 1:lin + 1] = x
    out = np.zeros((n, w_pm.shape[0], lout), dtype=np.int64)
    for k in range(3):
        tap = xp[:, :, k:k + stride * (lout - 1) + 1:stride]  # (N, Cin, Lout)
        out += np.einsum("nil,ci->ncl", tap, w_pm[:, :, k])
    return out


def _threshold(acc: np.ndarray, th: np.ndarray, dirn: np.ndarray) -> np.ndarray:
    th = th[None, :, None]
    ge = dirn[None, :, None] > 0
    bit = np.where(ge, acc >= th, acc <= th)
    return np.where(bit, 1, -1).astype(np.int64)


def run(params: dict, x: np.ndarray, return_internals: bool = False) -> dict:
    """
    params: integer parameter dict (see bsngp_model.export_integer_params)
    x: (N, 64, 6) or (N, 6, 64) int window(s) in [-127, 127]; (64, 6) also accepted.
    """
    x = np.asarray(x, dtype=np.int64)
    if x.ndim == 2:
        x = x[None]
    if x.shape[1] == WIN and x.shape[2] == N_IN:
        x = x.transpose(0, 2, 1)
    assert x.shape[1:] == (N_IN, WIN), x.shape
    assert np.abs(x).max(initial=0) <= X_MAX

    pm = lambda b: np.where(np.asarray(b) > 0, 1, -1).astype(np.int64)

    if int(params.get("dc_remove", 0)):
        x = remove_dc(x)
    acc1 = _conv_pm(x, pm(params["w1"]), 1)
    b1 = _threshold(acc1, params["th1"], params["dir1"])
    acc2 = _conv_pm(b1, pm(params["w2"]), 2)
    b2 = _threshold(acc2, params["th2"], params["dir2"])
    acc3 = _conv_pm(b2, pm(params["w3"]), 2)
    b3 = _threshold(acc3, params["th3"], params["dir3"])
    h = b3.sum(axis=2)                                        # (N, C3)

    s = h @ pm(params["wr"]).T                                # (N, D)
    idx = (s * (1 << int(params["ksh"])) + params["br"][None, :]) & (LUT_SIZE - 1)
    phi = params["cos_lut"][idx]                              # (N, D) int

    L = phi @ params["beta"] + int(params["beta0"])
    S = (phi * phi).sum(axis=1)
    P = phi @ params["u"].T                                   # (N, R)
    Q = P >> int(params["qsh"])                               # numpy >> on int64 is floor
    V = S - (Q * Q).sum(axis=1)

    # Range checks: these widths are what the RTL implements.
    assert np.abs(acc1).max() < 2 ** 15
    assert np.abs(s).max() < 2 ** 10
    assert np.abs(L).max() < 2 ** 31
    assert np.abs(P).max() < 2 ** 23
    assert np.abs(V).max() < 2 ** 31

    out = {
        "logit": L,
        "var": V,
        "abnormal": (L > int(params["tl"])).astype(np.int64),
        "unknown": (V > int(params["tv"])).astype(np.int64),
    }
    if return_internals:
        out.update({"b1": b1, "b2": b2, "b3": b3, "h": h, "idx": idx, "phi": phi, "S": S, "P": P})
    return out


def stream_windows(samples: np.ndarray) -> np.ndarray:
    """
    Windows the accelerator evaluates on a continuous sample stream:
    one window after the 64th sample and then every 32 samples.
    samples: (T, 6) int. Returns (M, 64, 6).
    """
    T = len(samples)
    ends = list(range(WIN, T + 1, STRIDE))
    return np.stack([samples[e - WIN:e] for e in ends]) if ends else np.zeros((0, WIN, N_IN), np.int64)


def load_params(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}
