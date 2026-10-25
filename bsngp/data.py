#!/usr/bin/env python3
"""
Data for Binary-SNGP: in-distribution splits, OOD sets and the continuous co-simulation stream.
All outputs are integer input codes (golden.quantize_input), shape (N, 6, 64) for windows.
"""

import os
import numpy as np

from anomaly_detection.preprocess import DatasetPreprocessor, extract_windows, load_single_csv
from anomaly_detection.online_datasets_loader import (
    load_uav_imu_dataset, load_zenodo_electric_motor_dataset, load_adi_cbm_motor_dataset,
)

import golden

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "anomaly_detection", "config.yaml")


_NORM = None  # optional (means, stds) override, e.g. isotropic stats for the orientation-robust model


def set_norm(means, stds):
    global _NORM
    _NORM = (np.asarray(means, np.float64), np.asarray(stds, np.float64))


def norm_stats(prep=None):
    if _NORM is not None:
        return _NORM
    prep = prep or _prep()
    return np.asarray(prep.means, np.float64), np.asarray(prep.stds, np.float64)


def quantize_raw_windows(raw):
    """raw (N, 6, 64) physical units -> int codes with the active normalisation."""
    m, s = norm_stats()
    return golden.quantize_input((raw - m[None, :, None]) / s[None, :, None])


def _prep():
    os.chdir(ROOT)  # config paths are relative to the project root
    prep = DatasetPreprocessor(CONFIG)
    prep.load_statistics()
    return prep


def id_splits_raw():
    """Raw physical windows (N, 6, 64) and labels per split."""
    prep = _prep()
    return {s: prep.prepare_split(s, window_size=golden.WIN, stride=golden.STRIDE, normalize=False)
            for s in ("train", "val", "test")}


def id_splits():
    return {k: (quantize_raw_windows(X), y) for k, (X, y) in id_splits_raw().items()}


def _norm_rec(prep, raw):
    m, s = norm_stats(prep)
    return (raw - m) / s


def test_recordings():
    """Normalised continuous test recordings (normal, abnormal) as float (T, 6)."""
    prep = _prep()
    split = prep.split_cfg
    rn = [_norm_rec(prep, load_single_csv(os.path.join(prep.data_dir, p))) for p in split["test_normal"]]
    ra = [_norm_rec(prep, load_single_csv(os.path.join(prep.data_dir, p))) for p in split["test_abnormal"]]
    return rn[0], ra[0]


def val_normal_recording():
    """Typical 500 Hz normal recording (validation split) as float (T, 6)."""
    prep = _prep()
    return _norm_rec(prep, load_single_csv(os.path.join(prep.data_dir, prep.split_cfg["val_normal"][0])))


def _windows(rec, max_n=None):
    w, _ = extract_windows(rec.astype(np.float32), label=1, window_size=golden.WIN, stride=golden.STRIDE)
    if max_n is not None:
        w = w[:max_n]
    return golden.quantize_input(w)


def sensor_faults(rec_list, seed=0):
    """
    Synthetic sensor faults applied to the test recordings (normalised domain).
    These conditions never occur in training, so they are OOD for the classifier.
    """
    rng = np.random.default_rng(seed)
    faults = {}
    prep = _prep()
    zero_raw = _norm_rec(prep, np.zeros((1, 6), np.float32))  # sensor reads all-zero (detached / bus fault)
    for name in ("stuck", "zero_output", "noise_burst", "axis_swap", "spikes"):
        ws = []
        for rec in rec_list:
            r = rec.copy()
            if name == "stuck":
                # Output frozen at a single value, refreshed every 64 samples (last-good-value hold).
                for s in range(0, len(r), golden.WIN):
                    r[s:s + golden.WIN] = r[s]
            elif name == "zero_output":
                r[:] = zero_raw
            elif name == "noise_burst":
                r = r + rng.normal(0.0, 1.5, size=r.shape)
            elif name == "axis_swap":
                r = r[:, [2, 1, 0, 5, 4, 3]]  # sensor re-mounted rotated: x<->z on accel and gyro
            elif name == "spikes":
                m = rng.random(r.shape) < 0.05
                r = np.where(m, rng.choice([-6.0, 6.0], size=r.shape), r)
            ws.append(_windows(r, max_n=60))
        faults[name] = np.concatenate(ws)
    return faults


def online_ood(max_per_source=200, seed=42):
    """
    Other machines from online_datasets/. UAV has a real 6-DoF IMU. Electric-motor and ADI CbM
    have accelerometer only; their 'gyro' channels are synthesised (diff*100) by the loader, so
    they are reported separately and are an easy (artefact-driven) OOD case.
    """
    prep = _prep()
    online_dir = os.path.join(ROOT, "online_datasets")
    rng = np.random.default_rng(seed)
    out = {}
    for name, fn in (("uav_6dof", load_uav_imu_dataset),
                     ("motor_acc_only", load_zenodo_electric_motor_dataset),
                     ("adi_cbm_acc_only", load_adi_cbm_motor_dataset)):
        recs, _ = fn(online_dir)
        ws = []
        for r in recs[:8]:
            w = _windows(_norm_rec(prep, r), max_n=100)
            if len(w):
                ws.append(w)
        if not ws:
            continue
        X = np.concatenate(ws)
        idx = rng.choice(len(X), min(max_per_source, len(X)), replace=False)
        out[name] = X[np.sort(idx)]
        out[name + "_raw_first"] = golden.quantize_input(_norm_rec(prep, recs[0]))  # continuous, for the stream
    return out


def cosim_stream(online):
    """
    Continuous int8 sample stream (T, 6) for RTL co-simulation:
    val normal -> test normal (atypical) -> test abnormal -> stuck/zero sensor -> UAV (other machine) -> noise burst.
    """
    rn, ra = test_recordings()
    q = golden.quantize_input
    prep = _prep()
    zero = np.repeat(_norm_rec(prep, np.zeros((1, 6), np.float32)), 256, axis=0)
    stuck = np.repeat(ra[:1], 256, axis=0)
    rng = np.random.default_rng(1)
    noisy = rn[:512] + rng.normal(0.0, 1.5, size=rn[:512].shape)
    vn = val_normal_recording()
    parts = [q(vn[:1024]), q(rn), q(ra[:2000]), q(stuck), q(zero), online["uav_6dof_raw_first"][:1024], q(noisy)]
    names = ["val_normal", "test_normal_atypical", "test_abnormal", "stuck", "zero_output", "uav_6dof", "noise_burst"]
    seg = []
    for n, p in zip(names, parts):
        seg.append((n, len(p)))
    return np.concatenate(parts).astype(np.int64), seg


def demo_rom_stream():
    """Short stream stored on-chip for the stand-alone board demo (loops forever)."""
    _, ra = test_recordings()
    rn = val_normal_recording()
    q = golden.quantize_input
    prep = _prep()
    zero = np.repeat(_norm_rec(prep, np.zeros((1, 6), np.float32)), 128, axis=0)
    parts = [q(rn[:192]), q(ra[:192]), q(zero)]
    return np.concatenate(parts).astype(np.int64), [("normal", 192), ("abnormal", 192), ("zero_output", 128)]
