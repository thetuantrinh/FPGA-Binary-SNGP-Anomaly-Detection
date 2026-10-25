#!/usr/bin/env python3
"""
Data for the input / feature-space / uncertainty figure of the manuscript.
Runs the integer reference model (golden.py) of the deployed orientation-robust Binary-SNGP
(export_robust/params.npz) on every data set and stores what the plots need.

Run from the repository root:
  python bsngp/viz_data.py
Output: bsngp/export_robust/viz_data.npz
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np

import golden
import data
import orientation as ori

OUT = os.path.join(HERE, "export_robust", "viz_data.npz")


def main():
    p = golden.load_params(os.path.join(HERE, "export_robust", "params.npz"))
    raw = data.id_splits_raw()
    (Rtr, ytr), (Rva, yva), (Rte, yte) = raw["train"], raw["val"], raw["test"]
    g_train = ori.gravity_direction(Rtr)
    data.set_norm(*ori.isotropic_stats(Rtr))
    q = data.quantize_raw_windows

    # Example raw windows (physical units) for panel (a)
    ex_normal = Rva[yva == 0][10]
    ex_abnormal = Rte[yte == 1][10]

    # Data sets
    rn, ra = data.test_recordings()
    sets = {
        "ID normal": q(Rva[yva == 0]),
        "ID abnormal": q(Rte[yte == 1]),
        "Field-mount normal": q(ori.field_sim(Rva[yva == 0], g_train, dead_gyro_axis=0, n_yaw=2)),
        "Field-mount abnormal": q(ori.field_sim(Rte[yte == 1][:60], g_train, dead_gyro_axis=0, n_yaw=2)),
    }
    online = data.online_ood()
    sets["UAV (other machine)"] = online["uav_6dof"]
    sets["Motor rig (acc. only)"] = online["motor_acc_only"]
    faults = data.sensor_faults([rn, ra])
    sets["Sensor stuck"] = faults["stuck"]
    sets["Sensor zero"] = faults["zero_output"]
    sets["Noise burst"] = faults["noise_burst"]
    sets["Spikes"] = faults["spikes"]

    out = {"ex_normal": ex_normal, "ex_abnormal": ex_abnormal, "tv": int(p["tv"]),
           "logit_scale": float(p["logit_scale"]), "names": np.array(list(sets))}
    # Training features (orientation-augmented training set, as used for the GP covariance)
    Atr, _ = ori.augment(Rtr, ytr, 7, 0.3, seed=1)
    htr = golden.run(p, q(Atr), return_internals=True)["h"].astype(np.float64)
    for k, (name, X) in enumerate(sets.items()):
        g = golden.run(p, X, return_internals=True)
        out[f"h_{k}"], out[f"V_{k}"], out[f"L_{k}"] = g["h"], g["var"], g["logit"]
        h = g["h"].astype(np.float64)
        d2 = (h ** 2).sum(1)[:, None] + (htr ** 2).sum(1)[None, :] - 2 * h @ htr.T
        out[f"dist_{k}"] = np.sqrt(np.maximum(d2.min(1), 0))  # distance to nearest training feature
    mu = htr.mean(0)
    _, S, Vt = np.linalg.svd(htr - mu, full_matrices=False)
    pcs = Vt[:2]
    out["pca_mu"], out["pca_pcs"], out["pca_var_ratio"] = mu, pcs, (S[:2] ** 2) / (S ** 2).sum()

    # GP-head variance on the PCA plane (h continuous; same phase/ROM/U arithmetic, float)
    z1 = np.linspace(-60, 60, 161)
    z2 = np.linspace(-60, 60, 161)
    Z1, Z2 = np.meshgrid(z1, z2)
    hg = mu + Z1.reshape(-1, 1) * pcs[0] + Z2.reshape(-1, 1) * pcs[1]
    wr = np.where(p["wr"] > 0, 1.0, -1.0)
    idx = (hg @ wr.T) * (1 << int(p["ksh"])) + p["br"][None, :]
    phi = 127.0 * np.cos(2 * np.pi * idx / 1024.0)
    Vg = (phi ** 2).sum(1) - (((phi @ p["u"].T) / 128.0) ** 2).sum(1)
    Lg = phi @ p["beta"] + int(p["beta0"])
    out["grid_z1"], out["grid_z2"] = z1, z2
    out["grid_V"], out["grid_L"] = Vg.reshape(Z1.shape), Lg.reshape(Z1.shape)
    # Dataset overview: per-recording vibration level over time, split, class, windows
    from anomaly_detection.preprocess import load_single_csv
    prep = data._prep()
    rec_rows = []
    for split_key, split, cls in (("train_normal", "train", 0), ("train_abnormal", "train", 1),
                                  ("val_normal", "val", 0), ("val_abnormal", "val", 1),
                                  ("test_normal", "test", 0), ("test_abnormal", "test", 1)):
        for rel in prep.split_cfg[split_key]:
            r = load_single_csv(os.path.join(prep.data_dir, rel))
            acc_dev = np.linalg.norm(r[:, 0:3] - r[:, 0:3].mean(0), axis=1)          # g, gravity removed
            n = len(r) // 250
            rms = np.sqrt((acc_dev[:n * 250].reshape(n, 250) ** 2).mean(1))          # 0.5 s blocks at 500 Hz
            rec_rows.append((os.path.basename(rel), split, cls, len(r), max(0, (len(r) - 64) // 32 + 1)))
            out[f"rec_rms_{len(rec_rows) - 1}"] = rms
    out["rec_name"] = np.array([r[0] for r in rec_rows])
    out["rec_split"] = np.array([r[1] for r in rec_rows])
    out["rec_class"] = np.array([r[2] for r in rec_rows])
    out["rec_len"] = np.array([r[3] for r in rec_rows])
    out["rec_windows"] = np.array([r[4] for r in rec_rows])
    # One example window (input codes) of every data set
    for k, (name, X) in enumerate(sets.items()):
        out[f"ex_{k}"] = X[len(X) // 2]
    np.savez(OUT, **out)
    print("wrote", OUT, {n: len(out[f'V_{k}']) for k, n in enumerate(sets)})


if __name__ == "__main__":
    main()
