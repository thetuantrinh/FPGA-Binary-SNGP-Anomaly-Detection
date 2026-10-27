#!/usr/bin/env python3
"""
Transfer the recorded dataset to a new sensor mounting (DE0-Nano field test, report_run.pdf §8):
gravity along +z instead of (-0.07, -0.46, -0.90) g, gyro x reading 0.

1. Reproduces the field failure in software: the deployed model (export/params.npz) on the
   recorded data rotated into the field mounting.
2. Trains an orientation-robust Binary-SNGP from the same recordings, without new labels:
     - isotropic input normalisation (one scale for accel axes, one for gyro axes);
     - per-window DC removal in the accelerator (gravity / offsets);
     - random 3-D rotation augmentation (accel + gyro rotated together) and gyro-axis dropout;
     - GP covariance and TV fitted on the augmented data (all orientations are "known").
3. Exports it to export_robust/ (params, co-sim vectors, demo ROM, isotropic norm stats).

Run from the repository root (CUDA GPU required):
  python bsngp/transfer_orientation.py
"""

import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np

from anomaly_detection.utils import init_hardware

import golden
import data
import orientation as ori
from train_eval import train_model, build_and_eval, summarize_seeds, ops_estimate, float_vs_golden

OUT = os.path.join(HERE, "export_robust")
DEPLOYED = os.path.join(HERE, "export", "params.npz")


def conditions(raw_normal, raw_abnormal, g_train, seed=123):
    """Evaluation conditions on held-out windows (raw physical units)."""
    rng = np.random.default_rng(seed)
    cond = {}
    for cls, raw in (("normal", raw_normal), ("abnormal", raw_abnormal)):
        cond[("original_mount", cls)] = raw
        cond[("field_mount_gx_dead", cls)] = ori.field_sim(raw, g_train, dead_gyro_axis=0)
        cond[("field_mount", cls)] = ori.field_sim(raw, g_train, dead_gyro_axis=None)
        cond[("random_orientation", cls)] = np.concatenate([ori.random_rotation(raw, rng) for _ in range(4)])
    return cond


def eval_conditions(params, cond):
    """Fraction of windows the integer model calls abnormal / unknown, per condition."""
    out = {}
    for (name, cls), raw in cond.items():
        g = golden.run(params, data.quantize_raw_windows(raw))
        out[f"{name}/{cls}"] = {"abnormal_rate": float(g["abnormal"].mean()),
                                "unknown_rate": float(g["unknown"].mean()),
                                "n": int(len(raw))}
    return out


def rec_stream(raw_rec, R=None, dead_gx=False):
    """Continuous (T, 6) recording -> int codes, optionally re-mounted."""
    r = raw_rec.T[None].astype(np.float64)
    if R is not None:
        r = ori.rotate(r, R)
    if dead_gx:
        r[:, 3] = 0.0
    return data.quantize_raw_windows(r)[0].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ksh", type=int, default=2)
    ap.add_argument("--copies", type=int, default=7, help="random orientations per training window")
    ap.add_argument("--gyro_drop", type=float, default=0.3)
    args = ap.parse_args()

    device = init_hardware(require_cuda=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    raw = data.id_splits_raw()
    (Rtr, ytr), (Rva, yva), (Rte, yte) = raw["train"], raw["val"], raw["test"]
    g_train = ori.gravity_direction(Rtr)
    print(f"training gravity direction: {np.round(g_train, 3)}")
    # Held-out evaluation windows. The test-normal file is the atypical 1 kHz still recording,
    # so normal = validation normal recording, abnormal = test abnormal recording.
    cond = conditions(Rva[yva == 0], Rte[yte == 1], g_train)

    # 1. Deployed model under the field mounting
    data.set_norm(*data.norm_stats())  # original per-axis stats
    deployed = eval_conditions(golden.load_params(DEPLOYED), cond)
    print("deployed model:")
    for k, v in deployed.items():
        print(f"  {k:36s} abnormal={v['abnormal_rate']*100:6.1f}%  unknown={v['unknown_rate']*100:6.1f}%")

    # 2. Orientation-robust model
    means, stds = ori.isotropic_stats(Rtr)
    data.set_norm(means, stds)
    print(f"isotropic normalisation: accel sd={stds[0]:.4f} g, gyro sd={stds[3]:.3f} deg/s")
    Atr, aytr = ori.augment(Rtr, ytr, args.copies, args.gyro_drop, seed=1)
    Ava, ayva = ori.augment(Rva, yva, 3, args.gyro_drop, seed=2)
    splits = {
        "train": (data.quantize_raw_windows(Atr), aytr),
        "val": (data.quantize_raw_windows(Ava), ayva),
        "test": (data.quantize_raw_windows(Rte), yte),
    }
    rn, ra = data.test_recordings()
    ood_sets = {k: v for k, v in data.online_ood().items() if not k.endswith("_raw_first")}
    ood_sets.update({"fault_" + k: v for k, v in data.sensor_faults([rn, ra]).items() if k != "axis_swap"})
    for k, (X, y) in splits.items():
        print(f"{k}: {X.shape} normal={int((y == 0).sum())} abnormal={int((y == 1).sum())}")

    seed_results, cond_results, main_model, main_params = [], [], None, None
    for s in range(args.seeds):
        model, vl = train_model(splits, device, args.ksh, s, args.epochs, dc_remove=True)
        p, res, _ = build_and_eval(model, splits, ood_sets, args.rank, device)
        res["seed"], res["val_loss"] = s, vl
        res["consistency_field"] = float_vs_golden(
            model, p, data.quantize_raw_windows(cond[("field_mount_gx_dead", "normal")]), device)
        seed_results.append(res)
        c = eval_conditions(p, cond)
        cond_results.append(c)
        print(f"[seed {s}] val_loss={vl:.4f} field normal: abnormal={c['field_mount_gx_dead/normal']['abnormal_rate']*100:.1f}% "
              f"unknown={c['field_mount_gx_dead/normal']['unknown_rate']*100:.1f}% | field abnormal: "
              f"abnormal={c['field_mount_gx_dead/abnormal']['abnormal_rate']*100:.1f}%")
        if s == 0:
            main_model, main_params = model, p
    summary = summarize_seeds(seed_results)
    cond_summary = {k: {m: (float(np.mean([c[k][m] for c in cond_results])),
                            float(np.std([c[k][m] for c in cond_results])))
                        for m in ("abnormal_rate", "unknown_rate")}
                    for k in cond_results[0]}

    # 3. Export (seed 0, fixed in advance)
    np.savez(os.path.join(OUT, "params.npz"), **main_params)
    json.dump({"channels": ["ax", "ay", "az", "gx", "gy", "gz"], "means": means.tolist(), "stds": stds.tolist(),
               "note": "Isotropic normalisation for the orientation-robust model (export_robust/params.npz). "
                       "Use instead of checkpoints/norm_stats.json when generating the IMU input constants."},
              open(os.path.join(OUT, "norm_stats_isotropic.json"), "w"), indent=2)
    prep = data._prep()
    from anomaly_detection.preprocess import load_single_csv
    rec_vn = load_single_csv(os.path.join(prep.data_dir, prep.split_cfg["val_normal"][0]))
    rec_ta = load_single_csv(os.path.join(prep.data_dir, prep.split_cfg["test_abnormal"][0]))
    R_field = ori.field_mount_rotations(g_train, n_yaw=8)[3]
    online = data.online_ood()
    parts = [("val_normal", rec_stream(rec_vn[:1024])),
             ("test_abnormal", rec_stream(rec_ta[:1024])),
             ("field_mount_normal_gx_dead", rec_stream(rec_vn[1024:2048], R_field, dead_gx=True)),
             ("field_mount_abnormal_gx_dead", rec_stream(rec_ta[1024:2048], R_field, dead_gx=True)),
             ("uav_6dof", online["uav_6dof_raw_first"][:1024]),
             ("zero_output", rec_stream(np.zeros((256, 6))))]
    stream = np.concatenate([p_[1] for p_ in parts]).astype(np.int64)
    exp = golden.run(main_params, golden.stream_windows(stream))
    np.save(os.path.join(OUT, "stream.npy"), stream)
    np.savez(os.path.join(OUT, "expected_stream.npz"), **exp)
    demo, demo_segs = data.demo_rom_stream()
    demo_exp = golden.run(main_params, golden.stream_windows(np.concatenate([demo, demo])))
    np.save(os.path.join(OUT, "demo_stream.npy"), demo)
    np.savez(os.path.join(OUT, "demo_expected.npz"), **demo_exp)

    out = {"config": vars(args), "g_train": g_train.tolist(), "deployed_model": deployed,
           "robust_conditions": cond_summary, "robust_summary": summary, "per_seed": seed_results,
           "stream_segments": [(n, len(x)) for n, x in parts], "demo_segments": demo_segs,
           "ops": ops_estimate(main_params),
           "hardware_model": {"tl": int(main_params["tl"]), "tv": int(main_params["tv"]),
                              "logit_scale": float(main_params["logit_scale"]),
                              "ridge": float(main_params["ridge"]), "dc_remove": 1},
           "runtime_s": time.time() - t0}
    json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)

    L = ["# Orientation transfer: deployed vs orientation-robust Binary-SNGP (integer models)", "",
         "Held-out windows: validation normal recording, test abnormal recording. Field mounting = training "
         "gravity rotated onto +z (DE0-Nano field test), 8 unknown yaw angles; 'gx_dead' also sets gyro x to 0.",
         f"Robust model: {args.seeds} seeds (mean ± std), {args.copies} random orientations per window, "
         f"gyro-axis dropout {args.gyro_drop}, DC removal in hardware, isotropic normalisation.", "",
         "| Condition | Class | Deployed: abnormal | Deployed: unknown | Robust: abnormal | Robust: unknown |",
         "|---|---|---|---|---|---|"]
    for k in deployed:
        name, cls = k.split("/")
        d, r = deployed[k], cond_summary[k]
        L.append(f"| {name} | {cls} | {d['abnormal_rate']*100:.1f}% | {d['unknown_rate']*100:.1f}% | "
                 f"{r['abnormal_rate'][0]*100:.1f} ± {r['abnormal_rate'][1]*100:.1f}% | "
                 f"{r['unknown_rate'][0]*100:.1f} ± {r['unknown_rate'][1]*100:.1f}% |")
    groups = sorted({k.split("/")[0] for k in summary if "/" in k})
    L += ["", "Robust model, OOD AUROC of V (ID = orientation-augmented validation windows):", "",
          "| OOD set | B-SNGP V | Mahalanobis on h | flagged unknown @TV |", "|---|---|---|---|"]
    f = lambda k: f"{summary[k]['mean']:.3f} ± {summary[k]['std']:.3f}"
    for g in groups:
        L.append(f"| {g} | {f(g+'/bsngp_var_vs_val/auroc')} | {f(g+'/baseline_mahalanobis_h_vs_val/auroc')} | "
                 f"{summary[g+'/flagged_unknown_rate']['mean']*100:.1f}% |")
    L += ["", f"Original-mount test accuracy (robust): {summary['id_accuracy']['mean']:.4f} ± {summary['id_accuracy']['std']:.4f}",
          f"Float vs integer mismatches on field-mount windows (seed 0): {seed_results[0]['consistency_field']}", ""]
    open(os.path.join(OUT, "results.md"), "w").write("\n".join(L))
    print("\n".join(L))
    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
