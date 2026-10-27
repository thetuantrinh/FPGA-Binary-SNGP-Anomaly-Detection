#!/usr/bin/env python3
"""
Train Binary-SNGP, export it to the integer accelerator spec, and evaluate the *integer*
model (the exact function the FPGA computes).

Run from the repository root (CUDA GPU required):
  python bsngp/train_eval.py

Outputs (bsngp/export/):
  params.npz               integer parameters of the hardware model (seed 0, main config)
  stream.npy, expected_stream.npz      co-simulation stimulus + golden results
  demo_stream.npy, demo_expected.npz   on-chip demo ROM contents + golden results
  results.json, results.md             metrics
"""

import os
import sys
import json
import time
import copy
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from anomaly_detection.utils import init_hardware, set_seed

import golden
import data
from bsngp_model import BinarySNGP, export_integer_params, fit_covariance, full_variance

EXPORT = os.path.join(HERE, "export")


def train_model(splits, device, ksh, seed, epochs, lr=1e-3, batch=128, dc_remove=False):
    set_seed(seed)
    model = BinarySNGP(ksh=ksh, seed=seed, dc_remove=dc_remove).to(device)
    for n, p in model.named_parameters():
        if p.device.type != "cuda":
            raise RuntimeError(f"parameter {n} not on CUDA")
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(device)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = F.binary_cross_entropy_with_logits(model(Xtr_t[b]), ytr_t[b], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(model(Xva_t), yva_t, pos_weight=pos_weight).item()
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def fpr_at_95tpr(score_id, score_ood):
    thr = np.percentile(score_ood, 5)  # 95% of OOD flagged
    return float(np.mean(score_id >= thr))


def ood_metrics(score_id, score_ood):
    y = np.r_[np.zeros(len(score_id)), np.ones(len(score_ood))]
    s = np.r_[score_id, score_ood].astype(np.float64)
    return {"auroc": float(roc_auc_score(y, s)), "fpr95": fpr_at_95tpr(score_id, score_ood)}


def mahalanobis_scores(h_train, y_train, h_eval, ridge=1e-3):
    """DDU-style baseline on the same binary features: min class-conditional Mahalanobis distance."""
    h_train = h_train.astype(np.float64)
    mus = [h_train[y_train == c].mean(0) for c in (0, 1)]
    cen = np.concatenate([h_train[y_train == c] - mus[c] for c in (0, 1)])
    cov = cen.T @ cen / len(cen) + ridge * np.eye(h_train.shape[1])
    P = np.linalg.inv(cov)
    d = [np.einsum("nd,dk,nk->n", h_eval - m, P, h_eval - m) for m in mus]
    return np.minimum(d[0], d[1])


@torch.no_grad()
def float_vs_golden(model, params, X, device):
    """Fraction of windows where the float training model and the integer spec disagree."""
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    h, (b1, b2, b3) = model.features(xt)
    logit = model(xt).cpu().numpy()
    g = golden.run(params, X, return_internals=True)
    return {
        "b3_bit_mismatch_frac": float(np.mean(b3.cpu().numpy().astype(np.int64) != g["b3"])),
        "decision_mismatch_frac": float(np.mean((logit > 0) != (g["logit"] > int(params["tl"])))),
    }


def build_and_eval(model, splits, ood_sets, rank, device, tv_pct=99.0, cov_X=None, tv_X=None):
    """cov_X / tv_X: optional (e.g. orientation-augmented) windows for the covariance fit and for TV."""
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]
    p = export_integer_params(model)
    p["tl"] = np.int64(0)
    p["tv"] = np.int64(0)
    p["u"] = np.zeros((rank, model.num_rff), np.int64)
    gtr = golden.run(p, Xtr, return_internals=True)
    phi_cov = gtr["phi"] if cov_X is None else golden.run(p, cov_X, return_internals=True)["phi"]
    p = fit_covariance(p, phi_cov, rank=rank)
    gva = golden.run(p, Xva, return_internals=True)
    v_tv = gva["var"] if tv_X is None else golden.run(p, tv_X)["var"]
    p["tv"] = np.int64(int(np.ceil(np.percentile(v_tv, tv_pct))))

    gte = golden.run(p, Xte, return_internals=True)
    res = {
        "id_test": {
            "n": int(len(yte)),
            "accuracy": float(accuracy_score(yte, gte["abnormal"])),
            "f1": float(f1_score(yte, gte["abnormal"])),
            "auroc_logit": float(roc_auc_score(yte, gte["logit"])),
            "unknown_rate": float(gte["unknown"].mean()),
        },
        "ood": {},
        "consistency_float_vs_integer": float_vs_golden(model, p, Xte, device),
    }
    gva["unknown"] = (gva["var"] > int(p["tv"])).astype(np.int64)  # gva was computed before TV was set
    res["id_val"] = {"n": int(len(yva)), "unknown_rate": float(gva["unknown"].mean())}
    # The test-normal recording (valid/normal/2026-09-17_10-57-42.csv) is itself atypical
    # (1 kHz logger, sensor nearly still), so OOD metrics are reported against two ID references:
    # 'test' (as configured) and 'val' (clean 500 Hz normal + abnormal recordings).
    h_tr = gtr["h"]
    refs = {"test": gte, "val": gva}
    for name, Xo in ood_sets.items():
        go = golden.run(p, Xo, return_internals=True)
        m = {"n": int(len(Xo)), "flagged_unknown_rate": float(go["unknown"].mean())}
        for rn, gi in refs.items():
            sfx = "" if rn == "test" else "_vs_val"
            m["bsngp_var" + sfx] = ood_metrics(gi["var"], go["var"])
            m["baseline_neg_abs_logit" + sfx] = ood_metrics(-np.abs(gi["logit"]), -np.abs(go["logit"]))
            m["baseline_mahalanobis_h" + sfx] = ood_metrics(
                mahalanobis_scores(h_tr, ytr, gi["h"].astype(np.float64)),
                mahalanobis_scores(h_tr, ytr, go["h"].astype(np.float64)))
            m["reference_full_rank_float_var" + sfx] = ood_metrics(
                full_variance(phi_cov, gi["phi"]), full_variance(phi_cov, go["phi"]))
        res["ood"][name] = m
    return p, res, gtr


def rank_ablation(model, splits, ood_sets, device, ranks=(2, 4, 8, 16, 32, 64, 128)):
    out = {}
    for r in ranks:
        _, res, _ = build_and_eval(model, splits, ood_sets, r, device)
        out[r] = {k: v["bsngp_var_vs_val"]["auroc"] for k, v in res["ood"].items()}
    return out


def ops_estimate(p):
    c1, c2, c3 = len(p["th1"]), len(p["th2"]), len(p["th3"])
    d, r = len(p["beta"]), len(p["u"])
    return {
        "binary_weight_bits": int(p["w1"].size + p["w2"].size + p["w3"].size),
        "rff_weight_bits": int(p["wr"].size),
        "int8_head_values": int(d + r * d),
        "layer1_addsub": int(c1 * 6 * 3 * 64),
        "layer2_xnor_bits": int(c2 * c1 * 3 * 32),
        "layer3_xnor_bits": int(c3 * c2 * 3 * 16),
        "head_mults": int(d * (2 + r) + r),
        "rtl_cycles_estimate": int((64 + 32 + 16) * 10 + 5 * d + r + 4),
    }


def summarize_seeds(results):
    keys = {}
    for res in results:
        keys.setdefault("id_accuracy", []).append(res["id_test"]["accuracy"])
        keys.setdefault("id_unknown_rate", []).append(res["id_test"]["unknown_rate"])
        keys.setdefault("id_val_unknown_rate", []).append(res["id_val"]["unknown_rate"])
        for name, m in res["ood"].items():
            for method in ("bsngp_var", "baseline_neg_abs_logit", "baseline_mahalanobis_h",
                           "reference_full_rank_float_var"):
                for sfx in ("", "_vs_val"):
                    keys.setdefault(f"{name}/{method}{sfx}/auroc", []).append(m[method + sfx]["auroc"])
            for sfx in ("", "_vs_val"):
                keys.setdefault(f"{name}/bsngp_var{sfx}/fpr95", []).append(m["bsngp_var" + sfx]["fpr95"])
            keys.setdefault(f"{name}/flagged_unknown_rate", []).append(m["flagged_unknown_rate"])
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in keys.items()}


def write_markdown(summary, ksweep, rank_abl, ops, n_seeds, path):
    groups = sorted({k.split("/")[0] for k in summary if "/" in k})
    L = ["# Binary-SNGP results (integer model = what the FPGA computes)", "",
         f"Main config: C=16/32/32, D=128 RFF, K=4, rank R=16, {n_seeds} seeds (mean ± std).", "",
         f"ID test accuracy: {summary['id_accuracy']['mean']:.4f} ± {summary['id_accuracy']['std']:.4f}. "
         f"Windows flagged unknown (TV = 99th percentile of validation V): "
         f"test {summary['id_unknown_rate']['mean']*100:.2f}%, val {summary['id_val_unknown_rate']['mean']*100:.2f}%. "
         "The test-normal recording (1 kHz logger, sensor nearly still) is itself atypical, so "
         "OOD AUROC is given against both ID references.", ""]
    f = lambda k: f"{summary[k]['mean']:.3f} ± {summary[k]['std']:.3f}"
    for sfx, title in (("_vs_val", "ID = validation windows (clean reference)"),
                       ("", "ID = test windows (includes the atypical test-normal recording)")):
        L += [f"## OOD AUROC, {title}", "",
              "| OOD set | B-SNGP V (int, R=16) | full-rank float var | Mahalanobis on h | -abs(logit) | FPR95 (B-SNGP) |",
              "|---|---|---|---|---|---|"]
        for g in groups:
            L.append(f"| {g} | {f(g+'/bsngp_var'+sfx+'/auroc')} | {f(g+'/reference_full_rank_float_var'+sfx+'/auroc')} | "
                     f"{f(g+'/baseline_mahalanobis_h'+sfx+'/auroc')} | {f(g+'/baseline_neg_abs_logit'+sfx+'/auroc')} | "
                     f"{summary[g+'/bsngp_var'+sfx+'/fpr95']['mean']*100:.1f}% |")
        L.append("")
    L += ["## Fraction of OOD windows flagged unknown by the hardware threshold TV", "",
          "| OOD set | flagged |", "|---|---|"]
    for g in groups:
        L.append(f"| {g} | {summary[g+'/flagged_unknown_rate']['mean']*100:.1f}% |")
    L += ["", "## Rank ablation (seed 0, AUROC of integer V vs validation ID)", "",
          "| R | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for r, m in rank_abl.items():
        L.append(f"| {r} | " + " | ".join(f"{m[g]:.3f}" for g in groups) + " |")
    L += ["", "## RFF frequency sweep (K = 2^ksh, AUROC vs validation ID; sensitivity only, not used for model selection)", "",
          "| K | ID acc | " + " | ".join(groups) + " |", "|---|---|" + "---|" * len(groups)]
    for k, s in ksweep.items():
        L.append(f"| {k} | {s['id_accuracy']['mean']:.3f} | " +
                 " | ".join(f"{s[g+'/bsngp_var_vs_val/auroc']['mean']:.3f}" for g in groups) + " |")
    L += ["", "## Hardware operation counts (seed 0)", "", "```", json.dumps(ops, indent=2), "```", ""]
    open(path, "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ksh", type=int, default=2)
    ap.add_argument("--sweep_seeds", type=int, default=3)
    args = ap.parse_args()

    device = init_hardware(require_cuda=True)
    os.makedirs(EXPORT, exist_ok=True)
    t0 = time.time()

    splits = data.id_splits()
    rn, ra = data.test_recordings()
    faults = data.sensor_faults([rn, ra])
    online = data.online_ood()
    ood_sets = {k: v for k, v in online.items() if not k.endswith("_raw_first")}
    ood_sets.update({"fault_" + k: v for k, v in faults.items()})
    for k, (X, y) in splits.items():
        print(f"{k}: {X.shape} normal={int((y == 0).sum())} abnormal={int((y == 1).sum())}")
    for k, X in ood_sets.items():
        print(f"OOD {k}: {X.shape}")

    # Main configuration, several seeds
    seed_results, main_model, main_params = [], None, None
    for s in range(args.seeds):
        model, vl = train_model(splits, device, args.ksh, s, args.epochs)
        p, res, _ = build_and_eval(model, splits, ood_sets, args.rank, device)
        res["seed"], res["val_loss"] = s, vl
        seed_results.append(res)
        print(f"[seed {s}] val_loss={vl:.4f} id_acc={res['id_test']['accuracy']:.4f} "
              + " ".join(f"{k}={v['bsngp_var']['auroc']:.3f}" for k, v in res["ood"].items()))
        if s == 0:  # hardware model is fixed in advance as seed 0 (no selection on OOD results)
            main_model, main_params = model, p
    summary = summarize_seeds(seed_results)

    rank_abl = rank_ablation(main_model, splits, ood_sets, device)

    ksweep = {}
    for ksh in sorted({0, 1, 2, 3} - {args.ksh}) + [args.ksh]:
        if ksh == args.ksh:
            ksweep[1 << ksh] = summarize_seeds(seed_results[:args.sweep_seeds])
            continue
        rs = []
        for s in range(args.sweep_seeds):
            m, _ = train_model(splits, device, ksh, s, args.epochs)
            rs.append(build_and_eval(m, splits, ood_sets, args.rank, device)[1])
        ksweep[1 << ksh] = summarize_seeds(rs)
    ksweep = dict(sorted(ksweep.items()))

    # Export hardware model + co-simulation vectors
    np.savez(os.path.join(EXPORT, "params.npz"), **main_params)
    stream, segs = data.cosim_stream(online)
    exp = golden.run(main_params, golden.stream_windows(stream))
    np.save(os.path.join(EXPORT, "stream.npy"), stream)
    np.savez(os.path.join(EXPORT, "expected_stream.npz"), **exp)
    demo, demo_segs = data.demo_rom_stream()
    # The demo loops: evaluate on two concatenated loops so wrap-around windows are covered.
    demo_exp = golden.run(main_params, golden.stream_windows(np.concatenate([demo, demo])))
    np.save(os.path.join(EXPORT, "demo_stream.npy"), demo)
    np.savez(os.path.join(EXPORT, "demo_expected.npz"), **demo_exp)

    ops = ops_estimate(main_params)
    out = {
        "config": vars(args),
        "summary": summary,
        "per_seed": seed_results,
        "rank_ablation_seed0": rank_abl,
        "ksweep": ksweep,
        "ops": ops,
        "stream_segments": segs,
        "demo_segments": demo_segs,
        "hardware_model": {"tl": int(main_params["tl"]), "tv": int(main_params["tv"]),
                           "logit_scale": float(main_params["logit_scale"]),
                           "ridge": float(main_params["ridge"])},
        "runtime_s": time.time() - t0,
    }
    json.dump(out, open(os.path.join(EXPORT, "results.json"), "w"), indent=2, default=float)
    write_markdown(summary, ksweep, rank_abl, ops, args.seeds, os.path.join(EXPORT, "results.md"))
    print(open(os.path.join(EXPORT, "results.md")).read())
    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
