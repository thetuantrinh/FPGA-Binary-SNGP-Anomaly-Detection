#!/usr/bin/env python3
"""
Fair comparison of uncertainty methods on the same binary backbone, data, seeds and OOD sets.

Methods (all share the 3-layer binary conv backbone of bsngp_model.BinarySNGP):
  deterministic   linear head on h; score = predictive entropy
  mc_dropout      dropout (p=0.2) on layer-2 bits and on h, T=20 passes; score = mutual information
  deep_ensemble   M=5 independently trained deterministic BNNs; score = mutual information
  mahalanobis     DDU-style class-conditional Gaussian on h of the deterministic BNN; score = min distance
  bsngp           Binary-SNGP, integer variance V exactly as computed by the FPGA

Settings:
  original  per-axis normalisation, no augmentation (as export/)
  robust    isotropic normalisation + DC removal + rotation augmentation (as export_robust/)

Run from the repository root (CUDA GPU required):
  python bsngp/compare_uncertainty.py
"""

import os
import sys
import json
import copy
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

from anomaly_detection.utils import init_hardware, set_seed
from anomaly_detection.bnn_models import binarize

import golden
import data
import orientation as ori
from bsngp_model import BinarySNGP
from train_eval import train_model, build_and_eval, ood_metrics, mahalanobis_scores

OUT = os.path.join(HERE, "export_compare")
METHODS = ["deterministic", "mc_dropout", "deep_ensemble", "mahalanobis", "bsngp"]


class LinearHeadBNN(BinarySNGP):
    """Same binary backbone, plain linear output on h. p_drop > 0 gives the MC-Dropout model."""

    def __init__(self, p_drop=0.0, **kw):
        super().__init__(**kw)
        self.p_drop = p_drop
        self.mc = False
        self.head = nn.Linear(self.c3, 1)

    def features(self, x_int):
        if self.dc_remove:
            x_int = x_int - torch.div(x_int.sum(dim=-1, keepdim=True), 64, rounding_mode="floor")
        drop = self.training or self.mc
        x = x_int / golden.INPUT_SCALE
        b1 = binarize(self.bn1(self.conv1(x)))
        b2 = binarize(self.bn2(self.conv2(b1)))
        b2 = F.dropout(b2, self.p_drop, training=drop)
        b3 = binarize(self.bn3(self.conv3(b2)))
        h = b3.sum(dim=-1)
        return F.dropout(h, self.p_drop, training=drop) if self.p_drop > 0 else h, None

    def forward(self, x_int):
        h, _ = self.features(x_int)
        return self.head(h / 16.0).squeeze(-1)


def train(make_model, splits, device, seed, epochs, lr=1e-3, batch=128):
    set_seed(seed)  # before construction, so the initialisation depends on the seed
    model = make_model().to(device)
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(device)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = F.binary_cross_entropy_with_logits(model(Xtr_t[b]), ytr_t[b], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(model(Xva_t), yva_t, pos_weight=pw).item()
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model


def bin_entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@torch.no_grad()
def probs(model, X, device, passes=1):
    """(passes, N) probabilities; MC dropout active when passes > 1."""
    model.mc = passes > 1
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    out = np.stack([torch.sigmoid(model(xt)).cpu().numpy() for _ in range(passes)])
    model.mc = False
    return out


@torch.no_grad()
def feats(model, X, device):
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    return model.features(xt)[0].cpu().numpy().astype(np.float64)


def scores_from_samples(P):
    """P: (S, N) member/pass probabilities -> mean prob, total entropy, mutual information."""
    pm = P.mean(0)
    tot = bin_entropy(pm)
    mi = tot - bin_entropy(P).mean(0)
    return pm, tot, mi


def ece(y, p, bins=15):
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    e, edges = 0.0, np.linspace(0.5, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def method_outputs(models, splits, sets, device, args):
    """Returns per method: dict set_name -> (score, prob) arrays; higher score = more uncertain."""
    det, mc, ens, sngp_p, Xtr_aug, ytr_aug = models
    out = {m: {} for m in METHODS}
    htr = feats(det, Xtr_aug, device)
    for name, X in sets.items():
        p = probs(det, X, device)[0]
        out["deterministic"][name] = (bin_entropy(p), p)
        pm, tot, mi = scores_from_samples(probs(mc, X, device, passes=args.mc_passes))
        out["mc_dropout"][name] = (mi, pm)
        pm, tot, mi = scores_from_samples(np.stack([probs(m, X, device)[0] for m in ens]))
        out["deep_ensemble"][name] = (mi, pm)
        out["mahalanobis"][name] = (mahalanobis_scores(htr, ytr_aug, feats(det, X, device)), p)
        g = golden.run(sngp_p, X)
        ps = 1.0 / (1.0 + np.exp(-(g["logit"] / float(sngp_p["logit_scale"]))
                                 / np.sqrt(1 + np.pi / 8 * np.maximum(g["var"], 0) / (127.0 ** 2))))
        out["bsngp"][name] = (g["var"].astype(np.float64), ps)
    return out


def hw_cost(dc_remove, args, c1=16, c2=32, c3=32, d=128, r=16):
    """Cycles per window on the sequential engine of bsngp_core.vhd (bsngp = measured in RTL)."""
    dc = 193 if dc_remove else 0
    l12, l3 = (64 + 32) * 10, 16 * 10
    lin = c3 + 4
    w_bb = c1 * 6 * 3 + c2 * c1 * 3 + c3 * c2 * 3
    return {
        "deterministic": {"cycles": dc + l12 + l3 + lin, "passes": 1, "weight_bits": w_bb + 8 * (c3 + 1),
                          "multipliers": 1, "note": "estimate"},
        "mc_dropout": {"cycles": dc + l12 + args.mc_passes * (l3 + lin), "passes": args.mc_passes,
                       "weight_bits": w_bb + 8 * (c3 + 1), "multipliers": 1,
                       "note": "estimate; L1-L2 shared, L3+head repeated; needs an on-chip RNG"},
        "deep_ensemble": {"cycles": args.members * (dc + l12 + l3 + lin), "passes": args.members,
                          "weight_bits": args.members * (w_bb + 8 * (c3 + 1)), "multipliers": 1,
                          "note": "estimate; M full networks run sequentially"},
        "mahalanobis": {"cycles": dc + l12 + l3 + lin + 2 * (c3 + 2), "passes": 1,
                        "weight_bits": w_bb + 8 * (c3 + 1) + 16 * (2 * c3 + c3 * c3), "multipliers": c3 + 1,
                        "note": "estimate; 32 parallel MACs for (h-mu)^T P (h-mu)"},
        "bsngp": {"cycles": (1971 if dc_remove else 1778), "passes": 1,
                  "weight_bits": w_bb + d * c3 + 10 * d + 8 * d * (r + 1) + 8 * 1024, "multipliers": r + 3,
                  "note": "measured in RTL simulation"},
    }


def prepare(setting, args):
    raw = data.id_splits_raw()
    (Rtr, ytr), (Rva, yva), (Rte, yte) = raw["train"], raw["val"], raw["test"]
    g_train = ori.gravity_direction(Rtr)
    if setting == "robust":
        data.set_norm(*ori.isotropic_stats(Rtr))
        Atr, aytr = ori.augment(Rtr, ytr, 7, 0.3, seed=1)
        Ava, ayva = ori.augment(Rva, yva, 3, 0.3, seed=2)
    else:
        data._NORM = None
        data.set_norm(*data.norm_stats())
        Atr, aytr, Ava, ayva = Rtr, ytr, Rva, yva
    q = data.quantize_raw_windows
    splits = {"train": (q(Atr), aytr), "val": (q(Ava), ayva), "test": (q(Rte), yte)}
    rn, ra = data.test_recordings()
    ood = {k: v for k, v in data.online_ood().items() if not k.endswith("_raw_first")}
    ood.update({"fault_" + k: v for k, v in data.sensor_faults([rn, ra]).items()
                if not (setting == "robust" and k == "axis_swap")})
    field_normal = q(ori.field_sim(Rva[yva == 0], g_train, dead_gyro_axis=0))
    return splits, ood, field_normal


def run_setting(setting, device, args):
    splits, ood, field_normal = prepare(setting, args)
    dc = setting == "robust"
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]
    sets = {"val": Xva, "test": Xte, "field_normal": field_normal, **ood}
    per_seed = []
    for s in range(args.seeds):
        t0 = time.time()
        ens = [train(lambda: LinearHeadBNN(seed=s, dc_remove=dc), splits, device, 1000 * s + m, args.epochs)
               for m in range(args.members)]
        det = ens[0]
        mc = train(lambda: LinearHeadBNN(p_drop=0.2, seed=s, dc_remove=dc), splits, device, 500 + s, args.epochs)
        sn, _ = train_model(splits, device, 2, s, args.epochs, dc_remove=dc)
        sngp_p, _, _ = build_and_eval(sn, splits, {}, 16, device)
        outs = method_outputs((det, mc, ens, sngp_p, *splits["train"]), splits, sets, device, args)
        res = {}
        for m in METHODS:
            o = outs[m]
            s_val = o["val"][0]
            tv = np.percentile(s_val, 99)
            r = {"id_test_acc": float(accuracy_score(yte, o["test"][1] >= 0.5)),
                 "id_test_ece": ece(yte, o["test"][1]),
                 "field_normal_abnormal_rate": float((o["field_normal"][1] >= 0.5).mean()),
                 "field_normal_unknown_rate": float((o["field_normal"][0] > tv).mean()),
                 "ood": {}}
            for name in ood:
                mm = ood_metrics(s_val, o[name][0])
                mm["flagged_at_99pct"] = float((o[name][0] > tv).mean())
                r["ood"][name] = mm
            r["mean_auroc"] = float(np.mean([v["auroc"] for v in r["ood"].values()]))
            res[m] = r
        per_seed.append(res)
        print(f"[{setting} seed {s}] {time.time()-t0:.0f}s  " +
              "  ".join(f"{m}={res[m]['mean_auroc']:.3f}" for m in METHODS), flush=True)
    return per_seed, list(ood), hw_cost(dc, args)


def agg(per_seed, m, key_fn):
    v = np.array([key_fn(r[m]) for r in per_seed])
    return float(v.mean()), float(v.std())


def write_md(all_res, path, args):
    L = ["# Uncertainty methods on the same binary backbone (5 seeds, mean ± std)", "",
         f"ID reference = validation windows; 'unknown' threshold = 99th percentile of the method's score on "
         f"validation. MC Dropout: T={args.mc_passes}, p=0.2. Deep Ensemble: M={args.members}.", ""]
    names = {"deterministic": "Deterministic BNN (entropy)", "mc_dropout": f"MC Dropout (T={args.mc_passes}, MI)",
             "deep_ensemble": f"Deep Ensemble (M={args.members}, MI)", "mahalanobis": "Mahalanobis / DDU on h",
             "bsngp": "**B-SNGP (ours, integer V)**"}
    for setting, (per_seed, ood_names, hw) in all_res.items():
        L += [f"## Setting: {setting}", "",
              "| Method | Mean OOD AUROC | " + " | ".join(ood_names) + " |",
              "|---|---|" + "---|" * len(ood_names)]
        for m in METHODS:
            mu, sd = agg(per_seed, m, lambda r: r["mean_auroc"])
            cells = [f"{agg(per_seed, m, lambda r, n=n: r['ood'][n]['auroc'])[0]:.3f}" for n in ood_names]
            L.append(f"| {names[m]} | {mu:.3f} ± {sd:.3f} | " + " | ".join(cells) + " |")
        L += ["", "| Method | ID acc | ID ECE | field-mount normal: abnormal | field-mount normal: unknown | "
              "OOD flagged @99% (mean) | cycles/window | passes | weight bits | multipliers |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            acc = agg(per_seed, m, lambda r: r["id_test_acc"])[0]
            e = agg(per_seed, m, lambda r: r["id_test_ece"])[0]
            fa = agg(per_seed, m, lambda r: r["field_normal_abnormal_rate"])[0]
            fu = agg(per_seed, m, lambda r: r["field_normal_unknown_rate"])[0]
            fl = agg(per_seed, m, lambda r: np.mean([v["flagged_at_99pct"] for v in r["ood"].values()]))[0]
            h = hw[m]
            L.append(f"| {names[m]} | {acc:.3f} | {e*100:.2f}% | {fa*100:.1f}% | {fu*100:.1f}% | {fl*100:.1f}% | "
                     f"{h['cycles']:,} ({h['note'].split(';')[0]}) | {h['passes']} | {h['weight_bits']:,} | {h['multipliers']} |")
        L.append("")
    open(path, "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--mc_passes", type=int, default=20)
    ap.add_argument("--settings", default="robust,original")
    args = ap.parse_args()
    device = init_hardware(require_cuda=True)
    os.makedirs(OUT, exist_ok=True)
    all_res = {}
    for setting in args.settings.split(","):
        all_res[setting] = run_setting(setting, device, args)
        json.dump({k: {"per_seed": v[0], "ood_sets": v[1], "hw": v[2]} for k, v in all_res.items()},
                  open(os.path.join(OUT, "results.json"), "w"), indent=2)
        write_md(all_res, os.path.join(OUT, "results.md"), args)
    print(open(os.path.join(OUT, "results.md")).read())


if __name__ == "__main__":
    main()
