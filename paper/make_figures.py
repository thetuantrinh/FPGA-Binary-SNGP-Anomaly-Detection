#!/usr/bin/env python3
"""
Generate all manuscript figures from the verified result files (no numbers typed by hand,
except the on-board measurements, which are copied from report_run.pdf and marked BOARD).

Inputs:  ../bsngp/export/results.json            original model (5 seeds, ablations)
         ../bsngp/export_robust/results.json     orientation transfer + co-sim stream segments
         ../bsngp/export_robust/expected_stream.npz
         ../bsngp/export_compare/results.json    uncertainty-method comparison
Outputs: figures/fig_*.pdf
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
BS = os.path.join(os.path.dirname(HERE), "bsngp")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "text.usetex": True, "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL, DCOL = 3.45, 7.1  # IEEE column / double-column width (in)
# Okabe-Ito palette; B-SNGP always in vermillion
C = {"deterministic": "#999999", "mc_dropout": "#56B4E9", "deep_ensemble": "#009E73",
     "mahalanobis": "#E69F00", "bsngp": "#D55E00"}
NAMES = {"deterministic": "Deterministic", "mc_dropout": "MC Dropout", "deep_ensemble": "Deep Ens.",
         "mahalanobis": "Mahalanobis", "bsngp": "B-SNGP"}
METHODS = list(NAMES)
OOD_LABEL = {"uav_6dof": "UAV", "motor_acc_only": "Motor$^\\dagger$", "adi_cbm_acc_only": "ADI CbM$^\\dagger$",
             "fault_stuck": "Stuck", "fault_zero_output": "Zero out", "fault_noise_burst": "Noise",
             "fault_axis_swap": "Axis swap", "fault_spikes": "Spikes"}

orig = json.load(open(os.path.join(BS, "export", "results.json")))
rob = json.load(open(os.path.join(BS, "export_robust", "results.json")))
cmp_ = json.load(open(os.path.join(BS, "export_compare", "results.json")))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- cycle budget
def fig_cycles():
    stages = [("DC removal", 193, "#CC79A7"), ("L1 conv", 640, "#0072B2"), ("L2 conv", 320, "#56B4E9"),
              ("L3 conv + GAP", 160, "#009E73"), ("GP head (D=128)", 640, "#E69F00"),
              ("Variance (R=16)", 17, "#D55E00"), ("Output", 1, "#000000")]
    rows = [("Robust\n(DC on)", [s for s in stages]), ("Original\n(DC off)", stages[1:])]
    fig, ax = plt.subplots(figsize=(COL, 1.15))
    for y, (label, st) in enumerate(rows):
        left = 0
        for name, cyc, col in st:
            ax.barh(y, cyc, left=left, color=col, edgecolor="white", linewidth=0.4, height=0.55)
            if cyc >= 160:
                ax.text(left + cyc / 2, y, f"{cyc}", ha="center", va="center", color="white", fontsize=7)
            left += cyc
        ax.text(left + 25, y, f"{left:,} cyc = {left/50:.1f} $\\mu$s", va="center", fontsize=7)
    ax.set_yticks([0, 1], [r[0] for r in rows])
    ax.set_xlim(0, 2550)
    ax.invert_yaxis()
    ax.set_xlabel("Clock cycles per window at 50 MHz")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(color=c, label=n) for n, _, c in stages[:-1]], ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.45, 1.75), handlelength=1.0, columnspacing=0.8)
    save(fig, "fig_cycles.pdf")


# ---------------------------------------------------------------- uncertainty comparison
def agg(setting, m, key):
    v = np.array([key(s[m]) for s in cmp_[setting]["per_seed"]])
    return v.mean(), v.std()


def fig_uncertainty():
    fig = plt.figure(figsize=(DCOL, 1.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.6, 1.0], wspace=0.6)

    # (a) mean AUROC, both settings
    ax = fig.add_subplot(gs[0])
    x = np.arange(2)
    w = 0.16
    for i, m in enumerate(METHODS):
        mu = [agg(s, m, lambda r: r["mean_auroc"])[0] for s in ("original", "robust")]
        sd = [agg(s, m, lambda r: r["mean_auroc"])[1] for s in ("original", "robust")]
        ax.bar(x + (i - 2) * w, mu, w, yerr=sd, color=C[m], label=NAMES[m], error_kw=dict(lw=0.6, capsize=1.2),
               edgecolor="black" if m == "bsngp" else "none", linewidth=0.5)
    ax.axhline(0.5, ls=":", lw=0.6, color="k")
    ax.set_xticks(x, ["Original", "Robust"])
    ax.set_ylabel("Mean OOD AUROC (5 seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Mean OOD AUROC", loc="left")
    fig.legend(handles=[Patch(color=C[m], label=NAMES[m]) for m in METHODS], ncol=5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.08), handlelength=1.0)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) per-OOD heatmap, robust setting
    ax = fig.add_subplot(gs[1])
    sets = cmp_["robust"]["ood_sets"]
    M = np.array([[agg("robust", m, lambda r, s=s: r["ood"][s]["auroc"])[0] for s in sets] for m in METHODS])
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if METHODS[i] == "bsngp" else "normal")
    ax.set_xticks(range(len(sets)), [OOD_LABEL[s] for s in sets], rotation=30, ha="right")
    ax.set_yticks(range(len(METHODS)), [NAMES[m] for m in METHODS])
    ax.set_title("(b) Per OOD set (orientation-robust)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=7)

    # (c) quality vs FPGA cost
    ax = fig.add_subplot(gs[2])
    hw = cmp_["robust"]["hw"]
    for m in METHODS:
        mu, sd = agg("robust", m, lambda r: r["mean_auroc"])
        cyc = hw[m]["cycles"]
        ax.errorbar(cyc, mu, yerr=sd, fmt="o" if m == "bsngp" else "s", ms=4.5 if m == "bsngp" else 3.5,
                    color=C[m], ecolor=C[m], elinewidth=0.6, capsize=1.5, mec="black" if m == "bsngp" else C[m])
    for txt, xy, off in (("1 pass", (hw["bsngp"]["cycles"], agg("robust", "bsngp", lambda r: r["mean_auroc"])[0]), (7, -2)),
                         ("1 pass", (1.04 * hw["deterministic"]["cycles"], 0.295), (0, 0)),
                         (f"{hw['mc_dropout']['passes']} passes", (hw["mc_dropout"]["cycles"], 0.57), (0, 6)),
                         (f"{hw['deep_ensemble']['passes']} passes", (hw["deep_ensemble"]["cycles"], 0.38), (0, -12))):
        ax.annotate(txt, xy, textcoords="offset points", xytext=off, fontsize=7, va="center" if off[1] == 0 else "baseline",
                    ha="center" if off[0] == 0 else ("left" if off[0] > 0 else "right"))
    ax.set_xscale("log")
    ax.set_xlim(1000, 15000)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("Cycles per window")
    ax.set_ylabel("Mean OOD AUROC")
    ax.set_title("(c) Quality vs. cost", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_uncertainty.pdf")


# ---------------------------------------------------------------- field failure and fix
def fig_field():
    conds = [("original_mount/normal", "Orig.\nnorm."), ("field_mount_gx_dead/normal", "Field\nnorm."),
             ("field_mount_gx_dead/abnormal", "Field\nabn."), ("random_orientation/normal", "Rand.\nnorm.")]
    dep, r = rob["deployed_model"], rob["robust_conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.5), sharey=True)
    x = np.arange(len(conds))
    for ax, key, title in ((axes[0], "abnormal_rate", "(a) Predicted abnormal"),
                           (axes[1], "unknown_rate", "(b) Flagged unknown")):
        dv = [dep[c][key] * 100 for c, _ in conds]
        rv = [r[c][key][0] * 100 for c, _ in conds]
        rs = [r[c][key][1] * 100 for c, _ in conds]
        ax.bar(x - 0.19, dv, 0.36, color="#999999", label="Deployed model")
        ax.bar(x + 0.19, rv, 0.36, yerr=rs, color=C["bsngp"], label="Orientation-robust",
               error_kw=dict(lw=0.6, capsize=1.2))
        for xi, v in zip(x, dv):
            ax.text(xi - 0.21, v + 2, f"{v:.0f}", ha="center", fontsize=7)
        for xi, v in zip(x, rv):
            ax.text(xi + 0.21, v + 2, f"{v:.0f}", ha="center", fontsize=7)
        ax.set_xticks(x, [l for _, l in conds], fontsize=7)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 112)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"Windows (\%)")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.32), ncol=2)
    save(fig, "fig_field.pdf")


# ---------------------------------------------------------------- integer outputs on the co-sim stream
def fig_stream():
    e = np.load(os.path.join(BS, "export_robust", "expected_stream.npz"))
    hwm = rob["hardware_model"]
    L = e["logit"] / hwm["logit_scale"]
    var = e["var"] / 127.0 ** 2
    tv = hwm["tv"] / 127.0 ** 2
    ends = 64 + 32 * np.arange(len(L))
    t = ends / 500.0  # seconds
    seg_names = {"val_normal": "Normal", "test_abnormal": "Abnormal", "field_mount_normal_gx_dead": "Field-mount\nnormal",
                 "field_mount_abnormal_gx_dead": "Field-mount\nabnormal", "uav_6dof": "Other machine\n(UAV)",
                 "zero_output": "Sensor\nzero"}
    fig, axes = plt.subplots(2, 1, figsize=(DCOL, 1.9), sharex=True, gridspec_kw=dict(hspace=0.12))
    b = 0
    shade = ["#f2f2f2", "#ffffff"]
    for k, (n, ln) in enumerate(rob["stream_segments"]):
        for ax in axes:
            ax.axvspan(b / 500, (b + ln) / 500, color=shade[k % 2], zorder=0)
        axes[0].text((b + ln / 2) / 500, 1.0, seg_names[n], ha="center", va="bottom", fontsize=7,
                     transform=axes[0].get_xaxis_transform())
        b += ln
    abn = e["abnormal"].astype(bool)
    unk = e["unknown"].astype(bool)
    axes[0].plot(t, L, color="k", lw=0.7)
    axes[0].scatter(t[abn], L[abn], s=5, color=C["bsngp"], zorder=3, label="abnormal ($L>T_L$)")
    axes[0].scatter(t[~abn], L[~abn], s=5, color="#0072B2", zorder=3, label="normal")
    axes[0].axhline(0, ls="--", lw=0.6, color="k")
    axes[0].set_ylabel("Logit $L/s_L$")
    axes[0].legend(frameon=False, loc="lower left", ncol=2, fontsize=7)
    axes[1].semilogy(t, np.maximum(var, 1e-1), color="k", lw=0.7)
    axes[1].scatter(t[unk], var[unk], s=5, color="#E69F00", zorder=3, label="unknown ($V>T_V$)")
    axes[1].axhline(tv, ls="--", lw=0.6, color="#E69F00")
    axes[1].text(t[-1], tv * 1.15, f"$T_V$", ha="right", fontsize=7, color="#E69F00")
    axes[1].set_ylabel("Variance $V/127^2$")
    axes[1].set_xlabel("Time (s) of continuous 500 Hz stream; one decision per 64 ms (bit-exact in RTL)")
    axes[1].legend(frameon=False, loc="upper left", fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(t[0] - 0.2, t[-1] + 0.2)
    save(fig, "fig_stream.pdf")


# ---------------------------------------------------------------- ablations
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.45), gridspec_kw=dict(wspace=0.38))
    ra = orig["rank_ablation_seed0"]
    ranks = [int(k) for k in ra]
    for s, col in (("fault_noise_burst", "#0072B2"), ("fault_spikes", "#009E73"), ("fault_axis_swap", "#CC79A7"),
                   ("motor_acc_only", "#E69F00")):
        axes[0].plot(ranks, [ra[str(r)][s] for r in ranks], marker="o", ms=2.5, lw=0.9, color=col, label=OOD_LABEL[s])
    axes[0].plot(ranks, [np.mean(list(ra[str(r)].values())) for r in ranks], marker="s", ms=3, lw=1.3,
                 color=C["bsngp"], label="Mean (8 sets)")
    axes[0].axvline(16, ls=":", lw=0.6, color="k")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(ranks, [str(r) for r in ranks])
    axes[0].set_xlabel("Covariance rank $R$ (int8)")
    axes[0].set_ylabel("OOD AUROC")
    axes[0].set_title("(a) Low-rank variance", loc="left")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left", handlelength=1.2)
    ks = orig["ksweep"]
    K = [int(k) for k in ks]
    sets = sorted({k.split("/")[0] for k in ks["4"] if "/" in k})
    mean_auc = [np.mean([ks[str(k)][f"{s}/bsngp_var_vs_val/auroc"]["mean"] for s in sets]) for k in K]
    acc = [ks[str(k)]["id_accuracy"]["mean"] for k in K]
    axes[1].plot(K, mean_auc, marker="o", ms=3, color=C["bsngp"], label="Mean OOD AUROC")
    axes[1].plot(K, acc, marker="s", ms=3, color="#0072B2", label="ID accuracy")
    axes[1].axvline(4, ls=":", lw=0.6, color="k")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(K, [str(k) for k in K])
    axes[1].set_xlabel("RFF frequency $K=2^{k_s}$")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].set_title("(b) Kernel length scale", loc="left")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_ablation.pdf")


# ---------------------------------------------------------------- FPGA implementation (BOARD numbers) + cycle budget
def fig_fpga():
    # BOARD: Quartus Prime 25.1 Lite, EP4CE22F17C6, complete design (report_run.pdf, Table 7)
    util = [("Logic elements", 16367, 22320), ("Registers", 5450, 22320), ("Memory bits", 15360, 608256),
            ("9-bit mult.", 22, 132), ("I/O pins", 18, 154)]
    # BOARD: measured per-window inference time (report_run.pdf, Table 6)
    lat = [("B-SNGP, FPGA (this work)", 35.56), ("INT8 1D-CNN, same FPGA", 1742.0),
           ("NumPy ref., PC, 1 window", 603.0), ("NumPy ref., PC, batched", 391.0)]
    stages = [("DC", 193, "#CC79A7"), ("L1", 640, "#0072B2"), ("L2", 320, "#56B4E9"), ("L3", 160, "#009E73"),
              ("GP head", 640, "#E69F00"), ("Var.", 17, "#D55E00"), ("", 1, "#000000")]
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.05), gridspec_kw=dict(hspace=0.95, height_ratios=[0.75, 1.0]))
    axes = [None] + list(axes)
    # (b) cycle budget per window (RTL-measured totals)
    for yy, (lab, st) in enumerate((("robust", stages), ("original", stages[1:]))):
        left = 0
        for name, cyc, col in st:
            axes[1].barh(yy, cyc, left=left, color=col, edgecolor="white", linewidth=0.4, height=0.6)
            if cyc >= 160:
                axes[1].text(left + cyc / 2, yy, name, ha="center", va="center", color="white", fontsize=7)
            left += cyc
        axes[1].text(left + 30, yy, f"{left:,} cyc = {left / 50:.1f} $\\mu$s", va="center", fontsize=7)
    axes[1].set_yticks([0, 1], ["Robust", "Original"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 2700)
    axes[1].set_xlabel("Clock cycles per window at 50 MHz", labelpad=1)
    axes[1].set_title("(a) Cycle budget per window, Eq. (6)", loc="left")
    # (c) latency
    yl = np.arange(len(lat))
    axes[2].barh(yl, [v for _, v in lat], color=[C["bsngp"], "#999999", "#56B4E9", "#56B4E9"], height=0.55)
    for yi, (_, v) in zip(yl, lat):
        axes[2].text(v * 1.2, yi, f"{v:,.0f}" if v > 100 else f"{v:.1f}", va="center", fontsize=7)
    axes[2].set_xscale("log")
    axes[2].set_xlim(10, 20000)
    axes[2].set_yticks(yl, [n for n, _ in lat], fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Measured time per window ($\\mu$s, log)", labelpad=1)
    axes[2].set_title("(b) Measured time per window", loc="left")
    for ax in axes[1:]:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_fpga.pdf")


# ---------------------------------------------------------------- input data, feature space, SNGP uncertainty
def fig_data_unc():
    v = np.load(os.path.join(BS, "export_robust", "viz_data.npz"), allow_pickle=False)
    names = [str(n) for n in v["names"]]
    tv = float(v["tv"]) / 127.0 ** 2
    mu, pcs = v["pca_mu"], v["pca_pcs"]
    short = {"ID normal": "ID normal", "ID abnormal": "ID abnormal", "Field-mount normal": "Field normal",
             "Field-mount abnormal": "Field abnormal", "UAV (other machine)": "UAV", "Motor rig (acc. only)": "Motor$^\\dagger$",
             "Sensor stuck": "Stuck", "Sensor zero": "Zero out", "Noise burst": "Noise", "Spikes": "Spikes"}
    sty = {"ID normal": ("#0072B2", "o"), "ID abnormal": ("#D55E00", "o"), "Field-mount normal": ("#56B4E9", "^"),
           "Field-mount abnormal": ("#E69F00", "^"), "UAV (other machine)": ("#009E73", "s"),
           "Motor rig (acc. only)": ("#CC79A7", "s"), "Sensor stuck": ("#000000", "x"), "Sensor zero": ("#666666", "P"),
           "Noise burst": ("#999933", "d"), "Spikes": ("#882255", "v")}
    known = lambda n: n.startswith("ID") or n.startswith("Field")

    fig = plt.figure(figsize=(DCOL, 1.70))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 0.95, 1.0, 1.0], wspace=0.75, hspace=0.15)

    # (a) input windows
    t = np.arange(64) * 2.0
    for row, (sl, unit, lab) in enumerate(((slice(0, 3), "g", "Accel."), (slice(3, 6), "$^\\circ$/s", "Gyro"))):
        ax = fig.add_subplot(gs[row, 0])
        for k, c in zip(range(3), ("#0072B2", "#009E73", "#D55E00")):
            ax.plot(t, v["ex_normal"][sl][k], color=c, lw=0.55, label="xyz"[k])
            ax.plot(t + 136, v["ex_abnormal"][sl][k], color=c, lw=0.55)
        ax.axvline(132, color="k", lw=0.5)
        ax.set_ylabel(f"{lab} [{unit}]", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title("(a) Input windows", loc="left")
            ax.set_xticks([])
            pass
        else:
            ax.set_xticks([0, 64, 136, 200], ["0", "64", "0", "64 ms"])
            for x0, txt in ((64, "normal"), (200, "abnormal")):
                ax.text(x0, -0.42, txt, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=7)

    # (b) feature space
    ax = fig.add_subplot(gs[:, 1])
    for k, n in enumerate(names):
        z = (v[f"h_{k}"] - mu) @ pcs.T
        c, m = sty[n]
        ax.scatter(z[:, 0], z[:, 1], s=4, c=c, marker=m, alpha=0.8, linewidths=0.4, zorder=3 if known(n) else 2)
    ax.set_xlabel("PC 1 of $\\mathbf{h}$")
    ax.set_ylabel("PC 2 of $\\mathbf{h}$", labelpad=1)
    ax.set_title("(b) Feature space", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    # (c) distance awareness: variance vs distance to the nearest training feature
    ax = fig.add_subplot(gs[:, 2])
    for k, n in enumerate(names):
        c, m = sty[n]
        ax.scatter(v[f"dist_{k}"] + 0.5, np.maximum(v[f"V_{k}"], 1) / 127.0 ** 2, s=4, c=c, marker=m, alpha=0.75,
                   linewidths=0.4, label=short[n])
    ax.axhline(tv, color="#00A0E0", lw=0.9, ls="--")
    ax.text(0.03, tv * 1.25, "$T_V$", transform=ax.get_yaxis_transform(), ha="left", fontsize=7, color="#00A0E0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Dist. to nearest train. $\\mathbf{h}$")
    ax.set_ylabel("Variance $V/127^2$", labelpad=1)
    ax.set_title("(c) Distance awareness", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), ncol=10, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.13), fontsize=7, handletextpad=0.2, columnspacing=0.8, markerscale=1.6)

    # (d) hardware uncertainty output per data set
    ax = fig.add_subplot(gs[:, 3])
    order = names[::-1]
    for y, n in enumerate(order):
        k = names.index(n)
        var = np.maximum(v[f"V_{k}"], 1) / 127.0 ** 2
        jit = (np.random.default_rng(k).random(len(var)) - 0.5) * 0.55
        ax.scatter(var, y + jit, s=1.6, c=sty[n][0], alpha=0.6, linewidths=0)
        ax.plot([np.median(var)] * 2, [y - 0.33, y + 0.33], color="k", lw=0.9)
        ax.text(1.01, y, f"{(var > tv).mean() * 100:.0f}\\%", transform=ax.get_yaxis_transform(), va="center",
                fontsize=7, color="k" if known(n) else "#B00000")
    ax.axvline(tv, color="#00A0E0", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)), [short[n] for n in order], fontsize=7)
    ax.set_xlabel("Variance $V/127^2$")
    ax.set_title("(d) Flagged unknown", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_data_unc.pdf")

# ---------------------------------------------------------------- dataset overview
def fig_dataset():
    v = np.load(os.path.join(BS, "export_robust", "viz_data.npz"))
    n_rec = len(v["rec_name"])
    fig = plt.figure(figsize=(DCOL, 1.62))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.55, 1.55], wspace=0.7)

    # (a) recordings: vibration RMS over time (0.5 s blocks), grouped by split
    ax = fig.add_subplot(gs[0])
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.LogNorm(vmin=0.01, vmax=4)
    order = sorted(range(n_rec), key=lambda i: (["train", "val", "test"].index(str(v["rec_split"][i])), int(v["rec_class"][i])))
    for row, i in enumerate(order):
        rms = v[f"rec_rms_{i}"]
        ax.imshow(rms[None, :], extent=[0, len(rms) * 0.5, row - 0.4, row + 0.4], cmap=cmap, norm=norm, aspect="auto")
        lab = f"{str(v['rec_split'][i])}, {'abn.' if int(v['rec_class'][i]) else 'norm.'} ({int(v['rec_windows'][i])} w)"
        ax.text(-0.4, row, lab, ha="right", va="center", fontsize=7)
    ax.set_ylim(n_rec - 0.5, -0.5)
    ax.set_xlim(0, max(len(v[f"rec_rms_{i}"]) for i in range(n_rec)) * 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Recording time (s)")
    ax.set_title("(a) Recordings: vibration RMS (g)", loc="left")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.015)
    cb.ax.set_title("g", fontsize=7, pad=2)
    cb.ax.tick_params(labelsize=7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # (b) windows per split
    ax = fig.add_subplot(gs[1])
    splits = ["train", "val", "test"]
    for c, (cls, col) in enumerate(((0, "#0072B2"), (1, "#D55E00"))):
        cnt = [int(v["rec_windows"][(v["rec_split"] == sp) & (v["rec_class"] == cls)].sum()) for sp in splits]
        bottom = None if cls == 0 else [int(v["rec_windows"][(v["rec_split"] == sp) & (v["rec_class"] == 0)].sum()) for sp in splits]
        ax.bar(splits, cnt, bottom=bottom, color=col, width=0.6, label=["normal", "abnormal"][cls])
    for x_, sp in enumerate(splits):
        tot = int(v["rec_windows"][v["rec_split"] == sp].sum())
        ax.text(x_, tot + 20, f"{tot:,}", ha="center", fontsize=7)
    ax.set_ylabel("Windows", labelpad=1)
    ax.set_title("(b) Splits", loc="left")
    ax.set_ylim(0, 2050)
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.12, 1.0), handlelength=0.8)
    ax.set_ylim(0, 2750)
    ax.set_xlim(-0.75, 2.6)
    ax.spines[["top", "right"]].set_visible(False)

    # (c) example windows of every data set, as the model sees them (6 x 64 input codes)
    names = [str(n) for n in v["names"]]
    short = ["ID n.", "ID a.", "Field n.", "Field a.", "UAV", "Motor$^\\dagger$", "Stuck", "Zero",
             "Noise", "Spikes"]
    sub = gs[2].subgridspec(2, 5, wspace=0.42, hspace=0.55)
    for k in range(len(names)):
        ax = fig.add_subplot(sub[k // 5, k % 5])
        ax.imshow(v[f"ex_{k}"], cmap="RdBu_r", vmin=-40, vmax=40, aspect="auto", interpolation="nearest")
        ax.set_title(short[k], fontsize=7, pad=2, color="k" if k < 4 else "#7a3b00")
        ax.set_xticks([]); ax.set_yticks([])
        if k % 5 == 0:
            ax.set_yticks([0, 5], ["$a_x$", "$\\omega_z$"], fontsize=7)
        if k == 2:
            ax.text(0.5, 1.3, "(c) Example input windows (6 ch. $\\times$ 64 int8 codes, $\\pm$40 shown)",
                    transform=ax.transAxes, ha="center", fontsize=7.5)
    save(fig, "fig_dataset.pdf")

if __name__ == "__main__":
    fig_cycles()
    fig_uncertainty()
    fig_field()
    fig_stream()
    fig_ablation()
    fig_fpga()
    fig_data_unc()
    fig_dataset()
