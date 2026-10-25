#!/usr/bin/env python3
"""
PlotNeuralNet drawing of the deployed Binary-SNGP (bsngp/export_robust/params.npz).
Layer sizes are read from the exported integer model, so the figure matches the hardware.

  python plotneuralnet/bsngp_architecture.py      ->  plotneuralnet/outputs/bsngp_architecture.{tex,pdf}
                                                      and paper/figures/fig_network.pdf
"""

import os
import shutil
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARAMS = os.path.join(ROOT, "bsngp", "export_robust", "params.npz")
OUT = os.path.join(HERE, "outputs")
NAME = "bsngp_architecture"

p = np.load(PARAMS)
C0, C1, C2, C3 = 6, len(p["th1"]), len(p["th2"]), len(p["th3"])
D, R = len(p["beta"]), p["u"].shape[0]
DC = int(p["dc_remove"]) if "dc_remove" in p.files else 0
K = 1 << int(p["ksh"])
L0, L1, L2, L3 = 64, 64, 32, 16
wbits = int(p["w1"].size + p["w2"].size + p["w3"].size)

# ---------------------------------------------------------------- embedded data plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIZ = os.path.join(ROOT, "bsngp", "export_robust", "viz_data.npz")
IMG = os.path.join(OUT, "img")
os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{amsmath}\usepackage{amssymb}", "font.family": "serif", "font.size": 15, "xtick.labelsize": 14, "ytick.labelsize": 14, "pdf.fonttype": 42})
v = np.load(VIZ)
vnames = [str(n) for n in v["names"]]
TV = float(v["tv"]) / 127.0 ** 2


def prob(k):
    var = np.maximum(v[f"V_{k}"], 0) / 127.0 ** 2
    return 1 / (1 + np.exp(-(v[f"L_{k}"] / float(v["logit_scale"])) / np.sqrt(1 + np.pi / 8 * var)))


# input: one real normal window (6 channels, physical units, stacked)
f, ax = plt.subplots(figsize=(2.2, 2.6))
x = v["ex_normal"]
for i, (lab, sc) in enumerate(zip(["$a_x$", "$a_y$", "$a_z$", "$\\omega_x$", "$\\omega_y$", "$\\omega_z$"],
                                  [1, 1, 1, 30, 30, 30])):
    sig = (x[i] - x[i].mean()) / sc
    ax.plot(np.arange(64) * 2, sig - 2.2 * i, lw=1.0, color=["#0072B2", "#009E73", "#D55E00"][i % 3])
    ax.text(-4, -2.2 * i, lab, ha="right", va="center", fontsize=14)
ax.set_xticks([0, 64, 128], ["0", "64", "128 ms"])
ax.set_yticks([])
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
f.savefig(os.path.join(IMG, "input_window.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(f)

# mean head: distribution of the SNGP probability, normal vs abnormal (held-out recordings)
f, ax = plt.subplots(figsize=(2.9, 1.7))
bins = np.linspace(0, 1, 21)
for k, c, lab in ((vnames.index("ID normal"), "#0072B2", "normal"), (vnames.index("ID abnormal"), "#D55E00", "abnormal")):
    ax.hist(prob(k), bins=bins, color=c, alpha=0.8, label=lab, weights=np.full(len(v[f"L_{k}"]), 100 / len(v[f"L_{k}"])))
ax.set_xlabel("$p$(abnormal)")
ax.set_ylabel(r"\% windows")
ax.legend(frameon=False, fontsize=14, loc="upper center")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
f.savefig(os.path.join(IMG, "head_mean.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(f)

# variance head: uncertainty of known vs unknown inputs, with the hardware threshold
groups = [("Known (ID + field)", ["ID normal", "ID abnormal", "Field-mount normal", "Field-mount abnormal"], "#0072B2"),
          ("Other machines", ["UAV (other machine)", "Motor rig (acc. only)"], "#009E73"),
          ("Sensor faults", ["Sensor stuck", "Sensor zero", "Noise burst", "Spikes"], "#D55E00")]
f, ax = plt.subplots(figsize=(2.6, 1.6))
for yv, (lab, members, c) in enumerate(groups[::-1]):
    var = np.concatenate([np.maximum(v[f"V_{vnames.index(m)}"], 1) for m in members]) / 127.0 ** 2
    jit = (np.random.default_rng(yv).random(len(var)) - 0.5) * 0.5
    ax.scatter(var, yv + jit, s=2, color=c, alpha=0.5, linewidths=0)
    ax.text(1.02, yv, f"{(var > TV).mean() * 100:.0f}\\%", transform=ax.get_yaxis_transform(), va="center", fontsize=14)
ax.axvline(TV, color="k", ls="--", lw=1)
ax.text(TV * 1.1, 2.45, "$T_V$", fontsize=14)
ax.set_xscale("log")
ax.set_yticks(range(3), [g[0] for g in groups[::-1]], fontsize=14)
ax.set_xlabel("variance $V/127^2$ (\\% flagged)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
f.savefig(os.path.join(IMG, "head_var.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(f)

hgt = {64: 22, 32: 15, 16: 10}           # box height ~ sequence length
wid = lambda c: {6: 0.8, 16: 1.6, 32: 2.6}.get(c, 1.0)  # box width ~ channels
DEP = 10


def label(name, caption, where="south", dy=-0.9, width=3.0):
    anchor = "north" if where == "south" else "south"
    if name == "var":  # right-aligned to the box so it clears the RFF label
        anchor, width = "north west", 6.8
        return rf"""
\node[anchor={anchor}, text width={width}cm, align=left, font=\large] at ([xshift=-0.3cm, yshift={dy}cm]{name}-{where}) {{{caption}}};"""
    return rf"""
\node[anchor={anchor}, text width={width}cm, align=center, font=\large] at ([yshift={dy}cm]{name}-{where}) {{{caption}}};"""


def box(name, at, shift, caption, xlabel, ylabel, fill, w, h, d=DEP, opacity=0.85, where="south"):
    return rf"""
\pic[shift={{{shift}}}] at {at} {{Box={{name={name}, caption={{}}, xlabel={{{{{xlabel}, }}}},
    ylabel={{{ylabel}}}, zlabel={{}}, fill={fill}, opacity={opacity}, height={h}, width={w}, depth={d}}}}};""" + \
        label(name, caption, where, -0.9 if where == "south" else 0.4, 6.8 if name in ("mean", "var") else 3.0)


def banded(name, at, shift, caption, xlabel, ylabel, w, h):
    return rf"""
\pic[shift={{{shift}}}] at {at} {{RightBandedBox={{name={name}, caption={{}},
    xlabel={{{{{xlabel}, }}}}, ylabel={{{ylabel}}}, zlabel={{}}, fill=\BinColor, bandfill=\SignColor,
    opacity=0.85, height={h}, width={{{w}, 0.45}}, depth={DEP}}}}};""" + label(name, caption)


def arrow(a, b):
    return rf"""
\draw [connection] ({a}-east) -- node {{\midarrow}} ({b}-west);"""


body = [r"""
\node[anchor=east, inner sep=0] (rawin) at (-1.6,0) {\includegraphics[height=4.6cm]{img/input_window.pdf}};
\node[anchor=south, font=\large\bfseries] at (rawin.north) {real IMU window};
\draw [connection] (rawin.east) -- node {\midarrow} (-0.05,0);"""]
body.append(box("inp", "(0,0,0)", "(0,0,0)", rf"Input $\mathbf{{x}}$, ${C0}\times{L0}$\\int8, 500 Hz", C0, "", r"\InputColor", wid(C0), hgt[L0]))
prev = "inp"
if DC:
    body.append(box("dc", "(inp-east)", "(2.8,0,0)", r"DC removal\\$x-\lfloor\bar{x}\rfloor$", C0, "",
                    r"\DCColor", 0.35, hgt[L0]))
    body.append(arrow("inp", "dc"))
    prev = "dc"
body.append(banded("c1", f"({prev}-east)", "(2.8,0,0)", rf"BinConv1, ${C1}\times{L1}$\\$\pm$1, add/sub\\k3 s1", C1, "", wid(C1), hgt[L1]))
body.append(arrow(prev, "c1"))
body.append(banded("c2", "(c1-east)", "(2.6,0,0)", rf"BinConv2, ${C2}\times{L2}$\\XNOR-popc.\\k3 s2", C2, "", wid(C2), hgt[L2]))
body.append(arrow("c1", "c2"))
body.append(banded("c3", "(c2-east)", "(2.6,0,0)", rf"BinConv3, ${C3}\times{L3}$\\XNOR-popc.\\k3 s2", C3, "", wid(C3), hgt[L3]))
body.append(arrow("c2", "c3"))
body.append(rf"""
\pic[shift={{(1.6,0,0)}}] at (c3-east) {{Ball={{name=gap, fill=\SumColor, opacity=0.6, radius=1.6, logo=$\Sigma_t$}}}};
\draw [connection] (c3-east) -- node {{\midarrow}} (gap-west);""")
body.append(box("h", "(gap-east)", "(1.4,0,0)", r"$\mathbf{h}\in[-16,16]^{" + str(C3) + "}$", C3, "", r"\VecColor",
                0.35, 12, 3))
body.append(r"""
\draw [connection] (gap-east) -- node {\midarrow} (h-west);""")
body.append(box("rff", "(h-east)", "(2.6,0,0)", rf"RFF, $D$={D}\\$\pm$1 proj., $K$={K}\\cos ROM", D, "",
                r"\RffColor", 0.6, 24, 3))
body.append(arrow("h", "rff"))
body.append(box("mean", "(rff-east)", "(2.6,2.3,0)", r"GP mean\\$L=\boldsymbol{\beta}^{\top}\boldsymbol{\phi}+\beta_0$",
                1, "", r"\MeanColor", 0.6, 4, 3, where="north"))
body.append(box("var", "(rff-east)", "(2.6,-2.3,0)", rf"GP variance, $R$={R}\\$V=\|\boldsymbol{{\phi}}\|^2-\sum_k(\mathbf{{u}}_k^{{\top}}\boldsymbol{{\phi}}\gg7)^2$",
                1, "", r"\VarColor", 0.6, 4, 3))
body.append(r"""
\draw [connection] (rff-east) -- node {\midarrow} (mean-west);
\draw [connection] (rff-east) -- node {\midarrow} (var-west);
\node[draw, rounded corners, fill=white, align=center, font=\large, right=2.2cm of mean-east] (abn) {abnormal\\$L>T_L$};
\node[draw, rounded corners, fill=white, align=center, font=\large, right=2.2cm of var-east] (unk) {unknown\\$V>T_V$};
\draw [connection] (mean-east) -- node {\midarrow} (abn.west);
\draw [connection] (var-east) -- node {\midarrow} (unk.west);
\node[anchor=west, inner sep=0] (pm) at ([xshift=0.5cm]abn.east) {\includegraphics[height=3.6cm]{img/head_mean.pdf}};
\node[anchor=west, inner sep=0] (pv) at ([xshift=0.5cm]unk.east) {\includegraphics[height=3.6cm]{img/head_var.pdf}};
\draw [connection] (abn.east) -- (pm.west);
\draw [connection] (unk.east) -- (pv.west);""")
# braces naming the two parts (below the labels)
body.append(rf"""
\path (inp-south) ++(0,-3.6) coordinate (ybase);
\draw[decorate, decoration={{brace, amplitude=10pt, mirror}}, line width=1pt]
  (c1-west |- ybase) -- (gap-east |- ybase)
  node[midway, below=14pt, align=center, font=\Large] {{\textbf{{1-bit backbone}}\\{wbits:,} weight bits, BN folded into thresholds, 0 multipliers}};
\draw[decorate, decoration={{brace, amplitude=10pt, mirror}}, line width=1pt]
  (h-west |- ybase) -- (pv.east |- ybase)
  node[midway, below=14pt, align=center, font=\Large] {{\textbf{{Integer GP head}}\\{R + 3} multipliers, no sigmoid / sqrt / divide}};
\node[font=\Large\bfseries, anchor=south west] at ([yshift=1.2cm]inp-northwest) {{Binary-SNGP ({'orientation-robust, ' if DC else ''}one decision per 64 ms)}};""")

tex = r"""\documentclass{article}
\usepackage{mathptmx}
\usepackage{tikz}
\usepackage{amsmath,amssymb}
\usepackage{import}
\subimport{""" + os.path.join(HERE, "layers") + r"""/}{init}
\usetikzlibrary{positioning,3d,decorations.pathreplacing}
\pagestyle{empty}
\def\InputColor{rgb:blue,2;green,3;white,4}
\def\DCColor{rgb:magenta,2;white,5}
\def\BinColor{rgb:yellow,5;red,2.5;white,5}
\def\SignColor{rgb:red,4;black,1;white,2}
\def\SumColor{rgb:blue,5;green,15}
\def\VecColor{rgb:green,3;white,4}
\def\RffColor{rgb:orange,5;white,3}
\def\MeanColor{rgb:blue,5;red,2.5;white,5}
\def\VarColor{rgb:red,5;yellow,2;white,3}
\newsavebox\netbox
\begin{document}
\sbox\netbox{%
\begin{tikzpicture}[font=\large]
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
""" + "\n".join(body) + r"""
\end{tikzpicture}}%
\pdfpagewidth=\dimexpr\wd\netbox+24pt\relax
\pdfpageheight=\dimexpr\ht\netbox+\dp\netbox+24pt\relax
\hoffset=-1in \voffset=-1in
\shipout\hbox{\kern12pt\vbox{\kern12pt\hbox{\usebox\netbox}\kern12pt}}
\end{document}
"""

os.makedirs(OUT, exist_ok=True)
tex_path = os.path.join(OUT, NAME + ".tex")
open(tex_path, "w").write(tex)
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", NAME + ".tex"], cwd=OUT,
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:])
    raise SystemExit("pdflatex failed")
pdf = os.path.join(OUT, NAME + ".pdf")
os.makedirs(os.path.join(ROOT, "paper", "figures"), exist_ok=True)
dst = os.path.join(ROOT, "paper", "figures", "fig_network.pdf")
shutil.copy(pdf, dst)
print(f"C={C1}/{C2}/{C3} D={D} R={R} DC={DC} K={K}; wrote {pdf} and {dst}")
