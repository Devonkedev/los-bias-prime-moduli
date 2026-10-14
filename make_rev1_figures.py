"""Figures 1 and 2 of the revised manuscript (uncertainty shown).

Output: figures/rev1/Fig1.{pdf,eps,png}, Fig2.{pdf,eps,png}. PNG at 600 dpi is
what gets embedded in the Word file; PDF/EPS are the vector deliverables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import theory  # noqa: E402

OUT = ROOT / "figures" / "rev1"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#555555"


def save(fig, name):
    for ext in ("pdf", "eps", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig1(q=3):
    df = pd.read_csv(ROOT / "data" / "main_run.csv")
    inc = pd.read_csv(ROOT / "revision" / "increment_fit.csv")
    th = pd.read_csv(ROOT / "data" / "theory_C_HL.csv")
    phi2 = (q - 1) ** 2
    sub = df[df.q == q].copy()
    sub["bias"] = sub.count_ab / (sub.pi_x / phi2) - 1
    sub["sigma"] = np.sqrt(phi2 / sub.pi_x)
    pairs = sorted(set(zip(sub.a, sub.b)))
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6), sharex=True)
    for ax, (a, b), letter in zip(axes.ravel(), pairs, "abcd"):
        c = sub[(sub.a == a) & (sub.b == b)].sort_values("x")
        t = 1 / np.log(c.x.to_numpy(float))
        fit_rng = c.x.to_numpy(float) >= 1e8
        ax.errorbar(t[fit_rng], c.bias[fit_rng], yerr=2 * c.sigma[fit_rng], fmt="o",
                    ms=3, color="black", mfc="black", elinewidth=0.6, capsize=1.5,
                    label="Measured (fitted range)")
        ax.errorbar(t[~fit_rng], c.bias[~fit_rng], yerr=2 * c.sigma[~fit_rng], fmt="o",
                    ms=3, color="black", mfc="white", elinewidth=0.6, capsize=1.5,
                    label="Measured (not fitted)")
        c1 = theory.c1(q, a, b)
        c2m = float(inc[(inc.q == q) & (inc.a == a) & (inc.b == b)].c2_inc.iloc[0])
        c2p = float(th[(th.q == q) & (th.a == a) & (th.b == b)].c2_HL.iloc[0])
        tt = np.linspace(t.min(), t.max(), 200)
        ax.plot(tt, c1 * tt * np.log(1 / tt) + c2m * tt, "-", color=BLUE,
                label="Two-term fit (measured $c_2$)")
        ax.plot(tt, c1 * tt * np.log(1 / tt) + c2p * tt, "--", color=ORANGE,
                label="Prediction (predicted $c_2$)")
        ax.axhline(0, color=GREY, lw=0.4, ls=":")
        ax.text(0.03, 0.92, f"({letter})", transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")
        ax.set_ylabel("Normalised bias $B$")
    for ax in axes[1]:
        ax.set_xlabel("1 / log $x$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "Fig1")


def fig2():
    inc = pd.read_csv(ROOT / "revision" / "increment_fit.csv")
    th = pd.read_csv(ROOT / "data" / "theory_C_HL.csv")[["q", "a", "b", "c2_HL"]]
    m = inc.merge(th, on=["q", "a", "b"])
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.6))
    for ax, (q, g), letter in zip(axes.ravel(), m.groupby("q"), "abcdef"):
        off = g[g.a != g.b]
        dia = g[g.a == g.b]
        ax.errorbar(off.c2_HL, off.c2_inc, yerr=2 * off.se_inc_scaled, fmt="o", ms=2.6,
                    color=BLUE, mfc=BLUE, elinewidth=0.5, capsize=0, alpha=0.85,
                    label="Off-diagonal pairs")
        ax.errorbar(dia.c2_HL, dia.c2_inc, yerr=2 * dia.se_inc_scaled, fmt="^", ms=3.6,
                    color=ORANGE, mfc="white", elinewidth=0.5, capsize=0,
                    label="Diagonal pairs")
        lo = min(g.c2_HL.min(), g.c2_inc.min())
        hi = max(g.c2_HL.max(), g.c2_inc.max())
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.6, label="Measured = predicted")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"({letter}) q = {q}", loc="left", fontsize=9, fontweight="bold", pad=3)
        ax.tick_params(length=2.5)
    for ax in axes[1]:
        ax.set_xlabel("Predicted $c_2$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Measured $c_2$")
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "Fig2")


if __name__ == "__main__":
    fig1()
    fig2()
    print("written to", OUT)
