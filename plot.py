"""Figure generation. Six functions, one per planned figure.

Fit frames may carry either the two-term constant (`c2`, from fit.fit_two_term)
or the naive one-term constant (`C`, from fit.fit_wls). `_constant_columns`
resolves whichever is present so the figures work with both.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("figures")


def _constant_columns(fits: pd.DataFrame) -> tuple[str, str, str]:
    """Return (value_col, err_col, label) for whichever fit produced `fits`."""
    if "c2" in fits.columns:
        return "c2", "c2_err", r"$c_2$"
    if "c2_emp" in fits.columns:
        return "c2_emp", "c2_emp_err", r"$c_2$"
    return "C", "C_err", "C"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 9,
    })


def fig01_bias_vs_logx(df: pd.DataFrame, fits: pd.DataFrame, q: int = 3,
                       save: bool = True) -> None:
    """B(x) vs 1/log(x) at fixed q, panel per (a, b), fit overlay."""
    FIG_DIR.mkdir(exist_ok=True)
    sub = df[df["q"] == q]
    pairs = sorted(set(zip(sub["a"], sub["b"])))
    n = len(pairs)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    value_col, _, label = _constant_columns(fits)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    for ax, (a, b) in zip(axes, pairs):
        cell = sub[(sub["a"] == a) & (sub["b"] == b)].sort_values("x")
        x_vals = cell["x"].to_numpy(dtype=float)
        inv_log_x = 1.0 / np.log(x_vals)
        ax.errorbar(inv_log_x, cell["bias"], yerr=cell["sigma_bias"],
                    fmt="o", markersize=3, capsize=2, label="empirical")

        row = fits[(fits["q"] == q) & (fits["a"] == a) & (fits["b"] == b)]
        if len(row) == 1:
            const = float(row[value_col].iloc[0])
            xs = np.linspace(inv_log_x.min(), inv_log_x.max(), 100)
            if value_col.startswith("c2") and "c1_fixed" in row.columns:
                # Two-term model: c1*loglog(x)/log(x) + c2/log(x). Parameterised
                # by t = 1/log x, the first term is c1 * t * log(1/t).
                c1 = float(row["c1_fixed"].iloc[0])
                curve = c1 * xs * np.log(1.0 / xs) + const * xs
                ax.plot(xs, curve, "-",
                        label=f"$c_1$={c1:.2f}, $c_2$={const:.2f}")
            else:
                ax.plot(xs, const * xs, "-", label=f"{label}={const:.3f}")

        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(f"({a}, {b}) mod {q}")
        ax.legend(loc="best", fontsize=7)

    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    fig.suptitle(f"B(x; q={q}, a, b) vs 1/log(x)")
    fig.supxlabel("1 / log(x)")
    fig.supylabel("B(x; q, a, b)")
    fig.tight_layout()
    if save:
        path = FIG_DIR / f"fig01_bias_vs_logx_q{q}.png"
        fig.savefig(path); print(f"Saved {path}")
    plt.close(fig)


def fig02_C_emp_vs_C_HL(comparison: pd.DataFrame, save: bool = True) -> None:
    """Scatter empirical c2 vs Hardy-Littlewood c2 across all (q, a, b)."""
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    value_col, err_col, _ = _constant_columns(comparison)

    for q, sub in comparison.groupby("q"):
        ax.errorbar(sub["c2_HL"], sub[value_col], yerr=sub[err_col],
                    fmt="o", markersize=4, capsize=2, label=f"q = {q}")

    lo = min(comparison["c2_HL"].min(), comparison[value_col].min())
    hi = max(comparison["c2_HL"].max(), comparison[value_col].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")

    ax.set_xlabel(r"$c_2$ (Hardy--Littlewood, LOS 2016)")
    ax.set_ylabel(r"$c_2$ (empirical)")
    ax.set_title("Empirical vs predicted second-order constants"); ax.legend()
    fig.tight_layout()
    if save:
        path = FIG_DIR / "fig02_C_emp_vs_C_HL.png"
        fig.savefig(path); print(f"Saved {path}")
    plt.close(fig)


def fig03_residuals(comparison: pd.DataFrame, save: bool = True) -> None:
    """Residual bars C_emp − C_HL by (q, a, b), colored by pair-type."""
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    comparison = comparison.copy()
    comparison["pair_type"] = np.where(
        comparison["a"] == comparison["b"], "diagonal (a=b)", "off-diagonal"
    )
    comparison["label"] = comparison.apply(
        lambda r: f"q={r['q']}\n({r['a']},{r['b']})", axis=1
    )
    comparison = comparison.sort_values(["q", "a", "b"]).reset_index(drop=True)

    _, err_col, _ = _constant_columns(comparison)

    colors = ["tab:red" if pt == "diagonal (a=b)" else "tab:blue"
              for pt in comparison["pair_type"]]
    ax.bar(range(len(comparison)), comparison["residual"], color=colors,
           yerr=comparison[err_col], capsize=2)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

    ax.set_xticks(range(len(comparison)))
    ax.set_xticklabels(comparison["label"], rotation=90, fontsize=6)
    ax.set_ylabel(r"$c_2^{\mathrm{emp}} - c_2^{\mathrm{HL}}$")
    ax.set_title("Residuals by (q, a, b)")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="tab:red", label="diagonal (a=b)"),
        Patch(color="tab:blue", label="off-diagonal"),
    ])
    fig.tight_layout()
    if save:
        path = FIG_DIR / "fig03_residuals.png"
        fig.savefig(path); print(f"Saved {path}")
    plt.close(fig)


def fig04_pair_type_split(fits: pd.DataFrame, save: bool = True) -> None:
    """C per (q, a, b), jittered by (b − a) mod q to expose substructure."""
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    value_col, err_col, label = _constant_columns(fits)

    for q, sub in fits.groupby("q"):
        sub = sub.copy()
        sub["diff"] = (sub["b"] - sub["a"]) % q
        ax.errorbar(sub["q"] + 0.02 * sub["diff"], sub[value_col],
                    yerr=sub[err_col],
                    fmt="o", markersize=4, capsize=2, label=f"q = {q}")

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel(r"$q$ (jittered by $(b-a)$ mod $q$)")
    ax.set_ylabel(f"{label}$(q, a, b)$")
    ax.set_title(f"Substructure in {label}$(q, a, b)$")
    ax.legend(); fig.tight_layout()
    if save:
        path = FIG_DIR / "fig04_pair_type_split.png"
        fig.savefig(path); print(f"Saved {path}")
    plt.close(fig)


def fig05_C_of_q(fits: pd.DataFrame, save: bool = True) -> None:
    """Mean C per q, split diagonal vs off-diagonal."""
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    fits = fits.copy()
    value_col, _, label = _constant_columns(fits)
    fits["pair_type"] = np.where(fits["a"] == fits["b"], "diagonal", "off-diagonal")

    for pt, sub in fits.groupby("pair_type"):
        agg = sub.groupby("q")[value_col].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["q"], agg["mean"], yerr=agg["std"], fmt="o-",
                    label=pt, capsize=3)

    ax.set_xlabel(r"$q$")
    ax.set_ylabel(f"Mean {label} (error bar = stdev across pairs)")
    ax.set_title(f"{label}$(q)$ across moduli")
    ax.legend(); fig.tight_layout()
    if save:
        path = FIG_DIR / "fig05_C_of_q.png"
        fig.savefig(path); print(f"Saved {path}")
    plt.close(fig)


def fig06_next_order(df: pd.DataFrame, fits: pd.DataFrame, q: int = 3,
                     save: bool = True) -> None:
    """B(x) − C/log(x) vs 1/log(x). Structure here = next-order story."""
    FIG_DIR.mkdir(exist_ok=True)
    sub = df[df["q"] == q]
    fits_q = fits[fits["q"] == q].set_index(["a", "b"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for (a, b), cell in sub.groupby(["a", "b"]):
        if (a, b) not in fits_q.index:
            continue
        C = float(fits_q.loc[(a, b), "C"])
        log_x = np.log(cell["x"].to_numpy(dtype=float))
        residual = cell["bias"].to_numpy() - C / log_x
        ax.plot(1.0 / log_x, residual, "o-", markersize=3, label=f"({a}, {b})")

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("1 / log(x)")
    ax.set_ylabel("B(x) − C/log(x)")
    ax.set_title(f"Next-order residuals at q = {q}")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    if save:
        path = FIG_DIR / f"fig06_next_order_q{q}.png"
        fig.savefig(path); print(f"Saved {path}")
    plt.close(fig)
