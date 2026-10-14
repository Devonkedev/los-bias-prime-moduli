"""Fit the LOS bias expansion to measured B(x; q, a, b).

The LOS Main Conjecture (see theory.py) gives

    B(x; q, a, b) = c1(q,a,b) * loglog(x)/log(x) + c2(q,a,b)/log(x) + O((log x)^-7/4)

so the bias has TWO leading terms, not one. This module provides:

fit_two_term   -- PRIMARY. Fixes c1 at its LOS-predicted value (which is exactly
                  determined by whether a == b) and fits c2. Because c1 is known,
                  this is a one-parameter fit: subtract the c1 term and regress
                  the remainder on 1/log x.

fit_wls        -- The naive one-term B ~ C/log x model. Retained as a comparator
                  and for the drift diagnostic below, NOT as a primary result.

fit_window_drift -- Refits the naive model over sliding x-windows. Because a
                  one-term model has to absorb the slowly-varying loglog x into
                  a constant, its fitted C drifts systematically with the window.
                  Quantifying that drift is how we show the one-term model is
                  misspecified rather than merely imprecise.

fit_bootstrap  -- Block bootstrap over independent stretches of the prime stream.
                  The WLS errors are optimistic: checkpoints are NESTED (the
                  count at 1e10 contains every pair counted at 1e9), so residuals
                  across checkpoints are strongly positively correlated and the
                  usual independent-errors formula understates the variance.
                  Resampling disjoint prime-stream blocks respects that
                  structure. Requires the per-block CSV from counters.run_counters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    q: int
    a: int
    b: int
    C: float
    C_err: float
    D: Optional[float]
    D_err: Optional[float]
    chi2_dof: float
    n_checkpoints: int


def fit_wls(df: pd.DataFrame, include_next_order: bool = False,
            x_min_for_fit: float = 1e8) -> pd.DataFrame:
    """Naive WLS fit of B ~ C/log x (optionally + D/log^2 x).

    Kept as a comparator to the two-term fit. The fitted C is NOT comparable to
    any LOS constant: with a genuine c1*loglog(x)/log(x) term present, a
    one-term model absorbs it and the fitted value depends on the fit window.
    See fit_window_drift.
    """
    results = []
    for (q, a, b), sub in df.groupby(["q", "a", "b"]):
        sub = sub[sub["x"] >= x_min_for_fit].sort_values("x")
        if len(sub) < (3 if include_next_order else 2):
            continue

        log_x = np.log(sub["x"].to_numpy(dtype=float))
        y = sub["bias"].to_numpy(dtype=float)
        sigma = sub["sigma_bias"].to_numpy(dtype=float)
        w = 1.0 / np.where(sigma > 0, sigma**2, np.inf)

        X = (np.column_stack([1.0 / log_x, 1.0 / log_x**2])
             if include_next_order else (1.0 / log_x).reshape(-1, 1))

        WX = X * w[:, None]
        try:
            coef = np.linalg.solve(X.T @ WX, X.T @ (w * y))
            cov = np.linalg.inv(X.T @ WX)
        except np.linalg.LinAlgError:
            continue

        residuals = y - X @ coef
        dof = max(len(y) - X.shape[1], 1)
        chi2_dof = float(np.sum(w * residuals**2) / dof)

        results.append(FitResult(
            q=int(q), a=int(a), b=int(b),
            C=float(coef[0]), C_err=float(np.sqrt(cov[0, 0])),
            D=float(coef[1]) if include_next_order else None,
            D_err=float(np.sqrt(cov[1, 1])) if include_next_order else None,
            chi2_dof=chi2_dof, n_checkpoints=len(y),
        ))

    return pd.DataFrame([r.__dict__ for r in results])


def fit_two_term(df: pd.DataFrame, x_min_for_fit: float = 1e8,
                 fix_c1: bool = True) -> pd.DataFrame:
    """PRIMARY FIT. B ~ c1*loglog(x)/log(x) + c2/log(x), with c1 fixed.

    With c1 fixed at its predicted value the model is linear in the single
    unknown c2:

        (B - c1 * loglog x / log x) = c2 * (1 / log x)

    Why fix c1: over any computationally reachable range the two basis
    functions loglog(x)/log(x) and 1/log(x) are nearly collinear -- measured
    correlation 0.999 over x in [1e6, 1e11]. Fitting both free inflates the
    error on each by more than an order of magnitude and yields no usable
    constraint. Since LOS determine c1 exactly (it depends only on whether
    a == b), fixing it is both defensible and necessary.

    Set fix_c1=False to fit both coefficients anyway; the returned c1_err then
    shows directly how badly conditioned that is.

    Returns columns: q, a, b, c1_fixed, c2, c2_err, chi2_dof, n_checkpoints
    (plus c1_fitted / c1_err when fix_c1=False).
    """
    import theory

    rows = []
    for (q, a, b), sub in df.groupby(["q", "a", "b"]):
        sub = sub[sub["x"] >= x_min_for_fit].sort_values("x")
        if len(sub) < 2:
            continue

        x = sub["x"].to_numpy(dtype=float)
        log_x = np.log(x)
        u = np.log(log_x) / log_x          # loglog x / log x
        v = 1.0 / log_x                    # 1 / log x
        y = sub["bias"].to_numpy(dtype=float)
        sigma = sub["sigma_bias"].to_numpy(dtype=float)
        w = 1.0 / np.where(sigma > 0, sigma**2, np.inf)

        c1_pred = theory.c1(int(q), int(a), int(b))

        if fix_c1:
            resid_y = y - c1_pred * u
            X = v.reshape(-1, 1)
        else:
            resid_y = y
            X = np.column_stack([u, v])

        WX = X * w[:, None]
        try:
            coef = np.linalg.solve(X.T @ WX, X.T @ (w * resid_y))
            cov = np.linalg.inv(X.T @ WX)
        except np.linalg.LinAlgError:
            continue

        fitted = X @ coef
        dof = max(len(resid_y) - X.shape[1], 1)
        chi2_dof = float(np.sum(w * (resid_y - fitted) ** 2) / dof)

        row = {
            "q": int(q), "a": int(a), "b": int(b),
            "c1_fixed": c1_pred,
            "chi2_dof": chi2_dof,
            "n_checkpoints": int(len(resid_y)),
        }
        if fix_c1:
            row["c2"] = float(coef[0])
            row["c2_err"] = float(np.sqrt(cov[0, 0]))
        else:
            row["c1_fitted"] = float(coef[0])
            row["c1_err"] = float(np.sqrt(cov[0, 0]))
            row["c2"] = float(coef[1])
            row["c2_err"] = float(np.sqrt(cov[1, 1]))
        rows.append(row)

    return pd.DataFrame(rows)


def fit_window_drift(df: pd.DataFrame, n_windows: int = 4,
                     x_min_for_fit: float = 1e8) -> pd.DataFrame:
    """Refit both models over sliding x-windows to expose misspecification.

    A correctly specified model gives a window-independent constant. The
    one-term C/log x model cannot: it has to absorb c1*loglog x, which varies
    across the window. The two-term fit with c1 fixed should be markedly more
    stable. Reported in the paper as the evidence that the one-term framing is
    wrong, not merely noisy.

    Returns one row per (q, a, b, window) with the fitted C and c2.
    """
    xs = np.sort(df.loc[df["x"] >= x_min_for_fit, "x"].unique())
    if len(xs) < 2 * n_windows:
        n_windows = max(2, len(xs) // 3)

    edges = np.array_split(xs, n_windows)
    out = []
    for w_idx, chunk in enumerate(edges):
        if len(chunk) < 2:
            continue
        lo, hi = float(chunk[0]), float(chunk[-1])
        sub = df[(df["x"] >= lo) & (df["x"] <= hi)]
        naive = fit_wls(sub, x_min_for_fit=lo)
        two = fit_two_term(sub, x_min_for_fit=lo)
        if naive.empty or two.empty:
            continue
        merged = naive[["q", "a", "b", "C", "C_err"]].merge(
            two[["q", "a", "b", "c2", "c2_err"]], on=["q", "a", "b"])
        merged["window"] = w_idx
        merged["x_lo"] = lo
        merged["x_hi"] = hi
        out.append(merged)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize_drift(drift: pd.DataFrame) -> pd.DataFrame:
    """Per-cell spread of each model's constant across windows.

    Ratio > 1 means the one-term C moves around more than the two-term c2,
    i.e. the one-term model is absorbing something window-dependent.
    """
    if drift.empty:
        return pd.DataFrame()
    g = drift.groupby(["q", "a", "b"]).agg(
        C_spread=("C", lambda s: float(s.max() - s.min())),
        c2_spread=("c2", lambda s: float(s.max() - s.min())),
        C_mean=("C", "mean"),
        c2_mean=("c2", "mean"),
    ).reset_index()
    g["drift_ratio"] = g["C_spread"] / g["c2_spread"].replace(0, np.nan)
    return g


def fit_bootstrap(df: pd.DataFrame, blocks_csv: str | Path = "data/blocks.csv",
                  n_resamples: int = 400, x_min_for_fit: float = 1e8,
                  random_seed: int = 0) -> pd.DataFrame:
    """Block bootstrap over prime-stream blocks; honest error bars for c2.

    Method. counters.run_counters emits per-block pair counts over disjoint
    stretches of the prime stream. Those blocks are (to a good approximation)
    independent, whereas checkpoints are nested and hence strongly correlated.
    Each resample draws len(blocks) blocks with replacement, forms cumulative
    counts in the original block order, rebuilds the bias curve, and refits c2.
    The reported error is the standard deviation of c2 across resamples.

    This is the honest counterpart to the WLS error bars, which are computed as
    if checkpoints were independent draws. Expect the bootstrap error to be
    LARGER; the ratio is reported in RESULTS.md and the paper.

    Returns columns: q, a, b, c2_boot, c2_boot_err, c2_boot_lo, c2_boot_hi.
    Raises FileNotFoundError if the per-block CSV is absent.
    """
    import theory

    blocks_csv = Path(blocks_csv)
    if not blocks_csv.exists():
        raise FileNotFoundError(
            f"{blocks_csv} not found. Re-run counters.run_counters with "
            "blocks_csv=... to emit per-block counts for the bootstrap.")

    blocks = pd.read_csv(blocks_csv)
    rng = np.random.default_rng(random_seed)

    # Per-block prime counts (one value per block, not per cell).
    per_block_primes = (blocks.groupby("block_index")["n_primes"].first()
                        .sort_index())
    block_ids = per_block_primes.index.to_numpy()
    n_blocks = len(block_ids)
    if n_blocks < 4:
        raise ValueError(
            f"only {n_blocks} blocks available; need >= 4 for a bootstrap. "
            "Reduce block_size in run_counters.")

    # Pivot to (block, cell) count matrix per modulus cell.
    cells = blocks.groupby(["q", "a", "b"])
    results = []

    # x at the end of each block, used to rebuild the bias curve.
    block_hi = (blocks.groupby("block_index")["x_hi"].first().sort_index()
                .to_numpy(dtype=float))
    primes_arr = per_block_primes.to_numpy(dtype=float)

    for (q, a, b), cell in cells:
        counts = (cell.set_index("block_index")["count_ab"]
                  .reindex(block_ids, fill_value=0).to_numpy(dtype=float))
        phi = q - 1  # prime q
        c1_pred = theory.c1(int(q), int(a), int(b))

        draws = []
        for _ in range(n_resamples):
            pick = rng.integers(0, n_blocks, size=n_blocks)
            cum_counts = np.cumsum(counts[pick])
            cum_primes = np.cumsum(primes_arr[pick])
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = cum_primes / phi**2
                bias = np.where(expected > 0, cum_counts / expected - 1.0, np.nan)

            log_x = np.log(block_hi)
            keep = (block_hi >= x_min_for_fit) & np.isfinite(bias)
            if keep.sum() < 2:
                continue
            u = np.log(log_x[keep]) / log_x[keep]
            v = 1.0 / log_x[keep]
            resid = bias[keep] - c1_pred * u
            denom = float(v @ v)
            if denom <= 0:
                continue
            draws.append(float((v @ resid) / denom))

        if len(draws) < max(10, n_resamples // 10):
            continue
        arr = np.asarray(draws)
        results.append({
            "q": int(q), "a": int(a), "b": int(b),
            "c2_boot": float(arr.mean()),
            "c2_boot_err": float(arr.std(ddof=1)),
            "c2_boot_lo": float(np.percentile(arr, 2.5)),
            "c2_boot_hi": float(np.percentile(arr, 97.5)),
            "n_resamples": int(len(arr)),
        })

    return pd.DataFrame(results)


def compare_to_theory(fit_df: pd.DataFrame, theory_df: pd.DataFrame) -> pd.DataFrame:
    """Merge empirical fits with the predicted constants; residuals and z-scores.

    Accepts either the two-term output (column `c2`) or the naive output
    (column `C`), and matches against `c2_HL` / `C_HL` in theory_df.
    """
    theory_df = theory_df.copy()
    if "c2_HL" not in theory_df.columns and "C_HL" in theory_df.columns:
        theory_df["c2_HL"] = theory_df["C_HL"]

    value_col = "c2" if "c2" in fit_df.columns else "C"
    err_col = "c2_err" if "c2_err" in fit_df.columns else "C_err"

    keep = ["q", "a", "b", "c2_HL"] + (
        ["c1_HL"] if "c1_HL" in theory_df.columns else [])
    out = fit_df.merge(theory_df[keep], on=["q", "a", "b"], how="left")
    out["residual"] = out[value_col] - out["c2_HL"]
    out["z_score"] = out["residual"] / out[err_col].replace(0, np.nan)
    return out.rename(columns={value_col: "c2_emp", err_col: "c2_emp_err",
                               "C": "C_emp", "C_err": "C_emp_err"})


if __name__ == "__main__":
    from counters import compute_bias_from_checkpoints

    df = compute_bias_from_checkpoints("data/_smoke.csv")
    print("two-term (c1 fixed):")
    print(fit_two_term(df, x_min_for_fit=1e4).head(10))
