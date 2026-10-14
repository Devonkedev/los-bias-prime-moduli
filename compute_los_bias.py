"""CLI orchestrator: sieve -> counters -> fit -> plot.

The primary fit is the two-term LOS model with c1 fixed (fit.fit_two_term); the
naive one-term C/log x fit is also run, purely as a comparator and to produce
the window-drift diagnostic.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from counters import (
    MODULI, log_spaced_checkpoints, run_counters, compute_bias_from_checkpoints,
)
from fit import (
    compare_to_theory, fit_bootstrap, fit_two_term, fit_window_drift, fit_wls,
    summarize_drift,
)
import plot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--x-max", type=float, default=1e10)
    p.add_argument("--x-min", type=float, default=1e6)
    p.add_argument("--per-decade", type=int, default=4)
    p.add_argument("--segment", type=int, default=2 * 10**7,
                   help="sieve segment size in integers")
    p.add_argument("--block-size", type=int, default=None,
                   help="bootstrap block width in x; default x_max/40")
    p.add_argument("--output", type=str, default="data/main_run.csv")
    p.add_argument("--theory", type=str, default=None,
                   help="CSV with columns q, a, b, c2_HL (or C_HL)")
    p.add_argument("--skip-sieve", action="store_true")
    p.add_argument("--next-order", action="store_true")
    p.add_argument("--x-min-fit", type=float, default=None,
                   help="drop checkpoints below this in the fits")
    p.add_argument("--bootstrap", type=int, default=400,
                   help="bootstrap resamples; 0 to skip")
    args = p.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    blocks_path = output.parent / "blocks.csv"

    x_min_fit = args.x_min_fit if args.x_min_fit else max(args.x_min, 1e8)

    if not args.skip_sieve:
        ckpts = log_spaced_checkpoints(args.x_min, args.x_max, args.per_decade)
        block_size = args.block_size or max(int(args.x_max // 40), 10**6)
        print(f"Checkpoints ({len(ckpts)}): {ckpts[0]:.3g} ... {ckpts[-1]:.3g}")
        print(f"Sieving to {args.x_max:.3g} (segment {args.segment:,}, "
              f"block {block_size:,})")
        t0 = time.time()
        run_counters(None, ckpts, moduli=MODULI, output_csv=output,
                     segment=args.segment, blocks_csv=blocks_path,
                     block_size=block_size)
        print(f"Wrote {output} in {time.time() - t0:.1f}s")

    df = compute_bias_from_checkpoints(output)
    pi_max = int(df.loc[df["x"].idxmax(), "pi_x"])
    print(f"Loaded {len(df)} rows; pi({df['x'].max():.3g}) = {pi_max:,}")

    # Primary: two-term fit with c1 fixed at the LOS-predicted value.
    fits = fit_two_term(df, x_min_for_fit=x_min_fit)
    fits_path = output.parent / "fits.csv"
    fits.to_csv(fits_path, index=False)
    print(f"Wrote {fits_path} ({len(fits)} cells, two-term fit)")

    # Comparator: naive one-term model.
    naive = fit_wls(df, include_next_order=args.next_order,
                    x_min_for_fit=x_min_fit)
    naive.to_csv(output.parent / "fits_naive.csv", index=False)

    # Misspecification diagnostic.
    drift = fit_window_drift(df, x_min_for_fit=x_min_fit)
    if not drift.empty:
        drift.to_csv(output.parent / "drift.csv", index=False)
        summary = summarize_drift(drift)
        summary.to_csv(output.parent / "drift_summary.csv", index=False)
        med = summary["drift_ratio"].median()
        print(f"Window-drift: median spread ratio (one-term / two-term) = {med:.2f}")

    # Honest error bars.
    if args.bootstrap and blocks_path.exists():
        try:
            boot = fit_bootstrap(df, blocks_csv=blocks_path,
                                 n_resamples=args.bootstrap,
                                 x_min_for_fit=x_min_fit)
            boot.to_csv(output.parent / "bootstrap.csv", index=False)
            merged = fits.merge(boot, on=["q", "a", "b"], how="inner")
            if not merged.empty:
                ratio = (merged["c2_boot_err"] / merged["c2_err"]).median()
                print(f"Wrote bootstrap.csv; median error inflation "
                      f"(bootstrap / WLS) = {ratio:.2f}x")
        except (FileNotFoundError, ValueError) as exc:
            print(f"Bootstrap skipped: {exc}")

    plot.setup_style()
    plot.fig01_bias_vs_logx(df, fits, q=3)
    plot.fig04_pair_type_split(fits)
    plot.fig05_C_of_q(fits)
    if args.next_order:
        plot.fig06_next_order(df, naive, q=3)

    if args.theory and Path(args.theory).exists():
        theory_df = pd.read_csv(args.theory)
        comparison = compare_to_theory(fits, theory_df)
        comparison.to_csv(output.parent / "comparison.csv", index=False)
        print(f"Wrote comparison.csv")

        resid = comparison["residual"].abs()
        print(f"  |c2_emp - c2_HL|: mean {resid.mean():.4f}, "
              f"max {resid.max():.4f}")
        worst = comparison.loc[resid.idxmax()]
        print(f"  worst cell: q={int(worst['q'])} "
              f"({int(worst['a'])},{int(worst['b'])}) "
              f"emp={worst['c2_emp']:.4f} HL={worst['c2_HL']:.4f}")

        plot.fig02_C_emp_vs_C_HL(comparison)
        plot.fig03_residuals(comparison)
    else:
        print("No theory CSV - skipping comparison. "
              "Run `python theory.py` to generate data/theory_C_HL.csv.")


if __name__ == "__main__":
    main()
