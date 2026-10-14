"""Substructure regressions on the fitted second-order constants.

The question: beyond the diagonal/off-diagonal split, does c2(q, a, b) depend on
  (i)   (b - a) mod q,
  (ii)  the quadratic-residue status of a,
  (iii) the quadratic-residue status of b?

Two targets are regressed, and the distinction matters:

  target = c2_emp        Is there substructure at all?
  target = c2_emp - c2_HL  Is there substructure the LOS constants FAIL to predict?

The second is the discriminating one. LOS's formula already contains a chi(b-a)
term, so (b-a) structure in c2_emp is *expected* and confirms the theory. Leftover
(b-a) structure in the residuals is what would indicate something the singular
series misses -- the kind of transient, sieve-driven effect Holt 2024 describes.

Predictor coding. (b-a) mod q is circular, so entering it as an integer would
impose a spurious linear order (difference 1 and difference q-1 are adjacent on
the circle but maximally distant as integers). Two encodings are provided, and
the distinction turned out to matter a great deal:

  FULL basis (PRIMARY, run_difference_class_regression). d entered as a
    categorical factor: all q-1 difference classes get their own indicator.
    This is the assumption-free test of "does c2 depend on (b-a)?".

  HARMONIC basis (SECONDARY, run_substructure). The single pair
    cos(2 pi d/q), sin(2 pi d/q), plus the Legendre symbols (a|q), (b|q) in
    {-1,+1}. This is the lowest-frequency circular basis and superficially
    matches the chi(b-a) term of the theory.

A power check settled which to trust. Running the HARMONIC regression on the
THEORETICAL c2_HL values -- where (b-a) structure is present by construction --
gives R^2 = 0.14 (p = 0.43) at q = 7 and R^2 = 0.006 at q = 17 and q = 19. The
two-harmonic design cannot detect the theory's OWN structure at most moduli,
because c2 is a sum over all odd characters, not just the lowest frequency.
Consequently:
  * a null from the harmonic design is uninformative, and
  * a significant harmonic result is NOT evidence about the predicted structure.
The full basis recovers R^2 = 0.93-0.99 on the same theoretical values, as it
should, and is what the paper reports. `power_check` regenerates these numbers.

Diagonal cells (a == b) are excluded: c2 is constant there by LOS eq (1.1) and
d = 0 would sit at a degenerate point of the circular basis.

Multiple comparisons. Benjamini-Hochberg FDR-adjusted q-values are reported
alongside raw p-values; the paper quotes the adjusted values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def legendre_symbol(a: int, q: int) -> int:
    """(a|q) in {-1, 0, +1} for odd prime q, by Euler's criterion.

    >>> legendre_symbol(1, 7)
    1
    >>> legendre_symbol(3, 7)   # 3 is a non-residue mod 7
    -1
    >>> legendre_symbol(2, 7)   # 2 = 3^2 = 9 = 2 mod 7
    1
    """
    a %= q
    if a == 0:
        return 0
    r = pow(a, (q - 1) // 2, q)
    return 1 if r == 1 else -1


def build_design(fits: pd.DataFrame, q: int, target: str) -> pd.DataFrame:
    """Design matrix for one modulus: circular (b-a) harmonics + QR indicators."""
    sub = fits[(fits["q"] == q) & (fits["a"] != fits["b"])].copy()
    if sub.empty:
        return sub

    d = (sub["b"] - sub["a"]) % q
    sub["diff_mod_q"] = d
    sub["cos_d"] = np.cos(2 * np.pi * d / q)
    sub["sin_d"] = np.sin(2 * np.pi * d / q)
    sub["qr_a"] = [legendre_symbol(int(v), q) for v in sub["a"]]
    sub["qr_b"] = [legendre_symbol(int(v), q) for v in sub["b"]]
    sub["target"] = sub[target]
    return sub


def regress_one(sub: pd.DataFrame, weighted: bool = True):
    """OLS/WLS of target on [cos_d, sin_d, qr_a, qr_b]. Returns a fitted model."""
    import statsmodels.api as sm

    X = sm.add_constant(sub[["cos_d", "sin_d", "qr_a", "qr_b"]].to_numpy(float))
    y = sub["target"].to_numpy(float)

    if weighted and "c2_err" in sub.columns:
        err = sub["c2_err"].to_numpy(float)
        w = 1.0 / np.where(err > 0, err**2, np.nan)
        if np.all(np.isfinite(w)):
            return sm.WLS(y, X, weights=w).fit()
    return sm.OLS(y, X).fit()


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment.

    >>> np.round(_bh_adjust(np.array([0.01, 0.02, 0.03])), 4).tolist()
    [0.03, 0.03, 0.03]
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def noise_floor(sub: pd.DataFrame, model) -> dict:
    """What effect size would this data have been able to detect?

    Reports the median standard error on the fitted coefficients (the smallest
    resolvable effect at ~2 sigma), and the observed spread of the theoretical
    prediction across the same cells for scale. If the analysis returns a null,
    these numbers say how far short the data fell.
    """
    se = np.asarray(model.bse[1:], dtype=float)  # drop intercept
    out = {
        "median_coef_se": float(np.median(se)),
        "detectable_at_2sigma": float(2 * np.median(se)),
        "target_sd": float(np.std(sub["target"].to_numpy(float), ddof=1)),
    }
    if "c2_HL" in sub.columns:
        hl = sub["c2_HL"].to_numpy(float)
        out["predicted_sd"] = float(np.std(hl, ddof=1))
    return out


def run_difference_class_regression(fits: pd.DataFrame,
                                    output_csv: str | Path | None =
                                    "data/difference_class_regression.csv"
                                    ) -> pd.DataFrame:
    """PRIMARY substructure test: c2 regressed on (b-a) as a categorical factor.

    Uses the full difference-class basis rather than two harmonics, so it makes
    no assumption about which frequencies carry the structure. Run on three
    targets per modulus:

        c2_HL     the prediction (positive control -- structure is present by
                  construction, so a low R^2 here indicts the design)
        c2        the measurement
        residual  c2 - c2_HL (structure the singular series does NOT predict)

    Returns one row per (q, target).
    """
    import statsmodels.api as sm

    off = fits[fits["a"] != fits["b"]].copy()
    if off.empty:
        return pd.DataFrame()
    off["d"] = (off["b"] - off["a"]) % off["q"]

    targets = {"c2_HL": "c2_HL", "c2_emp": "c2"}
    if "c2_HL" in off.columns and "c2" in off.columns:
        off["residual"] = off["c2"] - off["c2_HL"]
        targets["residual"] = "residual"

    rows = []
    for q, s in off.groupby("q"):
        q = int(q)
        if q <= 3 or s["d"].nunique() < 3:
            continue
        dummies = pd.get_dummies(s["d"], prefix="d", drop_first=True).astype(float)
        X = sm.add_constant(dummies.to_numpy())
        for label, col in targets.items():
            if col not in s.columns:
                continue
            y = s[col].to_numpy(float)
            model = sm.OLS(y, X).fit()
            rows.append({
                "q": q,
                "target": label,
                "n_cells": int(len(s)),
                "n_classes": int(s["d"].nunique()),
                "r_squared": float(model.rsquared),
                "r_squared_adj": float(model.rsquared_adj),
                "f_pvalue": float(model.f_pvalue),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        mask = out["f_pvalue"].notna()
        out.loc[mask, "f_pvalue_bh"] = _bh_adjust(out.loc[mask, "f_pvalue"].to_numpy())
        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(output_csv, index=False)
    return out


def power_check(fits: pd.DataFrame) -> pd.DataFrame:
    """Can each design detect the structure the THEORY is known to contain?

    Regresses c2_HL -- where (b-a) dependence is present by construction -- on
    both designs. A design that scores poorly here cannot be trusted to detect
    the same structure in noisier empirical data, so its nulls are uninformative.
    """
    import statsmodels.api as sm

    off = fits[fits["a"] != fits["b"]].copy()
    if off.empty or "c2_HL" not in off.columns:
        return pd.DataFrame()
    off["d"] = (off["b"] - off["a"]) % off["q"]

    rows = []
    for q, s in off.groupby("q"):
        q = int(q)
        if q <= 3:
            continue
        y = s["c2_HL"].to_numpy(float)

        harm = build_design(fits.assign(c2_HL=fits["c2_HL"]), q, "c2_HL")
        Xh = sm.add_constant(
            harm[["cos_d", "sin_d", "qr_a", "qr_b"]].to_numpy(float))
        mh = sm.OLS(harm["target"].to_numpy(float), Xh).fit()

        dummies = pd.get_dummies(s["d"], prefix="d", drop_first=True).astype(float)
        mf = sm.OLS(y, sm.add_constant(dummies.to_numpy())).fit()

        rows.append({
            "q": q,
            "n_cells": int(len(s)),
            "harmonic_r2_on_theory": float(mh.rsquared),
            "harmonic_p_on_theory": float(mh.f_pvalue),
            "full_basis_r2_on_theory": float(mf.rsquared),
            "full_basis_p_on_theory": float(mf.f_pvalue),
        })
    return pd.DataFrame(rows)


def difference_class_summary(fits: pd.DataFrame) -> pd.DataFrame:
    """Per-modulus tests of the (b-a) structure predicted by LOS 2017.

    LOS 2017 eq (1.5) says c2(q;(a,b))/q = C(b-a) + O((log q)^2/sqrt q), and
    Theorem 1.1 that C(k) = -C(-k) is odd. Two consequences we can test without
    any regression machinery:

      corr_class_mean : correlation between the empirical and predicted mean of
                        c2 within each difference class d = (b-a) mod q. Tests
                        whether the *assignment* of structure to classes is right.
      spread_ratio    : ratio of empirical to predicted standard deviation of
                        those class means. Tests whether the *magnitude* of the
                        substructure is right, which can hold even when the
                        assignment is lost in noise.
      antisym_defect  : mean |m(d) + m(q-d)| / 2 over classes, which is 0 for a
                        perfectly odd function. Reported for both empirical and
                        predicted values so the predicted value acts as a scale.
    """
    out = []
    off = fits[fits["a"] != fits["b"]].copy()
    if off.empty or "c2_HL" not in off.columns:
        return pd.DataFrame()
    off["d"] = (off["b"] - off["a"]) % off["q"]

    for q, s in off.groupby("q"):
        q = int(q)
        if q <= 3:
            continue
        emp = s.groupby("d")["c2"].mean().reindex(range(1, q))
        hl = s.groupby("d")["c2_HL"].mean().reindex(range(1, q))
        if emp.isna().any() or hl.isna().any():
            continue

        pairs = [d for d in range(1, q) if d < q - d]
        emp_def = float(np.mean([abs(emp[d] + emp[q - d]) / 2 for d in pairs]))
        hl_def = float(np.mean([abs(hl[d] + hl[q - d]) / 2 for d in pairs]))

        out.append({
            "q": q,
            "n_classes": int(len(emp)),
            "corr_class_mean": float(np.corrcoef(emp.to_numpy(), hl.to_numpy())[0, 1]),
            "emp_class_spread": float(emp.std(ddof=1)),
            "hl_class_spread": float(hl.std(ddof=1)),
            "spread_ratio": float(emp.std(ddof=1) / hl.std(ddof=1)),
            "antisym_defect_emp": emp_def,
            "antisym_defect_hl": hl_def,
        })
    return pd.DataFrame(out)


def run_substructure(fits_csv: str | Path = "data/fits.csv",
                     comparison_csv: str | Path = "data/comparison.csv",
                     output_csv: str | Path = "data/substructure.csv"
                     ) -> pd.DataFrame:
    """Run all regressions and write the summary table."""
    comparison_csv = Path(comparison_csv)
    if comparison_csv.exists():
        fits = pd.read_csv(comparison_csv)
    else:
        fits = pd.read_csv(fits_csv)

    if "c2_emp" in fits.columns and "c2" not in fits.columns:
        fits = fits.rename(columns={"c2_emp": "c2", "c2_emp_err": "c2_err"})

    targets = ["c2"]
    if "c2_HL" in fits.columns:
        fits["residual_vs_HL"] = fits["c2"] - fits["c2_HL"]
        targets.append("residual_vs_HL")

    rows = []
    for target in targets:
        for q in sorted(fits["q"].unique()):
            q = int(q)
            if q <= 3:
                # q=3 has a single off-diagonal difference class per a; the
                # circular basis is rank-deficient. Skip with a note.
                rows.append({
                    "target": target, "q": q, "n_cells": 0,
                    "note": "skipped: too few off-diagonal cells for the "
                            "circular basis",
                })
                continue

            sub = build_design(fits, q, target)
            if len(sub) < 8:
                rows.append({"target": target, "q": q, "n_cells": len(sub),
                             "note": "skipped: too few cells"})
                continue

            model = regress_one(sub)
            floor = noise_floor(sub, model)
            names = ["const", "cos_d", "sin_d", "qr_a", "qr_b"]
            row = {
                "target": target,
                "q": q,
                "n_cells": int(len(sub)),
                "r_squared": float(model.rsquared),
                "r_squared_adj": float(model.rsquared_adj),
                "f_pvalue": float(model.f_pvalue),
                **floor,
            }
            for i, nm in enumerate(names):
                row[f"coef_{nm}"] = float(model.params[i])
                row[f"se_{nm}"] = float(model.bse[i])
                row[f"p_{nm}"] = float(model.pvalues[i])
            rows.append(row)

    out = pd.DataFrame(rows)

    # FDR across the omnibus tests that actually ran.
    ran = out["f_pvalue"].notna()
    if ran.any():
        out.loc[ran, "f_pvalue_bh"] = _bh_adjust(out.loc[ran, "f_pvalue"].to_numpy())
        for nm in ["cos_d", "sin_d", "qr_a", "qr_b"]:
            col = f"p_{nm}"
            if col in out.columns:
                mask = out[col].notna()
                if mask.any():
                    out.loc[mask, f"{col}_bh"] = _bh_adjust(
                        out.loc[mask, col].to_numpy())

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    classes = difference_class_summary(fits)
    if not classes.empty:
        classes.to_csv(output_csv.parent / "difference_classes.csv", index=False)

    return out


def describe(out: pd.DataFrame) -> str:
    """Human-readable summary for RESULTS.md / the paper."""
    lines = []
    for target in out["target"].dropna().unique():
        sub = out[(out["target"] == target) & out["r_squared"].notna()]
        if sub.empty:
            continue
        lines.append(f"\n### target = {target}")
        lines.append(f"{'q':>4} {'n':>4} {'R^2':>7} {'adjR^2':>7} "
                     f"{'F p':>10} {'F p (BH)':>10}  significant predictors (BH<0.05)")
        for _, r in sub.iterrows():
            sig = [nm for nm in ["cos_d", "sin_d", "qr_a", "qr_b"]
                   if r.get(f"p_{nm}_bh", 1) < 0.05]
            lines.append(
                f"{int(r['q']):>4} {int(r['n_cells']):>4} {r['r_squared']:>7.3f} "
                f"{r['r_squared_adj']:>7.3f} {r['f_pvalue']:>10.2e} "
                f"{r.get('f_pvalue_bh', float('nan')):>10.2e}  "
                f"{', '.join(sig) if sig else '(none)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    import doctest

    failures, tests = doctest.testmod()
    print(f"doctests: {tests - failures}/{tests} passed")

    import pandas as _pd

    src = Path("data/comparison.csv")
    fits = (_pd.read_csv(src).rename(columns={"c2_emp": "c2",
                                              "c2_emp_err": "c2_err"})
            if src.exists() else _pd.read_csv("data/fits.csv"))

    print("\n### POWER CHECK: can each design see the theory's own structure?")
    pc = power_check(fits)
    if not pc.empty:
        print(pc.to_string(index=False, float_format=lambda v: f"{v:10.4g}"))
        print("  -> the harmonic design fails on the theory itself at most q;")
        print("     its nulls are uninformative and it is reported only as a diagnostic.")

    print("\n### PRIMARY: c2 ~ (b-a) as a categorical factor (full basis)")
    dcr = run_difference_class_regression(fits)
    if not dcr.empty:
        print(dcr.to_string(index=False, float_format=lambda v: f"{v:10.4g}"))

    print("\n### SECONDARY (diagnostic only): two-harmonic + QR design")
    result = run_substructure()
    print(describe(result))

    classes = difference_class_summary(fits)
    if not classes.empty:
        print("\n### difference-class means (LOS 2017 predictions)")
        print(classes.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print("\nWrote data/substructure.csv, data/difference_classes.csv, "
          "data/difference_class_regression.csv")
