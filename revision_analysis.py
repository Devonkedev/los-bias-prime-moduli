"""Analyses added in the first revision of the manuscript.

Writes CSVs to revision/. Each section is independent and reads only the
existing run outputs in data/ plus theory.py. Nothing here re-runs the sieve.

  1. Euler-product truncation convergence           -> revision/euler_convergence*.csv
  2. Fit on disjoint increments (proper treatment of nested checkpoints)
                                                      -> revision/increment_fit.csv
  3. Diagnosis of the original block bootstrap       -> revision/bootstrap_diagnosis.csv
  4. Over-dispersion of pair counts across blocks    -> revision/dispersion.csv
  5. Table 1 scale: residuals in units that matter   -> revision/table1_scale.csv
  6. Table 2 validation: permutation, CV, symmetry   -> revision/table2_validation.csv
  7. Crossover vs convergence scales                 -> revision/convergence_scales.csv
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import theory  # noqa: E402

OUT = Path(__file__).parent / "revision"
OUT.mkdir(exist_ok=True)
DATA = Path(__file__).parent / "data"
MODULI = (3, 7, 11, 13, 17, 19)
RNG = np.random.default_rng(20261010)


# --------------------------------------------------------------------------- 1
def euler_convergence(limits=(10**4, 10**5, 10**6, 10**7, 10**8)):
    """Recompute A_{q,chi} in float64 complex for increasing truncation P."""
    pmax = max(limits)
    flags = np.ones(pmax, dtype=bool)
    flags[:2] = False
    for i in range(2, int(pmax**0.5) + 1):
        if flags[i]:
            flags[i * i::i] = False
    primes = np.flatnonzero(flags)

    rows_A, c2_by_P = [], {}
    for q in MODULI:
        g = theory.primitive_root(q)
        ind = np.zeros(q, dtype=np.int64)
        v = 1
        for k in range(q - 1):
            ind[v] = k
            v = v * g % q
        res = primes % q
        for j in range(1, q - 1):
            if not theory.is_odd_character(q, j):
                continue
            chi_p = np.where(res == 0, 0,
                             np.exp(2j * np.pi * j * ind[res] / (q - 1)))
            fac = np.where(primes == q, 1 - chi_p / primes,
                           1 - (1 - chi_p) ** 2 / (primes - 1.0) ** 2)
            logfac = np.log(fac.astype(np.complex128))
            cum = np.cumsum(logfac)
            for P in limits:
                n = np.searchsorted(primes, P)
                A = np.exp(cum[n - 1])
                rows_A.append({"q": q, "j": j, "P": P, "A_re": A.real, "A_im": A.imag})
    dfA = pd.DataFrame(rows_A)

    # c2 is linear in the C_{q,chi}; rebuild c2 per cell for each P by scaling
    # theory's C_{q,chi} (computed at P=1e6) by A(P)/A(1e6).
    recs = []
    for q in MODULI:
        phi = q - 1
        for a in range(1, q):
            for b in range(1, q):
                if a == b:
                    val = theory.c2_diagonal_closed_form(q)
                    for P in limits:
                        recs.append({"q": q, "a": a, "b": b, "P": P, "c2": val})
                    continue
                for P in limits:
                    total = 0j
                    for j in range(1, q - 1):
                        Cj = complex(theory.C_q_chi(q, j))
                        if Cj == 0:
                            continue
                        r = dfA[(dfA.q == q) & (dfA.j == j)]
                        A_P = complex(*r[r.P == P][["A_re", "A_im"]].iloc[0])
                        A_6 = complex(*r[r.P == 10**6][["A_re", "A_im"]].iloc[0])
                        Cj_P = Cj * A_P / A_6
                        chi = lambda t: complex(theory.chi_value(q, j, t))
                        total += Cj_P * (chi(b - a) + (chi(b) - chi(a)) / phi)
                    val = math.log(2 * math.pi / q) / 2 + q / phi * total.real
                    recs.append({"q": q, "a": a, "b": b, "P": P, "c2": val})
    dfc = pd.DataFrame(recs)
    ref = dfc[dfc.P == max(limits)].set_index(["q", "a", "b"])["c2"]
    summ = []
    for P in limits:
        cur = dfc[dfc.P == P].set_index(["q", "a", "b"])["c2"]
        d = (cur - ref).abs()
        summ.append({"P": P, "max_abs_dc2_vs_P1e8": d.max(), "median_abs_dc2": d.median()})
    # consistency with the published theory table (P = 1e6, mpmath)
    th = pd.read_csv(DATA / "theory_C_HL.csv").set_index(["q", "a", "b"])["c2_HL"]
    cur6 = dfc[dfc.P == 10**6].set_index(["q", "a", "b"])["c2"]
    check = float((cur6 - th).abs().max())
    dfA.to_csv(OUT / "euler_convergence_A.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / "euler_convergence.csv", index=False)
    # analytic tail bound: sum_{p>P} 4/(p-1)^2 < 4/(P-1) (crude), and
    # with Mertens-type density ~ 4/(P log P)
    bounds = [{"P": P, "tail_bound_crude": 4 / (P - 1), "tail_bound_density": 4 / (P * math.log(P))}
              for P in limits]
    pd.DataFrame(bounds).to_csv(OUT / "euler_tail_bounds.csv", index=False)
    return pd.DataFrame(summ), check


# --------------------------------------------------------------------------- 2
def increment_fit(x_min=1e8):
    """Fit c2 on disjoint increments between checkpoints.

    Cumulative counts N(x_k) are nested, so their errors are correlated. The
    increments D_k = N(x_k) - N(x_{k-1}) over disjoint ranges are (to the extent
    pair counts in disjoint stretches are independent) independent. Writing the
    model as E[N(x)] = pi(x)(1 + B(x))/phi^2 with B = c1*u + c2*v gives a model
    for each increment that is linear in c2, fitted by WLS with binomial
    variances. The first observation is the cumulative count at x_min itself,
    which is disjoint from every later increment. This is equivalent to GLS on
    the cumulative counts with Cov(N_i, N_j) = Var(N_min(i,j)).
    """
    df = pd.read_csv(DATA / "main_run.csv")
    old = pd.read_csv(DATA / "fits.csv").set_index(["q", "a", "b"])
    boot = pd.read_csv(DATA / "bootstrap.csv").set_index(["q", "a", "b"])
    rows = []
    for (q, a, b), sub in df.groupby(["q", "a", "b"]):
        sub = sub[sub.x >= x_min].sort_values("x")
        x = sub.x.to_numpy(float)
        N = sub.count_ab.to_numpy(float)
        pi = sub.pi_x.to_numpy(float)
        phi2 = (q - 1) ** 2
        p0 = 1 / phi2
        L = np.log(x)
        u = np.log(L) / L
        v = 1 / L
        c1 = theory.c1(q, a, b)
        # cumulative design: phi2*N - pi - c1*pi*u = c2*pi*v
        yc = phi2 * N - pi - c1 * pi * u
        xc = pi * v
        # increments (first row = cumulative at x_min)
        y = np.diff(np.concatenate([[0.0], yc]))
        X = np.diff(np.concatenate([[0.0], xc]))
        dpi = np.diff(np.concatenate([[0.0], pi]))
        var = phi2**2 * dpi * p0 * (1 - p0)
        w = 1 / var
        c2 = float((w * X * y).sum() / (w * X * X).sum())
        se = float(1 / math.sqrt((w * X * X).sum()))
        resid = y - c2 * X
        dof = len(y) - 1
        chi2_dof = float((w * resid**2).sum() / dof)
        rows.append({"q": q, "a": a, "b": b, "c2_inc": c2, "se_inc": se,
                     "chi2_dof_inc": chi2_dof,
                     "se_inc_scaled": se * math.sqrt(max(chi2_dof, 1.0)),
                     "c2_cumWLS": old.loc[(q, a, b), "c2"],
                     "se_cumWLS": old.loc[(q, a, b), "c2_err"],
                     "c2_boot_mean": boot.loc[(q, a, b), "c2_boot"],
                     "se_boot": boot.loc[(q, a, b), "c2_boot_err"],
                     "n_obs": len(y)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "increment_fit.csv", index=False)
    return out


# --------------------------------------------------------------------------- 3
def bootstrap_diagnosis(n_resamples=500, cells=((3, 1, 1), (7, 1, 3), (13, 2, 5), (19, 15, 2))):
    """Re-run the original block bootstrap and two variants on a few cells.

    original   : resample all 40 equal-width blocks (block 0 = [2, 2.5e9])
    fix_block0 : block 0 always first and never resampled; resample blocks 1..39
    The bias in block 0 (small x) is far larger than in later blocks, so how
    many copies of it a resample contains dominates the spread.
    """
    blocks = pd.read_csv(DATA / "blocks.csv")
    hi = blocks.groupby("block_index")["x_hi"].first().sort_index().to_numpy(float)
    npr = blocks.groupby("block_index")["n_primes"].first().sort_index().to_numpy(float)
    nb = len(hi)
    rows = []
    for (q, a, b) in cells:
        cnt = (blocks[(blocks.q == q) & (blocks.a == a) & (blocks.b == b)]
               .set_index("block_index")["count_ab"].sort_index().to_numpy(float))
        phi2 = (q - 1) ** 2
        c1 = theory.c1(q, a, b)
        L = np.log(hi)
        u, v = np.log(L) / L, 1 / L

        def fit_from(order):
            cc = np.cumsum(cnt[order])
            cp = np.cumsum(npr[order])
            bias = cc / (cp / phi2) - 1
            r = bias - c1 * u
            return float(v @ r / (v @ v))

        point = fit_from(np.arange(nb))
        for variant in ("original", "fix_block0"):
            draws = []
            for _ in range(n_resamples):
                if variant == "original":
                    pick = RNG.integers(0, nb, nb)
                else:
                    pick = np.concatenate([[0], RNG.integers(1, nb, nb - 1)])
                draws.append(fit_from(pick))
            d = np.asarray(draws)
            n0 = None
            rows.append({"q": q, "a": a, "b": b, "variant": variant,
                         "point_same_design": point, "boot_mean": d.mean(),
                         "boot_sd": d.std(ddof=1)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "bootstrap_diagnosis.csv", index=False)
    return out


# --------------------------------------------------------------------------- 4
def dispersion():
    """Empirical variance of per-block pair fractions vs binomial expectation.

    Blocks 1..39 (x from 2.5e9 to 1e11, equal width). Within each cell, the
    expected pair fraction drifts slowly with x; remove it with a quadratic in
    1/log x fitted across blocks, then compare residual variance with
    p(1-p)/n per block. Ratio ~ 1 means pair counts in disjoint stretches behave
    like independent binomial draws; > 1 means extra serial dependence.
    """
    blocks = pd.read_csv(DATA / "blocks.csv")
    blocks = blocks[blocks.block_index >= 1]
    rows = []
    for (q, a, b), g in blocks.groupby(["q", "a", "b"]):
        g = g.sort_values("block_index")
        n = g.n_primes.to_numpy(float)
        f = g.count_ab.to_numpy(float) / n
        mid = 0.5 * (g.x_lo.to_numpy(float) + g.x_hi.to_numpy(float))
        t = 1 / np.log(mid)
        X = np.column_stack([np.ones_like(t), t, t**2])
        w = n
        beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], f * np.sqrt(w), rcond=None)[0]
        r = f - X @ beta
        p = f.mean()
        z2 = r**2 / (p * (1 - p) / n)
        rows.append({"q": q, "a": a, "b": b,
                     "dispersion": float(z2.sum() / (len(f) - 3))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "dispersion.csv", index=False)
    return out


# --------------------------------------------------------------------------- 5
def table1_scale(x_ref=1e11):
    cmp_ = pd.read_csv(DATA / "comparison.csv")
    inc = pd.read_csv(OUT / "increment_fit.csv")
    m = cmp_.merge(inc[["q", "a", "b", "se_inc_scaled"]], on=["q", "a", "b"])
    L = math.log(x_ref)
    m["abs_res"] = (m.c2_emp - m.c2_HL).abs()
    m["rel_res"] = m.abs_res / m.c2_HL.abs()
    m["bias_shift_at_1e11"] = m.abs_res / L        # residual in units of B itself
    m["pred_bias_at_1e11"] = (m.c1_HL * math.log(L) / L + m.c2_HL / L).abs()
    m["shift_over_pred_bias"] = m.bias_shift_at_1e11 / m.pred_bias_at_1e11
    m["z_inc"] = m.abs_res / m.se_inc_scaled
    s = m.groupby("q").agg(
        cells=("abs_res", "size"),
        median_rel_res=("rel_res", "median"),
        median_shift_over_pred_bias=("shift_over_pred_bias", "median"),
        median_z_inc_scaled=("z_inc", "median"),
        min_z_inc_scaled=("z_inc", "min"),
    ).reset_index()
    s.to_csv(OUT / "table1_scale.csv", index=False)
    m.to_csv(OUT / "per_cell_scale.csv", index=False)
    return s


# --------------------------------------------------------------------------- 6
def _r2_by_class(y, cls):
    ybar = y.mean()
    means = pd.Series(y).groupby(cls).transform("mean").to_numpy()
    return 1 - ((y - means) ** 2).sum() / ((y - ybar) ** 2).sum()


def table2_validation(n_perm=10000, n_cv_rep=200, k=5):
    cmp_ = pd.read_csv(DATA / "comparison.csv")
    cmp_ = cmp_[cmp_.a != cmp_.b].copy()
    cmp_["d"] = (cmp_.b - cmp_.a) % cmp_.q
    rows = []
    for q, g in cmp_.groupby("q"):
        if q == 3:
            continue
        # symmetry orbit: (a,b) ~ (-b,-a); same difference class, same c2_HL
        g = g.copy()
        g["orbit"] = [tuple(sorted([(a, b), ((-b) % q, (-a) % q)])) for a, b in zip(g.a, g.b)]
        for target in ("c2_HL", "c2_emp", "residual"):
            y = g[target].to_numpy(float)
            cls = g.d.to_numpy()
            r2 = _r2_by_class(y, cls)
            perm = np.array([_r2_by_class(y, RNG.permutation(cls)) for _ in range(n_perm)])
            p_perm = (1 + (perm >= r2).sum()) / (n_perm + 1)
            # repeated K-fold CV: predict held-out cell by its class mean in training
            cv = []
            idx = np.arange(len(y))
            for _ in range(n_cv_rep):
                RNG.shuffle(idx)
                folds = np.array_split(idx, k)
                pred = np.empty_like(y)
                for f in folds:
                    tr = np.setdiff1d(idx, f)
                    cm = pd.Series(y[tr]).groupby(cls[tr]).mean()
                    pred[f] = [cm.get(c, y[tr].mean()) for c in cls[f]]
                cv.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            # collapse symmetric pairs: one observation per orbit
            go = g.groupby("orbit").agg(y=(target, "mean"), d=("d", "first")).reset_index()
            r2_orb = _r2_by_class(go.y.to_numpy(float), go.d.to_numpy())
            n_orb, n_cls = len(go), go.d.nunique()
            # F-test on orbit-collapsed data
            from mpmath import betainc
            if n_orb > n_cls:
                d1, d2 = n_cls - 1, n_orb - n_cls
                F = (r2_orb / d1) / ((1 - r2_orb) / d2)
                p_F_orb = float(betainc(d2 / 2, d1 / 2, 0, d2 / (d2 + d1 * F), regularized=True))
            else:
                p_F_orb = float("nan")
            rows.append({"q": q, "target": target, "n_cells": len(y),
                         "n_classes": len(set(cls)), "R2": r2,
                         "perm_p": p_perm, "perm_R2_95": float(np.percentile(perm, 95)),
                         "cv_R2_mean": float(np.mean(cv)),
                         "n_orbits": n_orb, "R2_orbits": r2_orb, "p_F_orbits": p_F_orb})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table2_validation.csv", index=False)
    return out


# --------------------------------------------------------------------------- 7
def convergence_scales(x_ref=1e11):
    th = pd.read_csv(DATA / "theory_C_HL.csv")
    off = th[th.a != th.b]
    rows = []
    for q, g in off.groupby("q"):
        r = (g.c2_HL.abs() / g.c1_HL.abs())
        for label, R in (("median", r.median()), ("max", r.max())):
            # crossover: |c1| loglog x = |c2|  ->  x = exp(exp(R))
            log10_x = math.exp(R) / math.log(10)
            rows.append({"q": q, "stat": label, "abs_c2_over_c1": R,
                         "log10_crossover_x": log10_x,
                         "ratio_term2_term1_at_1e11": R / math.log(math.log(x_ref))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "convergence_scales.csv", index=False)
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    if "1" in which:
        s, chk = euler_convergence()
        print("\n[1] Euler truncation\n", s.to_string(index=False))
        print("   max |c2(P=1e6, float) - published table| =", chk)
    if "2" in which:
        inc = increment_fit()
        g = inc.groupby("q").agg(
            med_abs_dc2_inc_vs_cum=("c2_inc", lambda s: 0),
        )
        inc["d_inc_cum"] = (inc.c2_inc - inc.c2_cumWLS).abs()
        inc["d_boot_cum"] = (inc.c2_boot_mean - inc.c2_cumWLS).abs()
        print("\n[2] Increment fit (per-q medians)\n",
              inc.groupby("q")[["d_inc_cum", "se_cumWLS", "se_inc", "chi2_dof_inc",
                                "se_inc_scaled", "se_boot", "d_boot_cum"]].median().to_string())
    if "3" in which:
        print("\n[3] Bootstrap diagnosis\n", bootstrap_diagnosis().to_string(index=False))
    if "4" in which:
        dsp = dispersion()
        print("\n[4] Dispersion by q (median, IQR)\n",
              dsp.groupby("q").dispersion.describe()[["50%", "25%", "75%"]].to_string())
    if "5" in which:
        print("\n[5] Table 1 scale\n", table1_scale().to_string(index=False))
    if "6" in which:
        print("\n[6] Table 2 validation\n", table2_validation().to_string(index=False))
    if "7" in which:
        print("\n[7] Convergence scales\n", convergence_scales().to_string(index=False))
