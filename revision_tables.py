"""Final numbers (Tables 1-4) for the revised manuscript.

Everything is computed from the increment (GLS-equivalent) fit in
revision/increment_fit.csv, so point estimates and uncertainties come from the
same estimator. Writes revision/rev1_numbers.json plus one CSV per table.
"""
from __future__ import annotations

import json
import math
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from mpmath import betainc

sys.path.insert(0, str(Path(__file__).parent))
import theory  # noqa: E402

ROOT = Path(__file__).parent
REV = ROOT / "revision"
RNG = np.random.default_rng(20261010)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    r = p[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(r, 0, 1)
    return out


def f_sf(F, d1, d2):
    return float(betainc(d2 / 2, d1 / 2, 0, d2 / (d2 + d1 * F), regularized=True))


def r2_class(y, c):
    mu = pd.Series(y).groupby(c).transform("mean").to_numpy()
    return 1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def increment_c2(sub, c1, phi2, model="two"):
    """c2 (or one-term C) from disjoint increments of one cell's checkpoints."""
    x = sub.x.to_numpy(float)
    N = sub.count_ab.to_numpy(float)
    pi = sub.pi_x.to_numpy(float)
    L = np.log(x)
    u, v = np.log(L) / L, 1 / L
    if model == "two":
        yc = phi2 * N - pi - c1 * pi * u
    else:  # one-term B = C / log x
        yc = phi2 * N - pi
    xc = pi * v
    y = np.diff(np.concatenate([[0.0], yc]))
    X = np.diff(np.concatenate([[0.0], xc]))
    dpi = np.diff(np.concatenate([[0.0], pi]))
    p0 = 1 / phi2
    w = 1 / (phi2**2 * dpi * p0 * (1 - p0))
    return float((w * X * y).sum() / (w * X * X).sum())


def main():
    inc = pd.read_csv(REV / "increment_fit.csv")
    th = pd.read_csv(ROOT / "data" / "theory_C_HL.csv")[["q", "a", "b", "c1_HL", "c2_HL"]]
    m = inc.merge(th, on=["q", "a", "b"])
    m["res"] = m.c2_inc - m.c2_HL
    m["absres"] = m.res.abs()
    m["rel"] = m.absres / m.c2_HL.abs()
    m["z"] = m.absres / m.se_inc_scaled
    nums = {}

    # ------------------------------------------------------------ Table 1
    t1 = []
    for q, g in m.groupby("q"):
        t1.append({"q": q, "cells": len(g),
                   "corr": np.corrcoef(g.c2_inc, g.c2_HL)[0, 1],
                   "mean_abs": g.absres.mean(), "max_abs": g.absres.max(),
                   "median_rel": g.rel.median(),
                   "median_se": g.se_inc_scaled.median()})
    t1 = pd.DataFrame(t1)
    dg = m[m.a == m.b]
    worst = m.loc[m.absres.idxmax()]
    t1_extra = {"diag_cells": len(dg),
                "diag_corr": float(np.corrcoef(dg.c2_inc, dg.c2_HL)[0, 1]),
                "diag_mean_abs": float(dg.absres.mean()),
                "diag_median_rel": float(dg.rel.median()),
                "all_mean_abs": float(m.absres.mean()), "all_max_abs": float(m.absres.max()),
                "worst": [int(worst.q), int(worst.a), int(worst.b),
                          float(worst.c2_inc), float(worst.c2_HL)],
                "z_median_by_q": {int(q): float(g.z.median()) for q, g in m.groupby("q")},
                "z_min_all": float(m.z.min()),
                "chi2_dof_median_by_q": {int(q): float(g.chi2_dof_inc.median()) for q, g in m.groupby("q")},
                "se_binom_median_by_q": {int(q): float(g.se_inc.median()) for q, g in m.groupby("q")},
                "se_cumwls_median_by_q": {int(q): float(g.se_cumWLS.median()) for q, g in m.groupby("q")},
                "shift_vs_cum_median_by_q": {int(q): float((g.c2_inc - g.c2_cumWLS).abs().median())
                                             for q, g in m.groupby("q")}}
    t1.to_csv(REV / "rev1_table1.csv", index=False)
    nums["table1"] = t1.to_dict("records")
    nums["table1_extra"] = t1_extra

    # ------------------------------------------------------------ Table 2
    off = m[m.a != m.b].copy()
    off["d"] = (off.b - off.a) % off.q
    t2 = []
    for q, g in off.groupby("q"):
        if q == 3:
            continue
        g = g.copy()
        g["orbit"] = [tuple(sorted([(a, b), ((-b) % q, (-a) % q)])) for a, b in zip(g.a, g.b)]
        g["resid"] = g.c2_inc - g.c2_HL
        for label, col in (("Predicted (control)", "c2_HL"), ("Measured", "c2_inc"),
                           ("Residual", "resid")):
            y = g[col].to_numpy(float)
            cls = g.d.to_numpy()
            R2 = r2_class(y, cls)
            # permutation
            perm = np.array([r2_class(y, RNG.permutation(cls)) for _ in range(10000)])
            p_perm = (1 + (perm >= R2).sum()) / 10001
            # repeated 5-fold CV
            cvs = []
            idx = np.arange(len(y))
            for _ in range(200):
                RNG.shuffle(idx)
                pred = np.empty_like(y)
                for f in np.array_split(idx, 5):
                    tr = np.setdiff1d(idx, f)
                    cm = pd.Series(y[tr]).groupby(cls[tr]).mean()
                    pred[f] = [cm.get(c, y[tr].mean()) for c in cls[f]]
                cvs.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            # one observation per symmetric orbit
            go = g.groupby("orbit").agg(y=(col, "mean"), d=("d", "first"))
            n_o, k = len(go), go.d.nunique()
            R2o = r2_class(go.y.to_numpy(float), go.d.to_numpy())
            Fo = (R2o / (k - 1)) / ((1 - R2o) / (n_o - k))
            p_o = f_sf(Fo, k - 1, n_o - k)
            t2.append({"q": q, "target": label, "cells": len(y), "classes": len(set(cls)),
                       "R2": R2, "cv_R2": float(np.mean(cvs)), "perm_p": p_perm,
                       "orbits": n_o, "R2_orbits": R2o, "p_orbits": p_o})
    t2 = pd.DataFrame(t2)
    t2["p_orbits_bh"] = bh(t2.p_orbits.to_numpy())
    t2.to_csv(REV / "rev1_table2.csv", index=False)
    nums["table2"] = t2.to_dict("records")

    # ------------------------------------------------------------ Table 3
    t3 = []
    for q, g in off.groupby("q"):
        if q == 3:
            continue
        emp = g.groupby("d").c2_inc.mean().reindex(range(1, q))
        hl = g.groupby("d").c2_HL.mean().reindex(range(1, q))
        pairs = [d for d in range(1, q) if d < q - d]
        de = float(np.mean([abs(emp[d] + emp[q - d]) / 2 for d in pairs]))
        dh = float(np.mean([abs(hl[d] + hl[q - d]) / 2 for d in pairs]))
        t3.append({"q": q, "classes": q - 1,
                   "corr": float(np.corrcoef(emp, hl)[0, 1]),
                   "emp_spread": float(emp.std(ddof=1)), "pred_spread": float(hl.std(ddof=1)),
                   "ratio": float(emp.std(ddof=1) / hl.std(ddof=1)),
                   "antisym_emp": de, "antisym_pred": dh,
                   "antisym_emp_over_spread": de / float(emp.std(ddof=1))})
    t3 = pd.DataFrame(t3)
    t3.to_csv(REV / "rev1_table3.csv", index=False)
    nums["table3"] = t3.to_dict("records")

    # ------------------------------------------------ windows: drift + convergence
    df = pd.read_csv(ROOT / "data" / "main_run.csv")
    xs = np.sort(df.loc[df.x >= 1e8, "x"].unique())
    edges = np.array_split(xs, 4)
    win = [(float(e[0]), float(e[-1])) for e in edges]
    rows = []
    for (q, a, b), sub in df.groupby(["q", "a", "b"]):
        sub = sub.sort_values("x")
        phi2 = (q - 1) ** 2
        c1 = theory.c1(q, a, b)
        rec = {"q": q, "a": a, "b": b}
        for wi, (lo, hi) in enumerate(win):
            s = sub[(sub.x >= lo) & (sub.x <= hi)]
            # first observation of a window is the increment from the previous
            # checkpoint, so windows are disjoint in the data they use
            prev = sub[sub.x < lo].tail(1)
            s2 = pd.concat([prev, s]) if len(prev) else s
            # refit on increments within the window only
            x = s2.x.to_numpy(float); N = s2.count_ab.to_numpy(float); pi = s2.pi_x.to_numpy(float)
            L = np.log(x); u, v = np.log(L) / L, 1 / L
            for mdl in ("two", "one"):
                yc = phi2 * N - pi - (c1 * pi * u if mdl == "two" else 0)
                xc = pi * v
                y = np.diff(yc); X = np.diff(xc); dpi = np.diff(pi)
                p0 = 1 / phi2
                w = 1 / (phi2**2 * dpi * p0 * (1 - p0))
                rec[f"{mdl}_{wi}"] = float((w * X * y).sum() / (w * X * X).sum())
        rows.append(rec)
    W = pd.DataFrame(rows).merge(th, on=["q", "a", "b"])
    W["spread_two"] = W[[f"two_{i}" for i in range(4)]].max(axis=1) - W[[f"two_{i}" for i in range(4)]].min(axis=1)
    W["spread_one"] = W[[f"one_{i}" for i in range(4)]].max(axis=1) - W[[f"one_{i}" for i in range(4)]].min(axis=1)
    W["ratio"] = W.spread_one / W.spread_two
    nums["drift"] = {"windows": win,
                     "median_spread_one": float(W.spread_one.median()),
                     "median_spread_two": float(W.spread_two.median()),
                     "median_ratio": float(W.ratio.median())}
    W["move"] = W.two_3 - W.two_0
    W["gap"] = W.c2_HL - W.two_0
    W["toward"] = np.sign(W.move) == np.sign(W.gap)
    W["closed"] = W.move / W.gap
    t4 = []
    for q, g in W.groupby("q"):
        n, k = len(g), int(g.toward.sum())
        p = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
        t4.append({"q": q, "cells": n, "toward": k, "frac": k / n,
                   "median_closed": float(g.closed.median()), "sign_p": p})
    t4 = pd.DataFrame(t4)
    t4.to_csv(REV / "rev1_table4.csv", index=False)
    W.to_csv(REV / "rev1_window_fits.csv", index=False)
    nums["table4"] = t4.to_dict("records")

    # ------------------------------------------------ crossover scales
    cs = pd.read_csv(REV / "convergence_scales.csv")
    nums["crossover"] = cs.to_dict("records")

    # ------------------------------------------------ dispersion, euler
    dsp = pd.read_csv(REV / "dispersion.csv")
    nums["dispersion_median_by_q"] = {int(q): float(g.dispersion.median()) for q, g in dsp.groupby("q")}
    nums["euler"] = pd.read_csv(REV / "euler_convergence.csv").to_dict("records")
    bd = pd.read_csv(REV / "bootstrap_diagnosis.csv")
    nums["bootstrap_diag"] = bd.to_dict("records")

    (REV / "rev1_numbers.json").write_text(json.dumps(nums, indent=1, default=float))
    pd.set_option("display.width", 220)
    print("T1\n", t1.round(4).to_string(index=False))
    print({k: v for k, v in t1_extra.items()})
    print("T2\n", t2.round(4).to_string(index=False))
    print("T3\n", t3.round(4).to_string(index=False))
    print("T4\n", t4.round(4).to_string(index=False))
    print("drift", nums["drift"])


if __name__ == "__main__":
    main()
