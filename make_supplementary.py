"""Table S1 (Online Resource 1): one row per (q, a, b) pair, all 864 pairs.

Reads revision/increment_fit.csv (written by revision_analysis.py) and
data/theory_C_HL.csv (written by theory.py). Writes
revision/supplementary_864_pairs.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent


def main() -> None:
    inc = pd.read_csv(ROOT / "revision" / "increment_fit.csv")
    th = pd.read_csv(ROOT / "data" / "theory_C_HL.csv")[["q", "a", "b", "c1_HL", "c2_HL"]]
    m = inc.merge(th, on=["q", "a", "b"]).sort_values(["q", "a", "b"])
    out = pd.DataFrame({
        "q": m.q, "a": m.a, "b": m.b,
        "pair_type": ["diagonal" if a == b else "off-diagonal" for a, b in zip(m.a, m.b)],
        "residue_difference": (m.b - m.a) % m.q,
        "c1_predicted": m.c1_HL,
        "c2_predicted": m.c2_HL,
        "c2_measured": m.c2_inc,
        "se_binomial": m.se_inc,
        "chi2_per_dof": m.chi2_dof_inc,
        "se_reported": m.se_inc_scaled,
        "residual": m.c2_inc - m.c2_HL,
        "z": (m.c2_inc - m.c2_HL) / m.se_inc_scaled,
        "c2_measured_cumulative_WLS": m.c2_cumWLS,
    })
    path = ROOT / "revision" / "supplementary_864_pairs.csv"
    out.to_csv(path, index=False, float_format="%.6g")
    print(f"Wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
