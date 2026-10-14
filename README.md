# Second-order constants of the Lemke Oliver–Soundararajan bias across prime moduli

Code and data for the paper *Measuring the Second-Order Constants of the Lemke Oliver–Soundararajan Bias Across Prime Moduli* by Devyansh Garg.

Consecutive primes avoid repeating their last digit, and more generally their residue modulo q, more often than chance would suggest (Lemke Oliver and Soundararajan, 2016). This repository counts consecutive prime pairs by their residues (a, b) modulo q for q = 3, 7, 11, 13, 17 and 19, which gives 864 residue pairs. It does this at 31 checkpoints up to x = 10^11. It then fits the normalised bias

    B(x) = c1 · log log x / log x + c2 / log x

with c1 fixed at its predicted value, and compares the measured second-order constant c2 with the value the conjecture predicts.

## Setup

Python 3.13. From the repository folder:

    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## Reproducing the paper

Run these from the repository folder, in this order. The data files from the original run are already included, so steps 3 onward can be run without repeating the enumeration.

| Step | Command | Writes | Time |
|---|---|---|---|
| 1. Tests | `python -m pytest tests/` | nothing | about 1 min |
| 2. Predicted constants | `python theory.py` | `data/theory_C_HL.csv` | about 1 min |
| 3. Enumeration to 10^11 | `python compute_los_bias.py --x-max 1e11 --per-decade 6 --segment 50000000 --bootstrap 500 --theory data/theory_C_HL.csv` | `data/main_run.csv`, `data/blocks.csv`, `data/fits.csv`, `data/comparison.csv` and the other files in `data/` | about 11 min on one core |
| 4. Revision analyses | `python revision_analysis.py` | `revision/*.csv` | about 1.5 min |
| 5. Tables 1–4 | `python revision_tables.py` | `revision/rev1_table1.csv` to `rev1_table4.csv`, `revision/rev1_numbers.json` | under 30 s |
| 6. Table S1 | `python make_supplementary.py` | `revision/supplementary_864_pairs.csv` | seconds |
| 7. Figures 1 and 2 | `python make_rev1_figures.py` | `figures/rev1/Fig1`, `Fig2` (PDF, EPS, PNG) | seconds |

Step 3 overwrites the files in `data/`. To check the enumeration without waiting, run it to 10^9 into a separate folder and compare the rows with the same x:

    python compute_los_bias.py --x-max 1e9 --per-decade 6 --segment 50000000 --bootstrap 0 --output check/main_run.csv

The run reported in the paper took 685 seconds on one core of a MacBook Air (Apple M4, 16 GB memory, macOS 26.1) under Python 3.13.

This package was checked on 14 October 2026 in a fresh Python 3.13.5 environment with the versions in `requirements.txt`. All 23 tests pass. The enumeration to 10^9 reproduces every count in `data/main_run.csv` at the 19 checkpoints up to 10^9. Steps 2 and 4–6 reproduce the files in this repository to floating-point precision. The one exception is `revision/table2_validation.csv`, an intermediate check that is not reported in the paper. Its permutation and cross-validation columns use random draws and differ in the second or third significant figure from the version included here. Figures 1 and 2 are drawn from the same numbers, and only the fonts may differ between systems.

## Where each result in the paper comes from

| Paper | File | Script |
|---|---|---|
| Table 1 | `revision/rev1_table1.csv` | `revision_tables.py` |
| Table 2 | `revision/rev1_table2.csv` | `revision_tables.py` |
| Table 3 | `revision/rev1_table3.csv` | `revision_tables.py` |
| Table 4 | `revision/rev1_table4.csv` | `revision_tables.py` |
| Numbers quoted in the text | `revision/rev1_numbers.json` | `revision_tables.py` |
| Figure 1 | `figures/rev1/Fig1.*` | `make_rev1_figures.py` |
| Figure 2 | `figures/rev1/Fig2.*` | `make_rev1_figures.py` |
| Online Resource 1, Table S1 | `revision/supplementary_864_pairs.csv` | `make_supplementary.py` |
| Online Resource 1, Table S2 | `revision/euler_convergence.csv` | `revision_analysis.py` |

## Data files

`data/main_run.csv` has one row per checkpoint and residue pair (26,784 rows). The columns are: `x` (checkpoint), `q` (modulus), `a`, `b` (residues of p and the next prime), `count_ab` (number of consecutive pairs up to x with those residues) and `pi_x` (number of primes up to x).

`data/blocks.csv` gives the same counts in 40 disjoint blocks of x of width 2.5 × 10^9. It is used for the dispersion check in Section 2.5.

`data/theory_C_HL.csv` gives the predicted constants `c1_HL` and `c2_HL` for every pair, from an Euler product truncated at 10^6.

`revision/increment_fit.csv` holds the fit used in the paper, on disjoint increments between checkpoints. `c2_inc` is the measured constant, `se_inc` its binomial standard error, `chi2_dof_inc` the chi-square per degree of freedom, and `se_inc_scaled` the reported standard error (binomial × √(chi-square per d.o.f.)).

`data/bootstrap.csv` is the output of the block bootstrap from the first version of the paper. It is not used for any reported uncertainty (see Section 2.5). It is kept because `revision_analysis.py` uses it to show why the bootstrap was replaced.

## Code

| File | Purpose |
|---|---|
| `counters.py` | Segmented sieve and the vectorised counting of consecutive pairs by residue |
| `sieve.py` | Prime counting and streaming helpers used by the tests |
| `theory.py` | Predicted constants c1 and c2 from the Lemke Oliver–Soundararajan conjecture |
| `fit.py` | Fits of the bias model |
| `compute_los_bias.py` | Runs the enumeration and the first fits |
| `substructure.py` | Regression of c2 on the residue difference (earlier diagnostic) |
| `plot.py` | Diagnostic figures from the first run |
| `revision_analysis.py` | Euler-product convergence, increment fit, dispersion and validation checks |
| `revision_tables.py` | Tables 1–4 and the numbers quoted in the paper |
| `make_supplementary.py` | Table S1 |
| `make_rev1_figures.py` | Figures 1 and 2 |
| `tests/` | Checks of the sieve, the counts and the theory code against known values |

## Licence

MIT. See `LICENSE`.
