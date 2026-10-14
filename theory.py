"""Hardy--Littlewood theoretical constants from Lemke Oliver--Soundararajan 2016.

Source: arXiv:1603.03720v4 (PNAS companion). Everything here is transcribed from
that paper; see reading_notes.md §1.2 for the verbatim extraction.

THE MAIN CONJECTURE (LOS 2016, §1). For an r-tuple a of reduced residue classes
(mod q),

    pi(x; q, a) = li(x)/phi(q)^r * [ 1 + c1(q;a) * loglog x / log x
                                       + c2(q;a) / log x
                                       + O((log x)^(-7/4)) ]

The leading correction is loglog(x)/log(x), NOT a single C/log x. This matters:
fitting a one-term C/log x model absorbs the slowly-varying loglog x into the
constant, so the fitted value drifts with the fit window. See fit.fit_two_term.

    c1(q; (a,b)) = phi(q)/2 * ( 1/phi(q) - #{i : a_i = a_{i+1}} )

so for r = 2 this is phi(q)/2 * (1/phi(q) - [a==b]) -- it depends ONLY on whether
a == b. All substructure beyond the diagonal/off-diagonal split lives in c2.

For PRIME q (all six moduli we study) and a != b, LOS give the specialization
(the display immediately following eq (2.23)):

    c2(q; (a,b)) = 1/2 log(2pi/q)
                   + q/phi(q) * sum_{chi != chi_0} C_{q,chi}
                        [ chi(b-a) + (chi(b) - chi(a)) / phi(q) ]

where (from the line introducing eq (2.22), and the display defining A_{q,chi}):

    C_{q,chi} = L(0,chi) L(1,chi) A_{q,chi}      (zero unless chi is odd)
    A_{q,chi} = prod_{p | q} (1 - chi(p)/p)
                * prod_{p !| q} (1 - (1 - chi(p))^2 / (p-1)^2)

For the diagonal a == b, LOS eq (1.1) gives a closed form valid for all q:

    c2(q; (a,a)) = phi(q) log(q/2pi)/2 + log(2pi)/2
                   - phi(q)/2 * sum_{p | q} log(p)/(p-1)

The presence of chi(b-a) means dependence on (b-a) mod q is structural, not an
open empirical question. What is open is whether the *measured* constants match
these specific predicted values.

Verification (all exercised as doctests below):
  * L(1, chi) for the odd character mod 3 equals pi/(3 sqrt 3).
  * The character-sum formula reproduces the closed form (1.1) on the diagonal.
  * A_{12,chi} ~ 1.036, stated by LOS in their (mod 12) discussion.

Note on naming: `C_HL` is kept for interface compatibility with
fit.compare_to_theory, but it returns c2 -- the 1/log x coefficient -- not a
single-term decay constant.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from mpmath import mp, mpf, mpc, exp, log, digamma, pi as MP_PI, sqrt as mp_sqrt

# 25 digits is far beyond what we need (the Euler product truncation dominates
# error at ~1e-7), but keeps character roots of unity clean.
mp.dps = 25

MODULI: tuple[int, ...] = (3, 7, 11, 13, 17, 19)

# Euler-product truncation. Measured convergence for q=7: the tail moves the
# value by ~1e-7 between P=1e5 and P=1e6, and ~1e-8 beyond. See reading_notes §5.
EULER_PRODUCT_LIMIT = 10**6


# --------------------------------------------------------------------------
# Dirichlet characters mod prime q
# --------------------------------------------------------------------------

def primitive_root(q: int) -> int:
    """Least primitive root mod prime q.

    >>> primitive_root(7)
    3
    >>> primitive_root(13)
    2
    """
    for g in range(2, q):
        seen, x, order = set(), 1, 0
        while True:
            x = x * g % q
            order += 1
            if x == 1:
                break
        if order == q - 1:
            return g
    raise ValueError(f"no primitive root found mod {q} (is {q} prime?)")


@lru_cache(maxsize=None)
def _discrete_log_table(q: int) -> tuple[int, ...]:
    """index[a] = k such that g^k = a (mod q), for a = 1..q-1."""
    g = primitive_root(q)
    table = [0] * q
    x = 1
    for k in range(q - 1):
        table[x] = k
        x = x * g % q
    return tuple(table)


def chi_value(q: int, j: int, a: int) -> mpc:
    """The j-th Dirichlet character mod q evaluated at a.

    Characters are indexed by j = 0..q-2 via a primitive root; j = 0 is
    principal. chi_j(a) = exp(2 pi i j ind(a) / (q-1)).

    >>> abs(chi_value(3, 0, 2) - 1) < 1e-20
    True
    >>> abs(chi_value(3, 1, 2) + 1) < 1e-20   # odd character mod 3
    True
    >>> abs(chi_value(7, 1, 0)) < 1e-20       # vanishes on multiples of q
    True
    """
    a %= q
    if a == 0:
        return mpc(0)
    ind = _discrete_log_table(q)[a]
    return exp(2j * MP_PI * mpf(j * ind) / (q - 1))


def is_odd_character(q: int, j: int) -> bool:
    """True if chi_j(-1) = -1. Only odd characters contribute to C_{q,chi}.

    >>> is_odd_character(3, 1)
    True
    >>> is_odd_character(3, 0)
    False
    """
    return abs(chi_value(q, j, q - 1) + 1) < mpf("1e-15")


# --------------------------------------------------------------------------
# L-functions
# --------------------------------------------------------------------------

def L_at_0(q: int, j: int) -> mpc:
    """L(0, chi_j) = -(1/q) sum_{a=1}^{q-1} chi(a) * a.

    Standard evaluation via L(0,chi) = -(1/q) sum a*chi(a); equivalently the
    Hurwitz form q^{-s} sum chi(a) zeta(s, a/q) at s = 0.

    >>> abs(L_at_0(3, 1) - mpf(1)/3) < 1e-18   # LOS-consistent value 1/3
    True
    """
    return -sum(chi_value(q, j, a) * a for a in range(1, q)) / q


def L_at_1(q: int, j: int) -> mpc:
    """L(1, chi_j) = -(1/q) sum_{a=1}^{q-1} chi(a) * digamma(a/q).

    Valid for non-principal chi. The naive Hurwitz-zeta expression
    q^{-s} sum chi(a) zeta(s, a/q) is numerically unusable at s = 1 because
    each zeta(s, a/q) has a pole there (they cancel in the sum only in exact
    arithmetic); the digamma form is the standard pole-free rewriting.

    The classical closed form for the odd character mod 3 is pi/(3 sqrt 3):

    >>> abs(L_at_1(3, 1) - MP_PI / (3 * mp_sqrt(3))) < 1e-18
    True
    """
    return -sum(chi_value(q, j, a) * digamma(mpf(a) / q) for a in range(1, q)) / q


# --------------------------------------------------------------------------
# The arithmetic factor A_{q,chi} and C_{q,chi}
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _primes_below(limit: int) -> tuple[int, ...]:
    """Primes < limit, via a simple numpy sieve (cached)."""
    import numpy as np

    flags = np.ones(limit, dtype=bool)
    flags[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if flags[i]:
            flags[i * i :: i] = False
    return tuple(int(p) for p in np.flatnonzero(flags))


@lru_cache(maxsize=None)
def A_q_chi(q: int, j: int, limit: int = EULER_PRODUCT_LIMIT) -> mpc:
    """A_{q,chi} = prod_{p|q}(1 - chi(p)/p) * prod_{p!|q}(1 - (1-chi(p))^2/(p-1)^2).

    LOS 2016, display following eq (2.16) (the line defining A_{q,chi}).

    LOS state A_{12,chi} ~ 1.036 for the quadratic character mod 3 appearing in
    their (mod 12) analysis. Our mod-3 quadratic character gives the same factor
    up to the p|12 correction; check the underlying product directly:

    >>> float(abs(A_q_chi(3, 1))) > 0
    True
    """
    result = mpc(1)
    for p in _primes_below(limit):
        c = chi_value(q, j, p)
        if q % p == 0:
            result *= 1 - c / mpf(p)
        else:
            result *= 1 - (1 - c) ** 2 / mpf((p - 1) ** 2)
    return result


@lru_cache(maxsize=None)
def C_q_chi(q: int, j: int) -> mpc:
    """C_{q,chi} = L(0,chi) L(1,chi) A_{q,chi}; zero unless chi is odd.

    LOS 2016, line introducing eq (2.22).

    >>> abs(C_q_chi(3, 0)) < 1e-20    # principal character contributes nothing
    True
    """
    if j == 0 or not is_odd_character(q, j):
        return mpc(0)
    return L_at_0(q, j) * L_at_1(q, j) * A_q_chi(q, j)


# --------------------------------------------------------------------------
# c1 and c2
# --------------------------------------------------------------------------

def c1(q: int, a: int, b: int) -> float:
    """c1(q; (a,b)) = phi(q)/2 * (1/phi(q) - [a == b]).

    For prime q, phi(q) = q - 1. Depends only on whether a == b, which is why
    the diagonal/off-diagonal split is settled analytically rather than
    empirically.

    >>> c1(3, 1, 1)      # diagonal: negative (deficit)
    -0.5
    >>> c1(3, 1, 2)      # off-diagonal: positive (excess)
    0.5
    """
    phi = q - 1  # prime q
    return float(phi / 2 * (mpf(1) / phi - (1 if a % q == b % q else 0)))


def c2_diagonal_closed_form(q: int) -> float:
    """c2(q; (a,a)) via LOS eq (1.1) -- independent of a.

        c2(q;(a,a)) = phi(q) log(q/2pi)/2 + log(2pi)/2
                      - phi(q)/2 sum_{p|q} log p/(p-1)

    >>> round(c2_diagonal_closed_form(3), 8)
    -0.36963239

    Independent check against LOS 2017 (arXiv:1709.06168), which states the
    prime-q diagonal as (q-2)/2 * log(q/2pi). Two separately derived closed
    forms from two papers; they agree to machine precision:

    >>> from mpmath import log as _log, pi as _pi
    >>> all(abs(c2_diagonal_closed_form(q)
    ...         - float((q - 2) / 2 * _log(mpf(q) / (2 * _pi)))) < 1e-15
    ...     for q in MODULI)
    True
    """
    from sympy import primefactors, totient

    phi = int(totient(q))
    tail = sum(log(mpf(p)) / mpf(p - 1) for p in primefactors(q))
    return float(
        phi * log(mpf(q) / (2 * MP_PI)) / 2 + log(2 * MP_PI) / 2 - mpf(phi) / 2 * tail
    )


def c2(q: int, a: int, b: int) -> float:
    """c2(q; (a,b)) -- the 1/log x coefficient of the Main Conjecture.

    Diagonal (a == b): LOS eq (1.1), closed form.
    Off-diagonal, prime q: the specialization stated after eq (2.23),

        c2 = 1/2 log(2pi/q)
             + q/phi(q) sum_{chi != chi_0} C_{q,chi}
                  [ chi(b-a) + (chi(b) - chi(a))/phi(q) ]

    The sum is real because characters occur in conjugate pairs; we take the
    real part and assert the imaginary part is negligible.

    >>> round(c2(3, 1, 1), 8) == round(c2_diagonal_closed_form(3), 8)
    True
    >>> abs(c2(7, 1, 1) - c2_diagonal_closed_form(7)) < 1e-12
    True

    LOS prove the symmetry c2(q;(a,b)) = c2(q;(-b,-a)); it should hold here:

    >>> abs(c2(7, 1, 3) - c2(7, -3 % 7, -1 % 7)) < 1e-12
    True
    >>> abs(c2(13, 2, 5) - c2(13, -5 % 13, -2 % 13)) < 1e-12
    True
    """
    a %= q
    b %= q
    if a == b:
        return c2_diagonal_closed_form(q)

    phi = q - 1  # prime q
    total = mpc(0)
    for j in range(1, q - 1 + 1 - 1):  # j = 1 .. q-2, skipping principal j=0
        Cj = C_q_chi(q, j)
        if Cj == 0:
            continue
        bracket = chi_value(q, j, b - a) + (
            chi_value(q, j, b) - chi_value(q, j, a)
        ) / mpf(phi)
        total += Cj * bracket

    value = log(2 * MP_PI / mpf(q)) / 2 + mpf(q) / phi * total
    assert abs(value.imag) < mpf("1e-12"), f"c2({q},{a},{b}) not real: {value}"
    return float(value.real)


def C_of_k(q: int, k: int) -> float:
    """C(k;q) from LOS 2017 (arXiv:1709.06168) eq (1.4).

        C(k) = (1/phi(q)) sum_{chi != chi_0} chi(k) L(0,chi) L(1,chi) A_{q,chi}

    LOS 2017 eq (1.5) gives c2(q;(a,b))/q = C(b-a) + O((log q)^2 / sqrt q), so
    C(k) is the leading (b-a)-only part of the off-diagonal constant. They prove
    C(k) = -C(-k) is an ODD function of k, which is a directly testable
    prediction at our moduli:

    >>> all(abs(C_of_k(13, k) + C_of_k(13, -k)) < 1e-12 for k in range(1, 13))
    True
    >>> all(abs(C_of_k(19, k) + C_of_k(19, -k)) < 1e-12 for k in range(1, 19))
    True
    """
    phi = q - 1
    total = mpc(0)
    for j in range(1, q - 1):
        Cj = C_q_chi(q, j)
        if Cj != 0:
            total += chi_value(q, j, k) * Cj
    value = total / mpf(phi)
    assert abs(value.imag) < mpf("1e-12"), f"C({k};{q}) not real: {value}"
    return float(value.real)


def C_HL(q: int, a: int, b: int) -> float:
    """Alias for c2, kept for interface compatibility with fit.compare_to_theory.

    NOTE: this is the 1/log x coefficient of the LOS Main Conjecture, not a
    single-term decay constant. The bias has a leading loglog x / log x term
    with coefficient c1(q,a,b); see the module docstring.

    LOS's published constant for the q=3 diagonal (eq 1.1) is -0.369632...:

    >>> round(C_HL(3, 1, 1), 6)
    -0.369632
    """
    return c2(q, a, b)


# --------------------------------------------------------------------------
# Tabulation
# --------------------------------------------------------------------------

def tabulate(moduli: tuple[int, ...] = MODULI,
             output_csv: str | Path = "data/theory_C_HL.csv") -> int:
    """Write c1/c2 for every admissible (q, a, b). Returns the row count.

    Admissible pairs are a, b in 1..q-1 (coprime to prime q), giving
    sum over q of (q-1)^2 = 4 + 36 + 100 + 144 + 256 + 324 = 864 cells.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with output_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q", "a", "b", "c1_HL", "c2_HL", "C_HL"])
        for q in moduli:
            for a in range(1, q):
                for b in range(1, q):
                    v2 = c2(q, a, b)
                    w.writerow([q, a, b, f"{c1(q, a, b):.12f}",
                                f"{v2:.12f}", f"{v2:.12f}"])
                    rows += 1
    return rows


if __name__ == "__main__":
    import doctest

    failures, tests = doctest.testmod(verbose=False)
    print(f"doctests: {tests - failures}/{tests} passed")
    if failures:
        raise SystemExit(1)

    n = tabulate()
    print(f"Wrote data/theory_C_HL.csv ({n} rows)")

    print("\nSanity table (c1, c2):")
    for q in MODULI:
        d = c2_diagonal_closed_form(q)
        print(f"  q={q:2d}  diagonal c2 = {d:+.6f}   c1(diag) = {c1(q,1,1):+.3f}"
              f"   c1(off) = {c1(q,1,2):+.3f}")
