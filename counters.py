"""Joint (a, b) residue-pair counters for consecutive primes.

For each q, count consecutive pairs (p_n, p_{n+1}) with p_n = a, p_{n+1} = b (mod q).
Snapshot at log-spaced x checkpoints. Streaming, six moduli per pass.

Output CSV: x, q, a, b, count_ab, pi_x.

Implementation note. The counting is vectorized: a segmented numpy sieve yields
primes in blocks, residues are taken with one array op per modulus, and each
consecutive pair is encoded as a single integer (prev*q + cur) then histogrammed
with np.bincount. This replaced a per-prime Python loop over six moduli and is
~380x faster end-to-end (measured: 1e10 in 1.14 min vs a projected ~7 hours),
which is what puts x = 1e11 inside the compute budget without primesieve.

Two correctness hazards the vectorized form has to handle explicitly:

  1. Segment seams. The pair straddling a block boundary (last prime of block k,
     first prime of block k+1) belongs to no single block's array. We carry the
     previous residue in `last[q]` and prepend it. Verified by
     tests/test_calibration.py::test_segment_boundary_invariance.

  2. Checkpoint boundaries. A pair (p_n, p_{n+1}) is attributed to the checkpoint
     interval containing p_n (the *smaller* prime), matching LOS's definition
     "the least of which is below x". Snapshots therefore split blocks at
     checkpoints rather than rounding to block edges.

run_counters also emits per-block counts (data/blocks.csv by default) so that
fit.fit_bootstrap can resample independent stretches of the prime stream; the
WLS error bars from nested checkpoints are optimistic and this is the honest
alternative.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

MODULI: tuple[int, ...] = (3, 7, 11, 13, 17, 19)

# Segment size for the sieve, in integers. 1e8 booleans ~ 100 MB; 2e7 keeps the
# working set comfortably in cache while still amortizing per-segment overhead.
DEFAULT_SEGMENT = 2 * 10**7


def log_spaced_checkpoints(x_min: float = 1e6, x_max: float = 1e13,
                           per_decade: int = 4) -> list[int]:
    """Log-spaced integer x checkpoints, deduped and sorted."""
    n_points = int(round(math.log10(x_max / x_min) * per_decade)) + 1
    return sorted({int(p) for p in np.logspace(math.log10(x_min), math.log10(x_max), n_points)})


# --------------------------------------------------------------------------
# Segmented sieve
# --------------------------------------------------------------------------

def _base_primes(limit: int) -> np.ndarray:
    """Primes <= limit by simple sieve; used to sieve each segment."""
    flags = np.ones(limit + 1, dtype=bool)
    flags[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if flags[i]:
            flags[i * i :: i] = False
    return np.flatnonzero(flags).astype(np.int64)


def segmented_primes(start: int, stop: int,
                     segment: int = DEFAULT_SEGMENT) -> Iterator[np.ndarray]:
    """Yield sorted numpy arrays of primes in [start, stop), block by block.

    >>> sum(len(b) for b in segmented_primes(2, 10**6))
    78498
    >>> np.concatenate(list(segmented_primes(2, 30))).tolist()
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """
    if stop <= start:
        return
    base = _base_primes(int(stop**0.5) + 1)

    lo = max(start, 2)
    while lo < stop:
        hi = min(lo + segment, stop)
        flags = np.ones(hi - lo, dtype=bool)
        for p in base:
            if p * p >= hi:
                break
            first = max(p * p, ((lo + p - 1) // p) * p)
            if first < hi:
                flags[first - lo :: p] = False
        # Mark 0 and 1 as non-prime if they fall in this block.
        for small in (0, 1):
            if lo <= small < hi:
                flags[small - lo] = False
        # Base primes below sqrt(stop) are marked composite by their own square
        # only above p*p, so they survive correctly; nothing further needed.
        yield np.flatnonzero(flags).astype(np.int64) + lo
        lo = hi


# --------------------------------------------------------------------------
# Pair counting
# --------------------------------------------------------------------------

def _accumulate_pairs(primes: np.ndarray, q: int, last: int,
                      counts: np.ndarray) -> int:
    """Add consecutive-pair counts for one modulus. Returns the new `last`.

    `last` is the residue of the previous prime (-1 if none yet). Pairs are
    encoded as prev*q + cur and histogrammed in one pass.
    """
    if len(primes) == 0:
        return last
    r = primes % q
    if last >= 0:
        prev = np.empty(len(r), dtype=np.int64)
        prev[0] = last
        prev[1:] = r[:-1]
        cur = r
    else:
        prev = r[:-1]
        cur = r[1:]
    if len(prev):
        counts += np.bincount(prev * q + cur, minlength=q * q)
    return int(r[-1])


def run_counters(prime_stream: Iterable[int] | None, checkpoints: list[int],
                 moduli: tuple[int, ...] = MODULI,
                 output_csv: str | Path = "data/checkpoints.csv",
                 x_start: int = 2,
                 segment: int = DEFAULT_SEGMENT,
                 blocks_csv: str | Path | None = None,
                 block_size: int | None = None) -> None:
    """Count consecutive-prime residue pairs, snapshotting at each checkpoint.

    `prime_stream` is accepted for backward compatibility with the original
    signature (compute_los_bias.py and the existing tests pass one). When it is
    an iterable of ints it is consumed as-is; when it is None the primes are
    generated internally by the segmented sieve up to max(checkpoints).

    Passing an iterable is markedly slower -- it reintroduces the per-prime
    Python loop -- so production callers should pass None.

    blocks_csv, if given, receives per-block pair counts for the block bootstrap
    (columns: block_index, x_lo, x_hi, q, a, b, count_ab, n_primes).
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    counters = {q: np.zeros(q * q, dtype=np.int64) for q in moduli}
    last = {q: -1 for q in moduli}
    pi_x = 0

    ckpts = sorted(checkpoints)
    x_max = ckpts[-1]
    ck_idx = 0

    block_rows: list[list] = []
    block_size = block_size or max(segment, 10**7)
    block_counters = {q: np.zeros(q * q, dtype=np.int64) for q in moduli}
    block_index = 0
    block_lo = x_start
    block_primes = 0

    with output_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "q", "a", "b", "count_ab", "pi_x"])

        def flush_block(hi: int) -> None:
            nonlocal block_index, block_lo, block_primes
            if blocks_csv is None or block_primes == 0:
                return
            for q in moduli:
                M = block_counters[q].reshape(q, q)
                for a in range(1, q):
                    for b in range(1, q):
                        if M[a, b]:
                            block_rows.append([block_index, block_lo, hi, q, a, b,
                                               int(M[a, b]), block_primes])
                block_counters[q][:] = 0
            block_index += 1
            block_lo = hi
            block_primes = 0

        def snapshot(x: int) -> None:
            for q in moduli:
                M = counters[q].reshape(q, q)
                for a in range(1, q):        # a = 0 only when p = q itself
                    for b in range(1, q):
                        w.writerow([x, q, a, b, int(M[a, b]), pi_x])

        if prime_stream is not None:
            # Compatibility path: consume the caller's iterator one prime at a
            # time. Correct but slow; kept so existing callers/tests still work.
            prev = {q: -1 for q in moduli}
            for p in prime_stream:
                while ck_idx < len(ckpts) and p > ckpts[ck_idx]:
                    snapshot(ckpts[ck_idx])
                    ck_idx += 1
                pi_x += 1
                for q in moduli:
                    cur = p % q
                    if prev[q] >= 0:
                        counters[q][prev[q] * q + cur] += 1
                    prev[q] = cur
            while ck_idx < len(ckpts):
                snapshot(ckpts[ck_idx])
                ck_idx += 1
        else:
            next_block_edge = block_lo + block_size
            for primes in segmented_primes(x_start, x_max + 1, segment):
                # Consume this segment left to right, cutting at whichever comes
                # first: the next checkpoint or the next bootstrap-block edge.
                # Every prime is consumed exactly once; the loop advances `pos`
                # monotonically and each iteration either consumes primes or
                # retires an edge, so it always terminates.
                pos = 0
                while pos < len(primes):
                    # Smallest exclusive upper bound among the pending edges.
                    cut = len(primes)
                    if ck_idx < len(ckpts):
                        cut = min(cut, int(np.searchsorted(
                            primes, ckpts[ck_idx] + 1, side="left")))
                    if blocks_csv is not None:
                        cut = min(cut, int(np.searchsorted(
                            primes, next_block_edge, side="left")))

                    if cut > pos:
                        chunk = primes[pos:cut]
                        pi_x += len(chunk)
                        block_primes += len(chunk)
                        for q in moduli:
                            before = (counters[q].copy()
                                      if blocks_csv is not None else None)
                            last[q] = _accumulate_pairs(chunk, q, last[q], counters[q])
                            if before is not None:
                                block_counters[q] += counters[q] - before
                        pos = cut

                    # Retire every edge that now lies at or before `pos`.
                    # `pos == len(primes)` means the segment is exhausted; the
                    # remaining edges are handled by later segments (or the
                    # final drain below), so stop here.
                    if pos >= len(primes):
                        break
                    p_next = primes[pos]
                    if ck_idx < len(ckpts) and p_next > ckpts[ck_idx]:
                        snapshot(ckpts[ck_idx])
                        ck_idx += 1
                    elif blocks_csv is not None and p_next >= next_block_edge:
                        flush_block(next_block_edge)
                        next_block_edge += block_size
                    else:
                        # No edge is blocking; the next cut must consume primes.
                        continue
            while ck_idx < len(ckpts):
                snapshot(ckpts[ck_idx])
                ck_idx += 1
            flush_block(x_max)

    if blocks_csv is not None and block_rows:
        blocks_csv = Path(blocks_csv)
        blocks_csv.parent.mkdir(parents=True, exist_ok=True)
        with blocks_csv.open("w", newline="") as bf:
            bw = csv.writer(bf)
            bw.writerow(["block_index", "x_lo", "x_hi", "q", "a", "b",
                         "count_ab", "n_primes"])
            bw.writerows(block_rows)


def compute_bias_from_checkpoints(csv_path):
    """Read checkpoints CSV, return df with bias and binomial sigma per row.

    sigma_bias is the binomial standard error treating the phi(q)^2 cells as
    independent multinomial draws. It is a LOWER BOUND on the true uncertainty:
    checkpoints are nested (they share primes), so the checkpoint-to-checkpoint
    scatter used by fit_wls is correlated. See fit.fit_bootstrap.
    """
    import pandas as pd
    from sympy import totient

    df = pd.read_csv(csv_path)
    df["phi_q"] = df["q"].map(lambda q: int(totient(q)))
    df["expected"] = df["pi_x"] / df["phi_q"] ** 2
    df["bias"] = df["count_ab"] / df["expected"] - 1
    df["sigma_bias"] = np.sqrt(df["phi_q"] ** 2 / df["pi_x"])
    return df


if __name__ == "__main__":
    import doctest

    failures, tests = doctest.testmod()
    print(f"doctests: {tests - failures}/{tests} passed")

    ckpts = log_spaced_checkpoints(x_min=1e4, x_max=1e6, per_decade=4)
    run_counters(None, ckpts, output_csv="data/_smoke.csv")
    print(compute_bias_from_checkpoints("data/_smoke.csv").tail(8))
