"""Streaming prime enumeration. primesieve primary, numpy segmented sieve fallback.

Note on performance: iterating primes one at a time in Python is the bottleneck
for this project regardless of the backend, so the production counting path
(counters.run_counters with prime_stream=None) does not use this module. It
consumes numpy arrays from counters.segmented_primes instead, which is ~380x
faster end to end. This module remains for pi(x) anchors, tests, and any caller
that genuinely wants a scalar iterator.
"""

from __future__ import annotations
from typing import Iterator


def stream_primes(start: int = 2, stop: int = 10**8, chunk: int = 10**8) -> Iterator[int]:
    """Yield primes in [start, stop). O(chunk) memory."""
    try:
        import primesieve  # type: ignore
        yield from _via_primesieve(start, stop, chunk)
    except ImportError:
        yield from _via_segmented(start, stop, chunk)


def _via_segmented(start: int, stop: int, chunk: int) -> Iterator[int]:
    """Fallback: numpy segmented sieve (see counters.segmented_primes).

    Preferred over sympy: it is exact, has no upper-range caveat, and is roughly
    two orders of magnitude faster at 10^9.
    """
    from counters import segmented_primes

    for block in segmented_primes(start, stop, segment=min(chunk, 2 * 10**7)):
        yield from (int(p) for p in block)


def _via_primesieve(start: int, stop: int, chunk: int) -> Iterator[int]:
    import primesieve  # type: ignore
    lo = max(start, 2)
    while lo < stop:
        hi = min(lo + chunk, stop)
        for p in primesieve.primes(lo, hi):
            yield int(p)
        lo = hi


def _via_sympy(start: int, stop: int) -> Iterator[int]:
    from sympy import sieve  # type: ignore
    for p in sieve.primerange(max(start, 2), stop):
        yield int(p)


def count_primes_up_to(x: int) -> int:
    """pi(x). Exact.

    >>> count_primes_up_to(10**6)
    78498
    """
    try:
        import primesieve  # type: ignore
        return int(primesieve.count_primes(2, x + 1))
    except ImportError:
        from counters import segmented_primes
        return sum(len(block) for block in segmented_primes(2, x + 1))


if __name__ == "__main__":
    n = sum(1 for _ in stream_primes(2, 10**6))
    print(f"pi(10^6) = {n} (expected 78498)")
