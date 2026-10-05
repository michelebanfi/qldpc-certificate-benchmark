"""Exact sampling conditioned on the fault count.

``K`` = number of fired mechanisms of a shot.  Under independent priors it is
Poisson-binomial.  For a band ``S = [lo, hi]``, sampling from the prior law
*conditioned on* ``K in S`` makes every shot fail ML with the same probability
``theta_S / P(S)``, where ``theta_S = P(ML fails, K in S)``.  The certified
failure count of a fixed number of such shots is therefore exactly binomial,
and ``P(S) * ClopperPearsonLower(F, n)`` is a valid lower confidence bound on
``theta_S <= theta``.  Rare ML failures concentrate at fault counts far above
the mean, so a band there needs orders of magnitude fewer shots than plain
sampling to see (and certify) them.

Both functions use the backward table ``T[j, c] = P(exactly c of mechanisms
j..n-1 fire)`` for ``c <= hi``, with per-row rescaling so deep tails never
underflow: the band mass is exact up to float rounding, and a shot is drawn
exactly -- first ``K`` from the truncated pmf, then each mechanism in turn from
its conditional firing probability given the count still to place.
"""

import math

import numpy as np
from numba import njit

__all__ = ["fault_count_pmf", "sample_fault_band"]


@njit(cache=True)
def _tail_table(p, hi):
    n = p.size
    T = np.zeros((n + 1, hi + 1))
    logscale = np.zeros(n + 1)
    T[n, 0] = 1.0
    for j in range(n - 1, -1, -1):
        q, pj = 1.0 - p[j], p[j]
        T[j, 0] = q * T[j + 1, 0]
        mx = T[j, 0]
        for c in range(1, hi + 1):
            v = q * T[j + 1, c] + pj * T[j + 1, c - 1]
            T[j, c] = v
            if v > mx:
                mx = v
        logscale[j] = logscale[j + 1]
        if 0.0 < mx < 1e-150:
            for c in range(hi + 1):
                T[j, c] /= mx
            logscale[j] += math.log(mx)
    return T, logscale


@njit(cache=True)
def _sample_band_core(p, T, logscale, lo, hi, shots, seed):
    np.random.seed(seed)
    n = p.size
    cum = np.cumsum(T[0, lo:hi + 1])
    counts = np.empty(shots, dtype=np.int64)
    for t in range(shots):                    # all counts first: sizes the output
        u = np.random.random() * cum[-1]
        k = lo + int(np.searchsorted(cum, u, side="right"))
        counts[t] = k if k <= hi else hi
    ptr = np.zeros(shots + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts)
    idx = np.empty(ptr[-1], dtype=np.int64)
    pos = 0
    for t in range(shots):
        c = counts[t]
        for j in range(n):
            if c == 0:
                break
            if c == n - j:                    # every remaining mechanism fires
                fire = True
            else:
                num = p[j] * T[j + 1, c - 1] * math.exp(logscale[j + 1] - logscale[j])
                fire = np.random.random() * T[j, c] < num
            if fire:
                idx[pos] = j
                pos += 1
                c -= 1
    return ptr, idx, counts


def fault_count_pmf(priors, hi):
    """``P(K = c)`` for ``c = 0..hi`` (float64 array of length hi + 1)."""
    p = np.ascontiguousarray(priors, dtype=np.float64)
    hi = int(hi)
    if hi < 0:
        raise ValueError("hi must be >= 0")
    T, logscale = _tail_table(p, hi)
    return T[0] * math.exp(logscale[0])


def sample_fault_band(priors, lo, hi, shots, seed):
    """Draw ``shots`` fired-mechanism sets from the priors conditioned on
    ``lo <= K <= hi``.  Returns CSR ``(indptr, indices)`` and the counts ``K``."""
    p = np.ascontiguousarray(priors, dtype=np.float64)
    lo, hi = int(lo), int(hi)
    if not 0 <= lo <= hi <= p.size:
        raise ValueError(f"need 0 <= lo <= hi <= {p.size}, got [{lo}, {hi}]")
    T, logscale = _tail_table(p, hi)
    if not T[0, lo:hi + 1].sum() > 0:
        raise ValueError(f"fault-count band [{lo}, {hi}] has zero probability")
    return _sample_band_core(p, T, logscale, lo, hi, int(shots), int(seed))
