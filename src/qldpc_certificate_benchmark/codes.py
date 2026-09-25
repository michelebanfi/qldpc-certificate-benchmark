"""Small code constructors for examples and tests (CSS codes).

Each constructor returns ``(Hx, Hz, Lx, Lz)`` as uint8 arrays.  To decode
X errors use ``DecodingProblem.from_code(Hz, Lz, p)``.  To decode Z errors use
``DecodingProblem.from_code(Hx, Lx, p)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["gf2_rank", "css_logicals", "toric_code", "bivariate_bicycle_code",
           "bb_72_12_6", "bb_144_12_12"]


def _rref(M):
    M = (np.asarray(M, dtype=np.uint8) & 1).copy()
    rows, cols = M.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.flatnonzero(M[r:, c])
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        hit = np.flatnonzero(M[:, c])
        hit = hit[hit != r]
        M[hit] ^= M[r]
        pivots.append(c)
        r += 1
    return M[:r], pivots


def gf2_rank(M) -> int:
    return len(_rref(M)[1])


def _nullspace(M):
    """Basis (rows) of {x : M x = 0 mod 2}."""
    R, piv = _rref(M)
    n = np.asarray(M).shape[1]
    free = [c for c in range(n) if c not in set(piv)]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, pc in enumerate(piv):
            basis[i, pc] = R[r, f]
    return basis


def css_logicals(Hx, Hz):
    """Logical operators of the CSS code ``(Hx, Hz)``.

    Returns ``(Lx, Lz)`` with ``k = n - rank Hx - rank Hz`` rows each.  Every
    ``Lz`` row lies in ker(Hx) outside rowspace(Hz), and every ``Lx`` row
    likewise with the roles swapped.
    """
    Hx = np.asarray(Hx, dtype=np.uint8) & 1
    Hz = np.asarray(Hz, dtype=np.uint8) & 1

    def _pick(ker_of, modulo):
        cand = _nullspace(ker_of)
        base = modulo.copy()
        r0 = gf2_rank(base)
        out = []
        for v in cand:
            trial = np.vstack([base, v])
            r = gf2_rank(trial)
            if r > r0:
                base, r0 = trial, r
                out.append(v)
        return np.array(out, dtype=np.uint8).reshape(len(out), Hx.shape[1])

    Lz = _pick(Hx, Hz)   # Z-type logicals: commute with X checks
    Lx = _pick(Hz, Hx)
    return Lx, Lz


def _cyclic_shift(n, s):
    return np.roll(np.eye(n, dtype=np.uint8), s, axis=1)


def bivariate_bicycle_code(l, m, A_terms, B_terms):
    """Bivariate-bicycle code (Bravyi et al., Nature 2024).

    ``A_terms``/``B_terms`` are lists of monomials, each ``("x", a)`` or
    ``("y", b)``, with ``x = S_l (x) I_m`` and ``y = I_l (x) S_m``.  Then
    ``Hx = [A | B]`` and ``Hz = [B^T | A^T]``.
    """
    x = np.kron(_cyclic_shift(l, 1), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), _cyclic_shift(m, 1))

    def poly(terms):
        M = np.zeros((l * m, l * m), dtype=np.uint8)
        for var, e in terms:
            base = x if var == "x" else y
            M ^= np.linalg.matrix_power(base.astype(np.int64), e).astype(np.uint8) & 1
        return M

    A, B = poly(A_terms), poly(B_terms)
    Hx = np.hstack([A, B]) & 1
    Hz = np.hstack([B.T, A.T]) & 1
    Lx, Lz = css_logicals(Hx, Hz)
    return Hx, Hz, Lx, Lz


def bb_72_12_6():
    """The [[72, 12, 6]] bivariate-bicycle code: A = x^3 + y + y^2, B = y^3 + x + x^2."""
    return bivariate_bicycle_code(6, 6, [("x", 3), ("y", 1), ("y", 2)],
                                  [("y", 3), ("x", 1), ("x", 2)])


def bb_144_12_12():
    """The [[144, 12, 12]] "gross" code: A = x^3 + y + y^2, B = y^3 + x + x^2 on l=12, m=6."""
    return bivariate_bicycle_code(12, 6, [("x", 3), ("y", 1), ("y", 2)],
                                  [("y", 3), ("x", 1), ("x", 2)])


def toric_code(L):
    """The [[2L^2, 2, L]] toric code (edges of an L x L periodic lattice)."""
    n = 2 * L * L
    idx_h = lambda i, j: (i % L) * L + (j % L)               # horizontal edges
    idx_v = lambda i, j: L * L + (i % L) * L + (j % L)       # vertical edges
    Hx = np.zeros((L * L, n), dtype=np.uint8)   # vertex (star) operators
    Hz = np.zeros((L * L, n), dtype=np.uint8)   # plaquette operators
    for i in range(L):
        for j in range(L):
            v = i * L + j
            for e in (idx_h(i, j), idx_h(i, j - 1), idx_v(i, j), idx_v(i - 1, j)):
                Hx[v, e] = 1
            for e in (idx_h(i, j), idx_h(i + 1, j), idx_v(i, j), idx_v(i, j + 1)):
                Hz[v, e] = 1
    Lx, Lz = css_logicals(Hx, Hz)
    return Hx, Hz, Lx, Lz
