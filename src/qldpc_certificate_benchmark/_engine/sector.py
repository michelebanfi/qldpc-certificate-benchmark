"""Sector data and the helpers that build / re-verify against it.

A :class:`Sector` bundles everything needed to decode and certify one CSS
sector of a shot.  ``make_sector`` builds it from the (H, L, priors) matrices; ``_witness_eval`` / ``_verify_witness`` re-check externally
supplied solutions in plain NumPy so a bogus witness can never enter a proof.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.sparse import csr_matrix

from .engine import CertEngine


def channel_llrs(channel_probs):
    """Prior log-odds w_i = log((1-q_i)/q_i), clipped exactly as the engine does."""
    q = np.asarray(channel_probs, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.log((1.0 - q) / q)
    return np.clip(np.nan_to_num(w), -50.0, 50.0)


def pack_class(bits, powers):
    """Pack a 0/1 logical-effect vector into one integer (k <= 63)."""
    return int(np.dot(np.asarray(bits, dtype=np.int64) & 1, powers))


@dataclass
class Sector:
    """Everything needed to decode and certify one sector (Z or X) of a shot."""
    name: str
    Hdec: np.ndarray           # (m, n) int8 -- syndrome rows
    Hdec_csr: csr_matrix       # unused in the library (kept for the selftest)
    H_log: np.ndarray          # (k, n) int64 -- logical rows of the full matrix
    H_aug: np.ndarray          # (m+k, n) int8 -- [Hdec; H_log]
    channel_probs: np.ndarray  # (n,)
    w: np.ndarray              # (n,) channel-prior LLRs
    powers: np.ndarray         # (k,) powers of two: pack a logical effect
    engine: CertEngine = field(repr=False, default=None)
    masks: np.ndarray = field(repr=False, default=None)  # (k+1, m+k) uint8
    # External Tesseract decoders, attached per-worker AFTER fork (see
    # attach_tesseract_decoders); None when Tesseract is disabled/unavailable.
    tess_beam: object = field(repr=False, default=None)
    tess_simplex: object = field(repr=False, default=None)
    dem_kept_cols: np.ndarray = field(repr=False, default=None)  # DEM-col -> orig col
    tess_dem: object = field(repr=False, default=None)
    # column-adjacency graph (columns sharing a syndrome row), built lazily by
    # _ensure_cluster_graph for the certified cluster decomposition
    graph_indptr: np.ndarray = field(repr=False, default=None)
    graph_indices: np.ndarray = field(repr=False, default=None)
    # int64 CSR copies of Hdec / H_aug for the numpy re-verification checks,
    # built lazily by hdec_mod2 / haug_mod2.  A dense int64 matvec allocates an
    # m x n temporary per check (~260 MB, 16 ms on a d=11 surface-code DEM);
    # the sparse product is exact integer arithmetic, just ~400x cheaper.
    Hdec_sp: csr_matrix = field(repr=False, default=None)
    H_aug_sp: csr_matrix = field(repr=False, default=None)


def hdec_mod2(sector, x):
    """Hdec @ x mod 2 (int64) via a cached sparse copy of Hdec."""
    if sector.Hdec_sp is None:
        sector.Hdec_sp = csr_matrix(sector.Hdec, dtype=np.int64)
    return (sector.Hdec_sp @ np.asarray(x, dtype=np.int64)) % 2


def haug_mod2(sector, x):
    """H_aug @ x mod 2 (int64) via a cached sparse copy of H_aug."""
    if sector.H_aug_sp is None:
        sector.H_aug_sp = csr_matrix(sector.H_aug, dtype=np.int64)
    return (sector.H_aug_sp @ np.asarray(x, dtype=np.int64)) % 2


def make_sector(name, Hdec, H_full, first_logical_row, channel_probs, k,
                pool_cap=1_000_000, verbose=True):
    Hdec = np.ascontiguousarray(Hdec, dtype=np.int8)
    H_log = np.ascontiguousarray(H_full[first_logical_row:first_logical_row + k],
                                 dtype=np.int64)
    H_aug = np.ascontiguousarray(np.vstack([Hdec, H_log.astype(np.int8)]))
    channel_probs = np.asarray(channel_probs, dtype=np.float64)

    # Trivial-bucket columns (zero syndrome AND zero logical action) are
    # decision-irrelevant: they can join any solution of any class, shifting
    # every class optimum by the same amount.  Their folded probability can
    # exceed 0.5 (the builder sums mechanism probabilities), so their weight
    # is zeroed -- equivalent to projecting them out of the model.  A zero-
    # syndrome column WITH logical action would be a weight-1 undetectable
    # logical error: fatal, never a bucket.
    zero_syn = ~Hdec.any(axis=0)
    if (zero_syn & H_log.any(axis=0)).any():
        raise RuntimeError(f"[{name}] zero-syndrome column with nonzero "
                           f"logical action: distance-1 model catastrophe")
    constrained = ~zero_syn | H_log.any(axis=0)
    assert channel_probs[constrained].max() < 0.5, (
        f"[{name}] a CONSTRAINED column has channel_probs >= 0.5: prior "
        f"weights would be non-positive and every certificate invalid")
    w = channel_llrs(channel_probs)
    n_bucket = int((~constrained).sum())
    if n_bucket and verbose:
        print(f"  [{name}] {n_bucket} trivial-bucket column(s) "
              f"{np.flatnonzero(~constrained).tolist()} excluded from the "
              f"cost model (zero syndrome, zero logical action)")
    w[~constrained] = 0.0
    m = Hdec.shape[0]
    masks = np.zeros((k + 1, m + k), dtype=np.uint8)
    masks[:, :m] = 1
    masks[0, m:] = 1                      # mask 0: TRUE search, all rows
    for i in range(k):
        masks[1 + i, m + i] = 1           # mask 1+i: wrong-bit-i search
    return Sector(
        name=name, Hdec=Hdec,
        Hdec_csr=None,
        H_log=H_log, H_aug=H_aug, channel_probs=channel_probs, w=w,
        powers=1 << np.arange(k, dtype=np.int64),
        engine=CertEngine(H_aug, m, w, pool_cap=pool_cap), masks=masks)


def _verify_witness(sector, b_full, row_active, support, value, context):
    """Independent numpy re-check of an exact-min certificate witness."""
    x = np.zeros(sector.H_aug.shape[1], dtype=np.int64)
    x[support] = 1
    lhs = haug_mod2(sector, x)
    act = row_active.astype(bool)
    assert np.array_equal(lhs[act], b_full.astype(np.int64)[act]), \
        f"{sector.name}/{context}: witness violates its system"
    assert abs(float(sector.w @ x) - value) < 1e-7, \
        f"{sector.name}/{context}: witness cost mismatch"


def _witness_eval(sector, s, x, verify=True):
    """(cost, packed_class) for a full 0/1 vector x that must satisfy Hdec x = s.

    Used to fold externally-found solutions (decoder outputs, ILP/LP witnesses)
    into the certificate: the syndrome match is re-verified in numpy so a bogus
    witness can never enter a proof.
    """
    x = np.asarray(x, dtype=np.int64)
    if verify:
        assert np.array_equal(hdec_mod2(sector, x),
                              np.asarray(s, dtype=np.int64)), \
            f"{sector.name}: witness violates the syndrome"
    cost = float(sector.w @ x)
    cls = pack_class(sector.H_log @ x % 2, sector.powers)
    return cost, cls
