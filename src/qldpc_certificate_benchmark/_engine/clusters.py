"""Certified cluster decomposition.

Before any search, partition the fired detectors into clusters that PROVABLY
cannot interact below the cap: a re-verified feasible dual ``phi`` of the
fractional covering LP charges every fired detector, so every solution costs
>= Phi = sum(phi); a connected error component bridging two clusters also pays
the reduced-weight shortest path between them, so groups farther apart than
``cap - Phi`` are independent for every below-cap solution.  Each cluster is then
decided by its own tiny unified search and the verdict composed (global min =
sum of cluster minima, class = XOR).  Any tie / budget-out / single-cluster shot
returns None so the caller falls back -- verdicts are identical to a
non-clustered run, only faster.
"""

import numpy as np
from scipy.sparse import csr_matrix

from .constants import (
    EPS_TIE, CLUSTER_SLACK, DEC_OK, V_SUCCESS, V_FAIL,
)
from .kernels import _cluster_labels_core
from .relaxations import _max_packing
from .sector import pack_class, _witness_eval

def _ensure_cluster_graph(sector):
    """Build (once) the column-adjacency CSR: columns adjacent iff they share
    a syndrome row.  Weight-0 bucket columns have no syndrome support and are
    isolated vertices, so they can never sit on a separating path."""
    if sector.graph_indptr is None:
        H = csr_matrix(sector.Hdec.astype(np.int32))
        A = H.T @ H
        A.setdiag(0)
        A.eliminate_zeros()
        A = A.tocsr()
        sector.graph_indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        sector.graph_indices = np.ascontiguousarray(A.indices, dtype=np.int32)


def _lp_dual_charges(sector, D, reduced=True):
    """Verified feasible dual charges phi for the fired detectors D.

    Any phi >= 0 with  sum_{d in rows(j) cap D} phi_d <= w_j  for every column
    j is admissible: each fired detector needs at least one flipping column in
    any solution and no column's weight is over-charged, so every solution
    costs >= Phi = sum(phi).  The LP (HiGHS) maximises Phi; feasibility is then
    re-verified in numpy and enforced by down-scaling, so a solver inaccuracy
    can never corrupt the separation test.  Returns (phi, wred) with the
    reduced weights wred = w - per-column charges >= 0 (None unless
    ``reduced``), or None on failure.
    """
    w = sector.w
    sub = sector.Hdec[D, :] != 0                    # |D| x n
    touch = np.flatnonzero(sub.any(axis=0))
    A_ub = sub[:, touch].T.astype(np.float64)       # one constraint per column
    x = _max_packing(A_ub, w[touch])
    if x is None:
        return None
    phi = np.maximum(x, 0.0)
    lhs = A_ub @ phi
    ratio = float(np.max(lhs / w[touch]))           # touch-columns have w > 0
    if ratio > 1.0:
        phi /= ratio * (1.0 + 1e-12)
        lhs = A_ub @ phi
    if not np.all(lhs <= w[touch]):                 # strict feasibility required
        phi /= 1.0 + 1e-9
        lhs = A_ub @ phi
        if not np.all(lhs <= w[touch]):
            return None
    if not reduced:
        return phi, None
    charges = sub.T.astype(np.float64) @ phi        # per-column total charge
    wred = np.maximum(w - charges, 0.0)             # >= -O(ulp) by feasibility;
    return phi, wred                                # clip absorbed by CLUSTER_SLACK


def _cluster_decompose(sector, s, cap, dual=None):
    """Partition the fired detectors into certified non-interacting clusters.

    Returns a list of >= 2 detector-index arrays such that NO solution with
    cost <= cap + CLUSTER_SLACK has a connected support component touching two
    different clusters (see the soundness note above _cluster_labels_core), or
    None when the shot does not split (or the LP/graph machinery fails --
    falling back is always safe).  `dual` may pass a precomputed
    _lp_dual_charges result to avoid re-solving the LP."""
    D = np.flatnonzero(np.asarray(s) != 0).astype(np.int64)
    if D.size < 2:
        return None
    if dual is None:
        dual = _lp_dual_charges(sector, D)
    if dual is None:
        return None
    phi, wred = dual
    budget = float(cap) - float(phi.sum()) + CLUSTER_SLACK
    if budget >= float(wred.sum()):
        return None                       # every pair merges: single cluster
    _ensure_cluster_graph(sector)
    # seed columns per fired detector (label = index into D)
    seed_lists = [np.flatnonzero(sector.Hdec[d]).astype(np.int32) for d in D]
    seed_ptr = np.zeros(D.size + 1, dtype=np.int32)
    seed_ptr[1:] = np.cumsum([len(t) for t in seed_lists]).astype(np.int32)
    seed_cols = np.concatenate(seed_lists).astype(np.int32)
    eng = sector.engine
    status, parent = _cluster_labels_core(
        sector.graph_indptr, sector.graph_indices,
        np.ascontiguousarray(wred, dtype=np.float64),
        seed_ptr, seed_cols, D.size, float(budget),
        eng.heap_B, eng.heap_id)
    if status != 0:
        return None
    groups = {}
    for t in range(D.size):
        r = int(parent[t])
        while int(parent[r]) != r:
            r = int(parent[r])
        groups.setdefault(r, []).append(int(D[t]))
    if len(groups) < 2:
        return None
    return [np.asarray(g, dtype=np.int64) for g in groups.values()]


def _cluster_verdict(sector, s, lam_bits, cap_true, node_budget, verify=True,
                     dual=None, row_phi=None, col_banned=None):
    """Decide this sector by certified cluster decomposition, or return None.

    Each cluster is decided by its own syndrome-only A* (exact minimum g_c,
    class, and an eps-cleared frontier).  The composed witness (XOR of the
    per-cluster minimizers) is the exact GLOBAL minimum at cost sum(g_c) --
    the minimizers are provably support-disjoint, since an overlap would XOR
    into a solution cheaper than the proven minimum -- and every solution in
    any other class costs > sum(g_c) + eps.  Verdict semantics are identical
    to the decomposed searches; anything inconclusive returns None."""
    clusters = _cluster_decompose(sector, s, cap_true, dual=dual)
    if clusters is None:
        return None
    eng = sector.engine
    k = len(sector.powers)
    lam0_pk = pack_class(lam_bits, sector.powers)
    total = 0.0
    cls_comp = 0
    dev_min = eng.wmin_all       # kernel floor: a zero-syndrome component
    supports = []                # costs >= wmin_all > eps
    nodes_tot = 0
    for K in clusters:
        s_c = np.zeros_like(np.asarray(s, dtype=np.int8))
        s_c[K] = 1
        (st, gmin, cg1, _mth, _mwh, _wcls, frontier, nds, supp,
         m2, _c2) = eng.decide(s_c, 0, cap_true, node_budget,
                               row_phi=row_phi, col_banned=col_banned)
        nodes_tot += nds
        if st != DEC_OK or not np.isfinite(gmin) or supp is None:
            return None          # cluster search inconclusive -> fall back
        if np.isfinite(m2):
            return None          # class-degenerate cluster min (tie) -> fall back
        dev_min = min(dev_min, frontier - gmin)
        total += gmin
        cls_comp ^= cg1
        supports.append(supp)
    if dev_min <= EPS_TIE:
        return None
    x = np.zeros(sector.H_aug.shape[1], dtype=np.int64)
    for supp in supports:
        x[supp] ^= 1
    if verify:
        cost_w, cls_w = _witness_eval(sector, s, x, verify=True)
        assert abs(cost_w - total) < 1e-7 and cls_w == cls_comp, \
            f"{sector.name}: composed cluster witness mismatch"
    winner_is_true = (cls_comp == lam0_pk)
    if not winner_is_true and cap_true <= total + EPS_TIE:
        # cap_true (a true-class UB) at the wrong-class global min would imply
        # an eps-tie the per-cluster searches must have seen; do not certify
        return None
    lb_dev = total + dev_min     # certified LB on every class != cls_comp
    statuses = np.full(k + 1, -1, dtype=np.int8)
    nodes_arr = np.zeros(k + 1, dtype=np.int64)
    nodes_arr[0] = nodes_tot
    out = {
        "statuses": statuses, "nodes": nodes_arr, "nodes_total": int(nodes_tot),
        "lb_wrong_lp": np.inf, "success_lp": False,
        "esc_margin": np.inf, "escalations": 0, "lp_screened": 0,
        "milp_calls": 0, "ilp_capped": 0,
        "unified": False, "clustered": True, "n_clusters": len(clusters),
        "banned_cols": 0,
    }
    if winner_is_true:
        out.update(verdict=V_SUCCESS, lb_true=total, ub_true=total,
                   lb_wrong=min(lb_dev, cap_true + CLUSTER_SLACK),
                   ub_wrong=np.inf, true_exact=True,
                   wrong_witness_class=-1, success_no_milp=True)
    else:
        out.update(verdict=V_FAIL, lb_true=min(lb_dev, cap_true),
                   ub_true=cap_true, lb_wrong=total, ub_wrong=total,
                   true_exact=False, wrong_witness_class=int(cls_comp),
                   success_no_milp=False)
    return out
