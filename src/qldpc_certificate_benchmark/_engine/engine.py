"""CertEngine: owns the per-sector scratch arrays and dispatches kernel runs.

A_full = [Hdec; H_log] is fixed per sector; a system is selected per call by an
active-row mask plus a target vector.  ``run`` executes one certified A* min
(``certified_min_core``); ``decide`` runs the unified single-coset decider
(``certified_decide_core``).
"""

import numpy as np

from .constants import EPS_TIE, MAX_DEPTH, ST_MIN, DEC_OK
from .kernels import certified_min_core, certified_decide_core

class CertEngine:
    """Owns the per-sector arrays for certified_min_core and dispatches runs.

    A_full = [Hdec; H_log] is fixed per sector; a system is selected per call
    by an active-row mask (all logical rows for the true-class search, exactly
    one for a wrong-bit search) plus the target vector b.
    """

    def __init__(self, A_full, m_syn, w, pool_cap=1_000_000):
        A_full = np.ascontiguousarray(A_full, dtype=np.int8)
        R, n = A_full.shape
        self.R, self.n, self.m_syn = R, n, int(m_syn)
        self.w = np.ascontiguousarray(w, dtype=np.float64)
        in_constraints = A_full.any(axis=0)
        assert not in_constraints.any() or self.w[in_constraints].min() > 0.0, (
            "non-positive prior weight on a constrained column: certificates "
            "would be invalid")

        row_supports = [np.flatnonzero(A_full[i]).astype(np.int32) for i in range(R)]
        self.row_ptr = np.zeros(R + 1, dtype=np.int32)
        self.row_ptr[1:] = np.cumsum([len(s) for s in row_supports]).astype(np.int32)
        self.row_idx = (np.concatenate(row_supports) if R else
                        np.zeros(0)).astype(np.int32)
        col_supports = [np.flatnonzero(A_full[:, j]).astype(np.int32) for j in range(n)]
        self.col_ptr = np.zeros(n + 1, dtype=np.int32)
        self.col_ptr[1:] = np.cumsum([len(s) for s in col_supports]).astype(np.int32)
        self.col_rows = (np.concatenate(col_supports) if n else
                         np.zeros(0)).astype(np.int32)
        self.row_minw = np.array(
            [self.w[s].min() if len(s) else np.inf for s in row_supports],
            dtype=np.float64)
        # constants of the cheap counting bound: a column toggles at most
        # cmax_syn syndrome rows, and any syndrome-fixing flip costs at least
        # wmin_syncols (min weight over columns hitting >= 1 syndrome row)
        syn_counts = A_full[:self.m_syn].astype(np.int64).sum(axis=0)
        self.cmax_syn = int(max(1, syn_counts.max()))
        hits_syn = syn_counts > 0
        self.wmin_syncols = float(self.w[hits_syn].min()) if hits_syn.any() \
            else np.inf

        # --- unified single-coset decider support --------------------------
        # k = number of logical rows (rows m_syn .. R-1 of A_full).  col_logmask[j]
        # packs which logical rows column j hits (bit i set <=> A_full[m+i, j]==1),
        # so a goal's logical class = XOR of col_logmask over its support columns.
        self.k = int(R - self.m_syn)
        col_logmask = np.zeros(n, dtype=np.int64)
        for i in range(self.k):
            col_logmask[A_full[self.m_syn + i].astype(bool)] |= np.int64(1) << i
        self.col_logmask = np.ascontiguousarray(col_logmask, dtype=np.int64)
        # syndrome-only active mask (logical rows inactive)
        self.mask_syn = np.zeros(R, dtype=np.uint8)
        self.mask_syn[:self.m_syn] = 1
        # wmin over ALL constrained columns: the per-added-column cost floor that
        # makes every prune-at-goal "invisible" superset exceed global_min + eps.
        # It MUST exceed EPS_TIE or the unified decider's frontier argument breaks.
        constrained_all = A_full.any(axis=0)
        self.wmin_all = float(self.w[constrained_all].min()) if constrained_all.any() \
            else np.inf
        assert self.wmin_all > EPS_TIE, (
            "min constrained weight <= EPS_TIE: unified decider frontier invalid")

        self.pool_parent = np.empty(pool_cap, dtype=np.int32)
        self.pool_brow = np.empty(pool_cap, dtype=np.int32)
        self.pool_bpos = np.empty(pool_cap, dtype=np.int32)
        self.pool_g = np.empty(pool_cap, dtype=np.float64)
        self.pool_hflag = np.empty(pool_cap, dtype=np.uint8)
        self.heap_B = np.empty(pool_cap, dtype=np.float64)
        self.heap_id = np.empty(pool_cap, dtype=np.int32)
        self.r = np.empty(R, dtype=np.int8)
        self.dec_stamp = np.zeros(n, dtype=np.int64)
        self.dec_val = np.zeros(n, dtype=np.uint8)
        self.mark_stamp = np.zeros(n, dtype=np.int64)
        self.path_row = np.empty(MAX_DEPTH, dtype=np.int32)
        self.path_pos = np.empty(MAX_DEPTH, dtype=np.int32)
        self.unsat = np.empty(R, dtype=np.int32)
        self.unsat_child = np.empty(R, dtype=np.int32)
        self.sol_out = np.empty(MAX_DEPTH, dtype=np.int32)
        self.epoch_box = np.zeros(2, dtype=np.int64)
        # neutral dual inputs (phi = 0, nothing banned) used when the caller
        # supplies none -- bit-identical to the pre-dual-bounds engine
        self.zero_phi = np.zeros(R, dtype=np.float64)
        self.zero_ban = np.zeros(n, dtype=np.uint8)

    def run(self, row_active, b_full, cap, node_budget,
            row_phi=None, col_banned=None):
        """One certified search. Returns (status, value, lb, nodes, support).

        row_phi: verified covering-dual charges per row (admissible heuristic);
        col_banned: columns proven irrelevant below the shot cap (excluded)."""
        status, value, lb, nodes, sol_len = certified_min_core(
            self.row_ptr, self.row_idx, self.col_ptr, self.col_rows,
            self.row_minw, self.m_syn, self.cmax_syn, self.wmin_syncols,
            self.zero_phi if row_phi is None else row_phi,
            self.zero_ban if col_banned is None else col_banned,
            np.ascontiguousarray(row_active, dtype=np.uint8),
            np.ascontiguousarray(b_full, dtype=np.int8),
            self.w, float(cap), EPS_TIE, int(node_budget),
            self.pool_parent, self.pool_brow, self.pool_bpos, self.pool_g,
            self.pool_hflag,
            self.heap_B, self.heap_id,
            self.r, self.dec_stamp, self.dec_val, self.mark_stamp,
            self.path_row, self.path_pos, self.unsat, self.unsat_child,
            self.sol_out, self.epoch_box)
        support = np.sort(self.sol_out[:sol_len].copy()) if status == ST_MIN else None
        return int(status), float(value), float(lb), int(nodes), support

    def decide(self, s, lam0_pk, cap, node_budget,
               row_phi=None, col_banned=None):
        """One unified syndrome-only search deciding nondeg-ML for this shot.

        Returns (status, global_min, class_g1, m_true_hat, m_wrong_hat,
                 wrong_class, frontier_lb, nodes, support_g1, m_second,
                 class_second).
        status DEC_OK: (class_g1 at global_min) is the exact ML winner; the loser
        class min is > global_min unless m_{loser}_hat <= global_min+eps (tie).
        status DEC_BUDGET: inconclusive; frontier_lb is a certified LB.
        support_g1 is the global-min witness (for numpy re-verification).
        m_second/class_second: cheapest popped goal in a class != class_g1
        (necessarily <= global_min + eps): finite iff the global minimum is
        class-degenerate within eps."""
        b_full = np.concatenate(
            [np.asarray(s, dtype=np.int8), np.zeros(self.k, dtype=np.int8)])
        (status, gmin, cg1, mth, mwh, wcls, frontier, nodes, g1_len,
         m2, c2) = certified_decide_core(
                self.row_ptr, self.row_idx, self.col_ptr, self.col_rows,
                self.row_minw, self.m_syn, self.cmax_syn, self.wmin_syncols,
                self.zero_phi if row_phi is None else row_phi,
                self.zero_ban if col_banned is None else col_banned,
                self.wmin_all, self.col_logmask, np.int64(lam0_pk),
                self.mask_syn, np.ascontiguousarray(b_full, dtype=np.int8),
                self.w, float(cap), EPS_TIE, int(node_budget),
                self.pool_parent, self.pool_brow, self.pool_bpos, self.pool_g,
                self.pool_hflag, self.heap_B, self.heap_id,
                self.r, self.dec_stamp, self.dec_val, self.mark_stamp,
                self.path_row, self.path_pos, self.unsat, self.unsat_child,
                self.sol_out, self.epoch_box)
        support = (np.sort(self.sol_out[:g1_len].copy())
                   if status == DEC_OK and gmin < np.inf else None)
        return (int(status), float(gmin), int(cg1), float(mth), float(mwh),
                int(wcls), float(frontier), int(nodes), support,
                float(m2), int(c2))
