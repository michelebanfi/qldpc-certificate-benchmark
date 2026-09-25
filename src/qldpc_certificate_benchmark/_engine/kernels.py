"""Numba A* / cluster kernels for the certification engine.

IMPORTANT -- these kernels are compiled ``@njit(cache=True, nogil=True)``
DELIBERATELY WITHOUT ``fastmath``.  The ``eps`` (EPS_TIE) comparisons that make
every certificate sound rely on exact IEEE semantics.  Enabling fastmath would
let the optimiser reorder or relax those compares, which could SILENTLY FLIP
VERDICTS.  Do NOT add ``fastmath=`` to any decorator below.

The status literals (``ST_*``, ``DEC_*``) and ``MAX_DEPTH`` are imported from
``constants`` and baked into the compiled kernels; if you change them there,
clear the ``__pycache__/*.nbc`` files next to this module.
"""

import numpy as np
from numba import njit

from .constants import (
    MAX_DEPTH, ST_MIN, ST_PROVED, ST_BUDGET, DEC_OK, DEC_BUDGET,
)

# ---------------------------------------------------------------------------
# Certified A* engine
# ---------------------------------------------------------------------------
# certified_min_core finds min{ w.x : A x = b on the active rows } by
# priority-first branch-and-bound over supports.
#
#   Node      = partial assignment, encoded by its path: at each level one
#               unsatisfied row c was selected and one of its undecided
#               support columns j fixed to 1, with every undecided support
#               column of c preceding j fixed to 0.  Children partition the
#               node's feasible completions, so each solution lives under
#               exactly one path (S0 exclusions are inherited verbatim).
#   Prune     = a node whose residual is zero is a solution; its subtree is
#               not extended.  Valid for MIN-finding: any completion is a
#               superset with strictly larger cost (w > 0), and the path to a
#               support-minimal solution never crosses an intermediate zero
#               residual (the prefix would be a smaller same-system solution).
#   Bounds    = two admissible heuristics, evaluated lazily:
#               * h_quick (at push, O(|child unsat|)): satisfying u unsat
#                 syndrome rows needs >= ceil(u / cmax_syn) flips of columns
#                 that hit syndrome rows, each costing >= wmin_syncols; max'd
#                 with the static min support weight of any unsat logical row
#                 (one flip in its support is unavoidable).
#               * h_full (at pop): greedy packing of unsatisfied syndrome
#                 rows with pairwise disjoint UNDECIDED supports, each
#                 contributing its min UNDECIDED support weight (an unsat row
#                 with no undecided support proves the node infeasible);
#                 max'd with the same logical term.  Lazy A*: if g + h_full
#                 exceeds the current queue minimum the node is re-pushed
#                 with the tighter key instead of expanded (once), so the
#                 expensive bound is paid only by nodes that reach the top.
#               Both never overestimate any completion cost -> admissible.
#   A*        = with admissible keys the FIRST solution popped is the exact
#               minimum; at any stop the queue minimum (together with the
#               min bound among cap-dropped children and pop-dropped nodes)
#               is a certified lower bound on everything not yet enumerated.
#
# Certificate-critical invariants (do not "optimize" these away):
#   * goals are recognised at POP time, never at push;
#   * no push is ever silently dropped -- pool/heap exhaustion aborts the
#     whole search as ST_BUDGET with lb = min(B of the node being expanded,
#     current queue minimum);
#   * children dropped by the cap filter are tracked in min_dropped_B;
#   * compiled WITHOUT fastmath so the eps arguments hold in IEEE semantics.
#   MAX_DEPTH, ST_*, DEC_* are imported from constants (baked into the kernels).


@njit(inline="always")
def _hpush(heap_B, heap_id, heap_n, B, nid):
    i = heap_n
    heap_B[i] = B
    heap_id[i] = nid
    while i > 0:
        p = (i - 1) >> 1
        if heap_B[p] <= heap_B[i]:
            break
        heap_B[p], heap_B[i] = heap_B[i], heap_B[p]
        heap_id[p], heap_id[i] = heap_id[i], heap_id[p]
        i = p
    return heap_n + 1


@njit(inline="always")
def _hpop(heap_B, heap_id, heap_n):
    top_B = heap_B[0]
    top_id = heap_id[0]
    heap_n -= 1
    if heap_n > 0:
        heap_B[0] = heap_B[heap_n]
        heap_id[0] = heap_id[heap_n]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= heap_n:
                break
            sm = l
            r = l + 1
            if r < heap_n and heap_B[r] < heap_B[l]:
                sm = r
            if heap_B[i] <= heap_B[sm]:
                break
            heap_B[i], heap_B[sm] = heap_B[sm], heap_B[i]
            heap_id[i], heap_id[sm] = heap_id[sm], heap_id[i]
            i = sm
    return top_B, top_id, heap_n


@njit(inline="always")
def _h_quick(unsat_child, nc, m_syn, row_minw, cmax_syn, wmin_syncols, row_phi):
    """Cheap admissible bound from the unsatisfied-row list alone.

    Three under-estimates, max'd (max of admissible bounds is admissible):
    the syndrome counting bound, the static logical-row bound, and the
    covering-LP dual bound (any completion must touch every unsatisfied fired
    detector, and no column's weight can be over-charged, so the charges of
    the unsatisfied rows sum to a valid lower bound; row_phi is zero on rows
    that were not fired at the root, which only weakens the bound)."""
    nc_syn = 0
    h_log = 0.0
    h_phi = 0.0
    for t in range(nc):
        row = unsat_child[t]
        if row < m_syn:
            nc_syn += 1
            h_phi += row_phi[row]
        elif row_minw[row] > h_log:
            h_log = row_minw[row]
    h_cnt = ((nc_syn + cmax_syn - 1) // cmax_syn) * wmin_syncols
    h = h_cnt if h_cnt > h_log else h_log
    return h if h > h_phi else h_phi


@njit(cache=True, nogil=True)
def certified_min_core(row_ptr, row_idx, col_ptr, col_rows, row_minw, m_syn,
                       cmax_syn, wmin_syncols, row_phi, col_banned,
                       row_active, b, w, cap, eps, node_budget,
                       pool_parent, pool_brow, pool_bpos, pool_g, pool_hflag,
                       heap_B, heap_id,
                       r, dec_stamp, dec_val, mark_stamp,
                       path_row, path_pos, unsat, unsat_child, sol_out,
                       epoch_box):
    """Returns (status, value, lb, nodes_expanded, sol_len).

    status ST_MIN: value == lb == exact min; sol_out[:sol_len] = its support.
    status ST_PROVED: min > cap + eps certified; lb = best dropped bound
                      (+inf -> the active system is infeasible).
    status ST_BUDGET: lb = certified lower bound on the min; nothing claimed
                      beyond it (value = +inf).
    """
    R = b.shape[0]
    pool_cap = pool_parent.shape[0]
    INF = np.inf

    # --- root --------------------------------------------------------------
    n_unsat = 0
    for i in range(R):
        if row_active[i] == 1 and b[i] == 1:
            unsat[n_unsat] = i
            n_unsat += 1
    if n_unsat == 0:  # the zero pattern solves the system
        return ST_MIN, 0.0, 0.0, 1, 0

    B_root = _h_quick(unsat, n_unsat, m_syn, row_minw, cmax_syn, wmin_syncols,
                      row_phi)
    if B_root > cap + eps:
        return ST_PROVED, INF, B_root, 0, 0

    pool_parent[0] = -1
    pool_brow[0] = -1
    pool_bpos[0] = -1
    pool_g[0] = 0.0
    pool_hflag[0] = 0
    pool_n = 1
    heap_n = _hpush(heap_B, heap_id, 0, B_root, 0)
    min_dropped_B = INF
    nodes = 0

    # --- main loop ----------------------------------------------------------
    while heap_n > 0:
        if nodes >= node_budget:
            lb = heap_B[0]
            if min_dropped_B < lb:
                lb = min_dropped_B
            return ST_BUDGET, INF, lb, nodes, 0
        B_star, nid, heap_n = _hpop(heap_B, heap_id, heap_n)

        # reconstruct the node: path (root->nid), decided columns, residual
        depth = 0
        cur = nid
        overflow = False
        while pool_parent[cur] >= 0:
            if depth >= MAX_DEPTH:  # unreachable for real weights; safe abort
                overflow = True
                break
            path_row[depth] = pool_brow[cur]
            path_pos[depth] = pool_bpos[cur]
            depth += 1
            cur = pool_parent[cur]
        if overflow:
            lb = B_star
            if heap_n > 0 and heap_B[0] < lb:
                lb = heap_B[0]
            return ST_BUDGET, INF, lb, nodes, 0
        epoch_box[0] += 1
        dec_epoch = epoch_box[0]
        for i in range(R):
            r[i] = b[i] if row_active[i] == 1 else 0
        for lev in range(depth - 1, -1, -1):  # stored leaf->root; replay root->leaf
            c = path_row[lev]
            pos = path_pos[lev]
            base = row_ptr[c]
            for t in range(pos):  # undecided support columns before pos -> 0
                jj = row_idx[base + t]
                if dec_stamp[jj] != dec_epoch:
                    dec_stamp[jj] = dec_epoch
                    dec_val[jj] = 0
            j1 = row_idx[base + pos]  # the branched column -> 1
            dec_stamp[j1] = dec_epoch
            dec_val[j1] = 1
            for t in range(col_ptr[j1], col_ptr[j1 + 1]):
                row = col_rows[t]
                if row_active[row] == 1:
                    r[row] ^= 1
        g = pool_g[nid]

        # unsatisfied rows; goal test
        n_unsat = 0
        for i in range(R):
            if row_active[i] == 1 and r[i] == 1:
                unsat[n_unsat] = i
                n_unsat += 1
        if n_unsat == 0:  # GOAL -- first goal popped is the exact min
            for lev in range(depth):
                c = path_row[lev]
                sol_out[lev] = row_idx[row_ptr[c] + path_pos[lev]]
            return ST_MIN, g, g, nodes, depth

        # full bound h_full: greedy disjoint packing of unsatisfied syndrome
        # rows over UNDECIDED columns with dynamic per-row minima; the same
        # pass collects the branch-row choice (min undecided count, lowest
        # row index tiebreak) and detects infeasibility (cnt == 0)
        epoch_box[1] += 1
        mark_epoch = epoch_box[1]
        h_syn = 0.0
        h_log = 0.0
        h_phi = 0.0
        bc = -1
        bcnt = 1 << 30
        infeasible = False
        for t in range(n_unsat):
            row = unsat[t]
            if row >= m_syn:
                if row_minw[row] > h_log:
                    h_log = row_minw[row]
                continue
            h_phi += row_phi[row]
            cnt = 0
            dminw = INF
            disjoint = True
            for u in range(row_ptr[row], row_ptr[row + 1]):
                jj = row_idx[u]
                if dec_stamp[jj] == dec_epoch or col_banned[jj] == 1:
                    continue
                cnt += 1
                if w[jj] < dminw:
                    dminw = w[jj]
                if mark_stamp[jj] == mark_epoch:
                    disjoint = False
            if cnt == 0:
                infeasible = True
                break
            if cnt < bcnt:
                bcnt = cnt
                bc = row
            if disjoint:
                for u in range(row_ptr[row], row_ptr[row + 1]):
                    jj = row_idx[u]
                    if dec_stamp[jj] != dec_epoch and col_banned[jj] == 0:
                        mark_stamp[jj] = mark_epoch
                h_syn += dminw
        if infeasible:
            continue  # no completion can satisfy that row
        h_full = h_syn if h_syn > h_log else h_log
        if h_phi > h_full:
            h_full = h_phi
        B_full = g + h_full
        if B_full > cap + eps:  # tighter bound pushes the node past the cap
            if B_full < min_dropped_B:
                min_dropped_B = B_full
            continue
        if pool_hflag[nid] == 0 and heap_n > 0 and B_full > heap_B[0]:
            # lazy A*: tighter key no longer the best -- re-queue once
            pool_hflag[nid] = 1
            if heap_n >= heap_B.shape[0]:
                lb = B_star if B_star < heap_B[0] else heap_B[0]
                return ST_BUDGET, INF, lb, nodes, 0
            heap_n = _hpush(heap_B, heap_id, heap_n, B_full, nid)
            continue
        nodes += 1

        if bc < 0:  # no unsatisfied syndrome row: branch the lowest logical
            for t in range(n_unsat):
                if unsat[t] >= m_syn:
                    bc = unsat[t]
                    break

        # children sweep, ascending position within the branch row
        base = row_ptr[bc]
        rl = row_ptr[bc + 1] - base
        for pos in range(rl):
            j = row_idx[base + pos]
            if dec_stamp[j] == dec_epoch or col_banned[j] == 1:
                continue
            # child's unsatisfied rows = unsat XOR active rows of column j
            ia = 0
            ib = col_ptr[j]
            ib_end = col_ptr[j + 1]
            nc = 0
            while True:
                while ib < ib_end and row_active[col_rows[ib]] == 0:
                    ib += 1
                a_row = unsat[ia] if ia < n_unsat else 1 << 30
                b_row = col_rows[ib] if ib < ib_end else 1 << 30
                if a_row == 1 << 30 and b_row == 1 << 30:
                    break
                if a_row < b_row:
                    unsat_child[nc] = a_row
                    nc += 1
                    ia += 1
                elif b_row < a_row:
                    unsat_child[nc] = b_row
                    nc += 1
                    ib += 1
                else:
                    ia += 1
                    ib += 1
            h_child = _h_quick(unsat_child, nc, m_syn, row_minw,
                               cmax_syn, wmin_syncols, row_phi)
            B_child = g + w[j] + h_child
            if B_child > cap + eps:  # cap-prune (tracked, never silent)
                if B_child < min_dropped_B:
                    min_dropped_B = B_child
                continue
            if pool_n >= pool_cap or heap_n >= heap_B.shape[0]:
                lb = B_full  # bounds this node's remaining subtree
                if heap_n > 0 and heap_B[0] < lb:
                    lb = heap_B[0]
                return ST_BUDGET, INF, lb, nodes, 0
            pool_parent[pool_n] = nid
            pool_brow[pool_n] = bc
            pool_bpos[pool_n] = pos
            pool_g[pool_n] = g + w[j]
            pool_hflag[pool_n] = 0
            heap_n = _hpush(heap_B, heap_id, heap_n, B_child, pool_n)
            pool_n += 1

    # heap empty: everything either enumerated or cap-dropped
    return ST_PROVED, INF, min_dropped_B, nodes, 0


# ---------------------------------------------------------------------------
# Unified single-coset decider
# ---------------------------------------------------------------------------
# certified_decide_core runs ONE A* over the syndrome-only system  Hdec x = s
# (no logical rows constrained) and decides nondeg-ML directly, replacing the
# 1-true + k-wrong search decomposition with a single search.
#
#   * Goals (coset representatives) are popped in ascending cost.  Because the
#     bound is admissible, the FIRST goal popped is the exact GLOBAL minimum
#     g1 = min{ w.x : Hdec x = s } -- i.e. the ML winner -- and its logical
#     class (read from col_logmask, never constrained) is the ML decision.
#   * We keep popping, recording the min cost seen in the TRUE class and in the
#     WRONG classes, until the queue minimum clears global_min + eps.  At that
#     point every not-yet-popped VISIBLE goal costs > global_min + eps, and
#     every INVISIBLE goal -- one whose branch path crossed an already-popped
#     goal and was therefore pruned (prune-at-goal) -- is a strict superset of
#     a popped goal, hence costs >= global_min + wmin_all > global_min + eps.
#     So both classes' minima outside what we enumerated exceed the frontier.
#   * Verdict: winner = class(g1) at exact cost global_min.  If the OTHER
#     class's recorded min is <= global_min + eps it is a tie; otherwise the
#     winner wins strictly -> SUCCESS if the winner is the true class, FAIL if
#     it is a wrong class.  Budget/pool exhaustion before the frontier clears
#     returns DEC_BUDGET with a certified frontier lower bound (fall back to the
#     decomposed searches, which never lose coverage).
#
# The invisible-goal bound is why wmin_all (min weight over ALL constrained
# columns) must exceed eps; CertEngine asserts it.


@njit(cache=True, nogil=True)
def certified_decide_core(row_ptr, row_idx, col_ptr, col_rows, row_minw, m_syn,
                          cmax_syn, wmin_syncols, row_phi, col_banned,
                          wmin_all, col_logmask, lam0_pk,
                          row_active, b, w, cap, eps, node_budget,
                          pool_parent, pool_brow, pool_bpos, pool_g, pool_hflag,
                          heap_B, heap_id,
                          r, dec_stamp, dec_val, mark_stamp,
                          path_row, path_pos, unsat, unsat_child, sol_out,
                          epoch_box):
    """Returns (status, global_min, class_g1, m_true_hat, m_wrong_hat,
                wrong_class, frontier_lb, nodes, g1_len, m_second, class_second).

    row_active must select the syndrome rows only.  sol_out[:g1_len] is the
    support of the global-min witness (the ML decision), for numpy re-check.
    m_second/class_second record the cheapest popped goal whose class differs
    from the FIRST goal's class (the cluster path's tie detector); goals only
    pop while B <= global_min + eps, so a finite m_second means the global
    minimum is degenerate across classes within eps."""
    R = b.shape[0]
    pool_cap = pool_parent.shape[0]
    INF = np.inf

    global_min = INF
    class_g1 = np.int64(-1)
    m_true_hat = INF
    m_wrong_hat = INF
    wrong_class = np.int64(-1)
    m_second = INF
    class_second = np.int64(-1)
    g1_len = 0

    n_unsat = 0
    for i in range(R):
        if row_active[i] == 1 and b[i] == 1:
            unsat[n_unsat] = i
            n_unsat += 1
    if n_unsat == 0:  # zero syndrome: the empty support is the unique cost-0 rep
        if lam0_pk == 0:
            m_true_hat = 0.0
        else:
            m_wrong_hat = 0.0
            wrong_class = np.int64(0)
        return (DEC_OK, 0.0, np.int64(0), m_true_hat, m_wrong_hat,
                wrong_class, wmin_all, 1, 0, m_second, class_second)

    B_root = _h_quick(unsat, n_unsat, m_syn, row_minw, cmax_syn, wmin_syncols,
                      row_phi)
    if B_root > cap + eps:   # global min exceeds the decoder UB: cannot decide here
        return (DEC_BUDGET, INF, np.int64(-1), INF, INF, np.int64(-1),
                B_root, 0, 0, INF, np.int64(-1))

    pool_parent[0] = -1
    pool_brow[0] = -1
    pool_bpos[0] = -1
    pool_g[0] = 0.0
    pool_hflag[0] = 0
    pool_n = 1
    heap_n = _hpush(heap_B, heap_id, 0, B_root, 0)
    min_dropped_B = INF
    nodes = 0
    cap_eff = cap             # tightens to global_min once the first goal is found

    while heap_n > 0:
        if nodes >= node_budget:
            lb = heap_B[0]
            if min_dropped_B < lb:
                lb = min_dropped_B
            return (DEC_BUDGET, global_min, class_g1, m_true_hat, m_wrong_hat,
                    wrong_class, lb, nodes, g1_len, m_second, class_second)
        B_star, nid, heap_n = _hpop(heap_B, heap_id, heap_n)

        # frontier cleared: first goal found and every remaining node is above
        # global_min + eps -> the recorded per-class minima are conclusive
        if global_min < INF and B_star > cap_eff + eps:
            lb = B_star
            if min_dropped_B < lb:
                lb = min_dropped_B
            return (DEC_OK, global_min, class_g1, m_true_hat, m_wrong_hat,
                    wrong_class, lb, nodes, g1_len, m_second, class_second)

        depth = 0
        cur = nid
        overflow = False
        while pool_parent[cur] >= 0:
            if depth >= MAX_DEPTH:
                overflow = True
                break
            path_row[depth] = pool_brow[cur]
            path_pos[depth] = pool_bpos[cur]
            depth += 1
            cur = pool_parent[cur]
        if overflow:
            lb = B_star
            if heap_n > 0 and heap_B[0] < lb:
                lb = heap_B[0]
            return (DEC_BUDGET, global_min, class_g1, m_true_hat, m_wrong_hat,
                    wrong_class, lb, nodes, g1_len, m_second, class_second)
        epoch_box[0] += 1
        dec_epoch = epoch_box[0]
        for i in range(R):
            r[i] = b[i] if row_active[i] == 1 else 0
        for lev in range(depth - 1, -1, -1):
            c = path_row[lev]
            pos = path_pos[lev]
            base = row_ptr[c]
            for t in range(pos):
                jj = row_idx[base + t]
                if dec_stamp[jj] != dec_epoch:
                    dec_stamp[jj] = dec_epoch
                    dec_val[jj] = 0
            j1 = row_idx[base + pos]
            dec_stamp[j1] = dec_epoch
            dec_val[j1] = 1
            for t in range(col_ptr[j1], col_ptr[j1 + 1]):
                row = col_rows[t]
                if row_active[row] == 1:
                    r[row] ^= 1
        g = pool_g[nid]

        n_unsat = 0
        for i in range(R):
            if row_active[i] == 1 and r[i] == 1:
                unsat[n_unsat] = i
                n_unsat += 1
        if n_unsat == 0:  # GOAL: classify from col_logmask, record, do NOT expand
            cls = np.int64(0)
            for lev in range(depth):
                jcol = row_idx[row_ptr[path_row[lev]] + path_pos[lev]]
                cls ^= col_logmask[jcol]
            if global_min == INF:          # first goal popped == exact global min
                global_min = g
                class_g1 = cls
                cap_eff = g
                for lev in range(depth):
                    sol_out[lev] = row_idx[row_ptr[path_row[lev]] + path_pos[lev]]
                g1_len = depth
            elif cls != class_g1 and g < m_second:
                m_second = g               # class-degenerate global min (<= gmin+eps)
                class_second = cls
            if cls == lam0_pk:
                if g < m_true_hat:
                    m_true_hat = g
            else:
                if g < m_wrong_hat:
                    m_wrong_hat = g
                    wrong_class = cls
            continue

        epoch_box[1] += 1
        mark_epoch = epoch_box[1]
        h_syn = 0.0
        h_phi = 0.0
        bc = -1
        bcnt = 1 << 30
        infeasible = False
        for t in range(n_unsat):
            row = unsat[t]
            h_phi += row_phi[row]
            cnt = 0
            dminw = INF
            disjoint = True
            for u in range(row_ptr[row], row_ptr[row + 1]):
                jj = row_idx[u]
                if dec_stamp[jj] == dec_epoch or col_banned[jj] == 1:
                    continue
                cnt += 1
                if w[jj] < dminw:
                    dminw = w[jj]
                if mark_stamp[jj] == mark_epoch:
                    disjoint = False
            if cnt == 0:
                infeasible = True
                break
            if cnt < bcnt:
                bcnt = cnt
                bc = row
            if disjoint:
                for u in range(row_ptr[row], row_ptr[row + 1]):
                    jj = row_idx[u]
                    if dec_stamp[jj] != dec_epoch and col_banned[jj] == 0:
                        mark_stamp[jj] = mark_epoch
                h_syn += dminw
        if infeasible:
            continue
        if h_phi > h_syn:
            h_syn = h_phi
        B_full = g + h_syn
        if B_full > cap_eff + eps:
            if B_full < min_dropped_B:
                min_dropped_B = B_full
            continue
        if pool_hflag[nid] == 0 and heap_n > 0 and B_full > heap_B[0]:
            pool_hflag[nid] = 1
            if heap_n >= heap_B.shape[0]:
                lb = B_star if B_star < heap_B[0] else heap_B[0]
                return (DEC_BUDGET, global_min, class_g1, m_true_hat, m_wrong_hat,
                        wrong_class, lb, nodes, g1_len, m_second, class_second)
            heap_n = _hpush(heap_B, heap_id, heap_n, B_full, nid)
            continue
        nodes += 1

        base = row_ptr[bc]
        rl = row_ptr[bc + 1] - base
        for pos in range(rl):
            j = row_idx[base + pos]
            if dec_stamp[j] == dec_epoch or col_banned[j] == 1:
                continue
            ia = 0
            ib = col_ptr[j]
            ib_end = col_ptr[j + 1]
            nc = 0
            while True:
                while ib < ib_end and row_active[col_rows[ib]] == 0:
                    ib += 1
                a_row = unsat[ia] if ia < n_unsat else 1 << 30
                b_row = col_rows[ib] if ib < ib_end else 1 << 30
                if a_row == 1 << 30 and b_row == 1 << 30:
                    break
                if a_row < b_row:
                    unsat_child[nc] = a_row
                    nc += 1
                    ia += 1
                elif b_row < a_row:
                    unsat_child[nc] = b_row
                    nc += 1
                    ib += 1
                else:
                    ia += 1
                    ib += 1
            h_child = _h_quick(unsat_child, nc, m_syn, row_minw,
                               cmax_syn, wmin_syncols, row_phi)
            B_child = g + w[j] + h_child
            if B_child > cap_eff + eps:
                if B_child < min_dropped_B:
                    min_dropped_B = B_child
                continue
            if pool_n >= pool_cap or heap_n >= heap_B.shape[0]:
                lb = B_full
                if heap_n > 0 and heap_B[0] < lb:
                    lb = heap_B[0]
                return (DEC_BUDGET, global_min, class_g1, m_true_hat, m_wrong_hat,
                        wrong_class, lb, nodes, g1_len, m_second, class_second)
            pool_parent[pool_n] = nid
            pool_brow[pool_n] = bc
            pool_bpos[pool_n] = pos
            pool_g[pool_n] = g + w[j]
            pool_hflag[pool_n] = 0
            heap_n = _hpush(heap_B, heap_id, heap_n, B_child, pool_n)
            pool_n += 1

    # heap empty: every coset rep <= cap_eff enumerated; the rest exceed it
    if global_min == INF:                 # never found a goal (should not happen)
        return (DEC_BUDGET, INF, np.int64(-1), INF, INF, np.int64(-1),
                min_dropped_B, nodes, 0, INF, np.int64(-1))
    return (DEC_OK, global_min, class_g1, m_true_hat, m_wrong_hat,
            wrong_class, min_dropped_B, nodes, g1_len, m_second, class_second)


# ---------------------------------------------------------------------------
# Certified cluster decomposition
# ---------------------------------------------------------------------------
# The searches above pay for the WHOLE syndrome at once: the A* tree depth is
# the full solution support, so its size is multiplicative across independent
# error regions.  When the fired detectors provably split into non-interacting
# clusters, each cluster can be decided by its own tiny search and the shot
# verdict composed -- additive instead of multiplicative cost.
#
# Soundness of the split (no heuristic anywhere):
#   * phi-charging.  Give each fired detector d a charge phi_d >= 0 such that
#     for EVERY column j, the charges of the fired detectors j flips sum to
#     <= w_j (a feasible dual of the fractional covering LP; _lp_dual_charges
#     maximises it with HiGHS and re-verifies feasibility in numpy).  Every
#     syndrome solution x must touch every fired detector with >= 1 column, and
#     no column is over-charged, so cost(x) >= Phi := sum_d phi_d.
#   * reduced distance.  With reduced weights w'_j = w_j - (charges on j) >= 0,
#     suppose a solution x with cost <= C has one CONNECTED support component P
#     (columns adjacent iff they share a syndrome row) that touches fired
#     detectors of clusters a AND b.  P contains a column path between them:
#     splitting each path column's weight as w' + charges, and charging every
#     other column of x as above, gives cost(x) >= dist_phi(a,b) + Phi (each
#     fired detector's charge is counted at most once, every one is touched).
#   * separation test.  Therefore if dist_phi(a,b) + Phi > C, NO solution below
#     C has a component spanning a and b.  Clusters = connected components of
#     the merge graph {dist_phi(a,b) <= C - Phi + CLUSTER_SLACK}, computed by a
#     labelled multi-source Dijkstra (_cluster_labels_core) whose meeting rule
#     union(labels) fires exactly on the single-linkage threshold graph.  The
#     seed columns enter at distance 0 (their own weight is deliberately not
#     charged: an under-estimate of the true path cost, which errs toward
#     merging = sound), and CLUSTER_SLACK absorbs float rounding the same way.
#
# Every solution with cost <= C (= cap_true + slack) then XOR-decomposes into
# independent per-cluster solutions plus zero-syndrome (kernel) components, so
#   global min = sum_c g_c   (g_c = exact per-cluster minimum, first A* goal),
#   its class = XOR of the cluster classes (kernel-free: a kernel component
#   would add >= wmin_all > eps), and any solution in ANY OTHER class deviates
#   from the argmin composition in >= 1 cluster (or adds a kernel component),
#   costing >= sum_c g_c + D_min with
#   D_min = min( min_c(frontier_c - g_c), wmin_all ) > eps.
# _cluster_verdict turns this into the same SUCCESS/FAIL certificates as the
# decomposed searches; ties, budget-outs and single-cluster shots return None
# and fall back, so enabling USE_CLUSTERS can never change a verdict.

@njit(inline="always")
def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True, nogil=True)
def _cluster_labels_core(g_indptr, g_indices, wred, seed_ptr, seed_cols,
                         n_labels, budget, heap_B, heap_id):
    """Labelled multi-source Dijkstra with union-at-meeting.

    Label L seeds at columns seed_cols[seed_ptr[L]:seed_ptr[L+1]] (the columns
    flipping fired detector L) at distance wred[seed], so dist(v) = min over
    seeded paths of the sum of reduced weights of ALL path columns (both
    endpoints included) -- the exact node-weighted path cost.  Two labels are
    unioned whenever adjacent columns u, v satisfy dist(u) + dist(v) <= budget
    (or a shared column v itself satisfies wred[v] <= budget):
      * every union exhibits a real connecting walk of reduced cost <= budget
        (its deduplicated support is a connected set touching both detectors);
      * conversely, along any connecting path P with cost(P) <= budget, EVERY
        adjacent pair satisfies dist(x_i) + dist(x_{i+1}) <= prefix + suffix
        = cost(P) <= budget (dist is a min over all seeds), so every label
        boundary on P fires and a chain of unions connects the two ends.
    Hence the resulting components are exactly the single-linkage clusters of
    the true reduced distance at threshold `budget`.  Returns (status, parent):
    status 0 = ok, 1 = heap overflow (caller must treat the shot as a single
    cluster -- merging everything is always sound)."""
    n = wred.shape[0]
    INF = np.inf
    dist = np.full(n, INF)
    label = np.full(n, -1, dtype=np.int32)
    parent = np.empty(n_labels, dtype=np.int32)
    for i in range(n_labels):
        parent[i] = i
    heap_cap = heap_B.shape[0]
    heap_n = 0
    for L in range(n_labels):
        for t in range(seed_ptr[L], seed_ptr[L + 1]):
            v = seed_cols[t]
            if label[v] < 0:
                label[v] = L
                dist[v] = wred[v]
                if heap_n >= heap_cap:
                    return 1, parent
                heap_n = _hpush(heap_B, heap_id, heap_n, wred[v], v)
            elif wred[v] <= budget:
                # one column flips fired detectors of two labels: the column
                # alone is a connecting set of reduced cost wred[v]
                ra = _uf_find(parent, label[v])
                rb = _uf_find(parent, L)
                if ra != rb:
                    parent[ra] = rb
    while heap_n > 0:
        d, v, heap_n = _hpop(heap_B, heap_id, heap_n)
        if d > dist[v] + 1e-15:
            continue                      # stale (superseded) entry
        rv = _uf_find(parent, label[v])
        for t in range(g_indptr[v], g_indptr[v + 1]):
            u = g_indices[t]
            if label[u] >= 0:
                ru = _uf_find(parent, label[u])
                if ru != rv and d + dist[u] <= budget:
                    parent[ru] = rv       # meeting within budget
            nd = d + wred[u]
            if nd <= budget and nd < dist[u] - 1e-15:
                dist[u] = nd
                label[u] = label[v]
                if heap_n >= heap_cap:
                    return 1, parent
                heap_n = _hpush(heap_B, heap_id, heap_n, nd, u)
    return 0, parent
