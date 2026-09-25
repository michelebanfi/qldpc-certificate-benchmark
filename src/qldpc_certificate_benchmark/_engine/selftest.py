"""Brute-force validation of the certification engine.

``selftest()`` cross-checks every claimed exact minimum, cap proof, budget
bound, sector verdict, adaptive-LP / exact-ILP result, unified decision, dual
heuristic, column ban, and cluster decomposition against exhaustive enumeration
over all 2^n patterns on synthetic systems small enough to brute-force
(n <= 20).  ``reference_min`` is a pure-Python heapq mirror of the A* kernel used
as an independent oracle.
"""

import heapq
import time

import numpy as np

from ..config import DEFAULT_CFG
from .constants import (
    EPS_TIE, BAN_SLACK, ST_MIN, ST_PROVED, ST_BUDGET, DEC_OK,
    V_SUCCESS, V_FAIL, V_TIE, V_BUDGET,
)
from .engine import CertEngine
from .sector import Sector
from .relaxations import (_adaptive_lp, _ilp_exact,
                                           _ilp_any_wrong_below)
from .clusters import _cluster_verdict, _lp_dual_charges
from .verdicts import certify_sector

def reference_min(A_full, m_syn, row_active, b, w, cap, node_budget,
                  row_phi=None, col_banned=None):
    """Pure-python oracle of certified_min_core (same rules, heapq + dicts).

    Mirrors the kernel exactly -- lazy A* with the cheap counting bound at
    push and the dynamic disjoint packing at pop, plus the covering-dual
    bound and banned-column exclusion.  Intentionally slow.
    """
    R, n = A_full.shape
    if row_phi is None:
        row_phi = np.zeros(R)
    banned = (set() if col_banned is None
              else set(np.flatnonzero(col_banned).tolist()))
    rows = [np.flatnonzero(A_full[i]) for i in range(R)]
    cols = [np.flatnonzero(A_full[:, j] * row_active) for j in range(n)]
    minw = [w[s].min() if len(s) else np.inf for s in rows]
    syn_counts = A_full[:m_syn].astype(np.int64).sum(axis=0)
    cmax_syn = int(max(1, syn_counts.max()))
    hits = syn_counts > 0
    wmin_syncols = float(w[hits].min()) if hits.any() else np.inf

    def h_quick(unsat):
        nc_syn = sum(1 for row in unsat if row < m_syn)
        h_log = max((minw[row] for row in unsat if row >= m_syn), default=0.0)
        h_phi = sum(row_phi[row] for row in unsat if row < m_syn)
        h_cnt = -(-nc_syn // cmax_syn) * wmin_syncols if nc_syn else 0.0
        return max(h_cnt, h_log, h_phi)

    def h_full(unsat, S1, S0):
        marked, h_syn = set(), 0.0
        h_log = max((minw[row] for row in unsat if row >= m_syn), default=0.0)
        h_phi = sum(row_phi[row] for row in unsat if row < m_syn)
        for row in unsat:
            if row >= m_syn:
                continue
            und = [int(c) for c in rows[row]
                   if c not in S0 and c not in S1 and c not in banned]
            if not und:
                return np.inf  # infeasible node
            if not any(c in marked for c in und):
                marked.update(und)
                h_syn += min(w[c] for c in und)
        return max(h_syn, h_log, h_phi)

    root_unsat = tuple(i for i in range(R) if row_active[i] and b[i])
    if not root_unsat:
        return ST_MIN, 0.0, 0.0, 1, ()
    B_root = h_quick(root_unsat)
    if B_root > cap + EPS_TIE:
        return ST_PROVED, np.inf, B_root, 0, None
    heap = [(B_root, 0, 0.0, root_unsat, frozenset(), frozenset(), False)]
    counter, nodes, min_dropped = 1, 0, np.inf
    while heap:
        if nodes >= node_budget:
            return ST_BUDGET, np.inf, min(heap[0][0], min_dropped), nodes, None
        B_star, _, g, unsat, S1, S0, hdone = heapq.heappop(heap)
        if not unsat:
            return ST_MIN, g, g, nodes + 1, tuple(sorted(S1))
        hf = h_full(unsat, S1, S0)
        if np.isinf(hf):
            continue  # infeasible: some unsatisfied row has no undecided support
        B_full = g + hf
        if B_full > cap + EPS_TIE:
            min_dropped = min(min_dropped, B_full)
            continue
        if not hdone and heap and B_full > heap[0][0]:
            heapq.heappush(heap, (B_full, counter, g, unsat, S1, S0, True))
            counter += 1
            continue
        nodes += 1
        cand = [(len([c for c in rows[row]
                      if c not in S0 and c not in S1 and c not in banned]), row)
                for row in unsat if row < m_syn]
        if cand:
            _, bc = min(cand)
        else:
            bc = min(row for row in unsat if row >= m_syn)
        S0_run = set(S0)
        for j in rows[bc]:
            j = int(j)
            if j in S0_run or j in S1 or j in banned:
                continue
            child_unsat = tuple(sorted(set(unsat) ^ {int(rr) for rr in cols[j]}))
            B_child = g + w[j] + h_quick(child_unsat)
            if B_child > cap + EPS_TIE:
                min_dropped = min(min_dropped, B_child)
            else:
                heapq.heappush(heap, (B_child, counter, g + w[j], child_unsat,
                                      S1 | {j}, frozenset(S0_run), False))
                counter += 1
            S0_run.add(j)
    return ST_PROVED, np.inf, min_dropped, nodes, None


def _brute_tables(Hdec, H_log, w):
    """Cost, packed syndrome and packed class for every x in F_2^n (n <= 20)."""
    m, n = Hdec.shape
    k = H_log.shape[0]
    X = ((np.arange(1 << n, dtype=np.int64)[:, None] >>
          np.arange(n, dtype=np.int64)) & 1).astype(np.uint8)
    syn = (X @ Hdec.T.astype(np.uint8)) & 1
    lam = (X @ H_log.T.astype(np.uint8)) & 1
    syn_int = syn.astype(np.int64) @ (1 << np.arange(m, dtype=np.int64))
    lam_int = lam.astype(np.int64) @ (1 << np.arange(k, dtype=np.int64))
    cost = X.astype(np.float64) @ w
    return syn_int, lam_int, cost


def _brute_class_minima(syn_int, lam_int, cost, s_int, k):
    mask = syn_int == s_int
    minima = np.full(1 << k, np.inf)
    if mask.any():
        np.minimum.at(minima, lam_int[mask], cost[mask])
    return minima


def _selftest_instance(rng, tie_rich):
    m, n, k = 10, 20, 3
    Hdec = (rng.random((m, n)) < 0.35).astype(np.int8)
    H_log = (rng.random((k, n)) < 0.4).astype(np.int8)
    Hdec[7, :] = 0                      # planted all-zero syndrome row
    Hdec[:, 5] = 0                      # planted no-effect bucket column
    H_log[:, 5] = 0
    Hdec[:, 11] = Hdec[:, 3]            # planted duplicate-syndrome pair...
    H_log[:, 11] = (H_log[:, 3] + 1) % 2  # ...with different logical action
    for j in range(n):                  # no accidental extra zero columns
        if j != 5 and not Hdec[:, j].any():
            Hdec[0, j] = 1
    if tie_rich:
        w = rng.choice([0.7, 1.3, 2.1], size=n)
    else:
        w = rng.uniform(0.5, 3.0, size=n)
    return Hdec, H_log, np.asarray(w, dtype=np.float64)


def _selftest_cluster_instance(rng, bridge_w):
    """Two independent random blocks + two bridge columns of weight bridge_w.

    Cheap bridges make the certified separation test merge everything (the
    cluster path must then fall back); expensive bridges produce genuinely
    decomposable instances.  n = 20 keeps the 2^20 brute force feasible."""
    mA, nA, mB, nB, k = 7, 9, 7, 9, 3
    m, n = mA + mB, nA + nB + 2
    Hdec = np.zeros((m, n), dtype=np.int8)
    Hdec[:mA, :nA] = rng.random((mA, nA)) < 0.3
    Hdec[mA:, nA:nA + nB] = rng.random((mB, nB)) < 0.3
    for j in (n - 2, n - 1):              # bridge columns touch both blocks
        Hdec[int(rng.integers(0, mA)), j] = 1
        Hdec[mA + int(rng.integers(0, mB)), j] = 1
    for j in range(n):                    # no all-zero columns
        if not Hdec[:, j].any():
            Hdec[int(rng.integers(0, m)), j] = 1
    H_log = (rng.random((k, n)) < 0.35).astype(np.int8)
    w = rng.uniform(0.5, 3.0, size=n)
    w[n - 2:] = bridge_w
    return Hdec, H_log, np.asarray(w, dtype=np.float64)


def selftest():
    print("Self-test: certified A* engine + verdicts vs exhaustive brute force")
    rng = np.random.default_rng(11)
    n_exact = n_proved = n_budget_lb = n_verdicts = n_ref = 0
    n_lp = n_ilp = n_any_wrong = n_seed = n_uni = n_phi = 0
    t0 = time.time()
    for inst in range(6):
        tie_rich = inst % 2 == 0
        Hdec, H_log, w = _selftest_instance(rng, tie_rich)
        m, n = Hdec.shape
        k = H_log.shape[0]
        A_full = np.vstack([Hdec, H_log]).astype(np.int8)
        eng = CertEngine(A_full, m, w, pool_cap=1_500_000)
        masks = np.zeros((k + 1, m + k), dtype=np.uint8)
        masks[:, :m] = 1
        masks[0, m:] = 1
        for i in range(k):
            masks[1 + i, m + i] = 1
        syn_int, lam_int, cost = _brute_tables(Hdec, H_log, w)
        powers_m = 1 << np.arange(m, dtype=np.int64)

        def mk_sector():
            return Sector(name="T", Hdec=Hdec, Hdec_csr=None,
                          H_log=H_log.astype(np.int64), H_aug=A_full,
                          channel_probs=np.full(n, 0.01), w=w,
                          powers=1 << np.arange(k, dtype=np.int64),
                          engine=eng, masks=masks)

        for trial in range(20):
            x0 = (rng.random(n) < 0.18).astype(np.int64)
            s = (Hdec.astype(np.int64) @ x0) % 2
            lam0 = (H_log.astype(np.int64) @ x0) % 2
            lam0_int = int(lam0 @ (1 << np.arange(k, dtype=np.int64)))
            minima = _brute_class_minima(syn_int, lam_int, cost,
                                         int(s @ powers_m), k)
            m_true = minima[lam0_int]
            wrong = np.delete(minima, lam0_int)
            m_wrong = wrong.min()
            b_true = np.concatenate([s, lam0]).astype(np.int8)

            if trial == 0:
                # A single integer feasibility model covers every wrong
                # logical class.  Check both sides of the exact brute-force
                # threshold, including its returned feasible witness.
                if np.isfinite(m_wrong):
                    st_no, _ = _ilp_any_wrong_below(
                        Hdec, H_log, s, lam0, w, m_wrong - 0.1,
                        time_limit=30)
                    st_yes, wit = _ilp_any_wrong_below(
                        Hdec, H_log, s, lam0, w, m_wrong + 0.1,
                        time_limit=30)
                    assert st_no == ST_PROVED and st_yes == ST_MIN
                    x_wrong = np.zeros(n, dtype=np.int8)
                    x_wrong[wit] = 1
                    assert np.array_equal(Hdec @ x_wrong % 2, s)
                    assert not np.array_equal(H_log @ x_wrong % 2, lam0)
                    assert float(w @ x_wrong) <= m_wrong + 0.1 + 1e-7
                    n_any_wrong += 2

            # (1) exact minima: true system and each wrong-bit system
            st, val, lb, _, sol = eng.run(masks[0], b_true, m_true + 0.5,
                                          1_000_000)
            assert st == ST_MIN and abs(val - m_true) < 1e-9, \
                f"true min mismatch {val} vs {m_true}"
            _verify = np.zeros(n, dtype=np.int64)
            _verify[sol] = 1
            assert (Hdec.astype(np.int64) @ _verify % 2 == s).all()
            assert (H_log.astype(np.int64) @ _verify % 2 == lam0).all()
            sub_minima = []
            for i in range(k):
                sel = (np.arange(1 << k) >> i) & 1 != ((lam0_int >> i) & 1)
                brute_i = minima[sel].min()
                b_i = b_true.copy()
                b_i[m + i] ^= 1
                st_i, val_i, lb_i, _, _ = eng.run(masks[1 + i], b_i,
                                                  brute_i + 0.5, 1_000_000)
                if np.isinf(brute_i):
                    assert st_i == ST_PROVED and np.isinf(lb_i), \
                        "infeasible subsystem not proved"
                else:
                    assert st_i == ST_MIN and abs(val_i - brute_i) < 1e-9, \
                        f"wrong-bit-{i} min mismatch {val_i} vs {brute_i}"
                sub_minima.append(val_i if st_i == ST_MIN else np.inf)
                n_exact += 1
            # (2) the union of the k subsystems covers all wrong classes
            assert abs(min(sub_minima) - m_wrong) < 1e-9 or \
                (np.isinf(min(sub_minima)) and np.isinf(m_wrong)), \
                "union-of-wrong-bit minima != brute wrong-class min"
            n_exact += 1

            # (3) cap proofs: cap strictly below the min -> ST_PROVED with a
            #     valid lower bound (<= min, > cap)
            if np.isfinite(m_true) and m_true > 0:
                cap = m_true - 0.01
                st_c, _, lb_c, _, _ = eng.run(masks[0], b_true, cap, 1_000_000)
                assert st_c == ST_PROVED and lb_c > cap + EPS_TIE and \
                    lb_c <= m_true + 1e-9, "cap proof invalid"
                n_proved += 1

            # (4) tiny budgets: certified lower bound never exceeds the min
            for budget in (1, 2, 5, 17):
                st_b, val_b, lb_b, _, _ = eng.run(masks[0], b_true,
                                                  m_true + 0.5, budget)
                assert lb_b <= m_true + 1e-9, "budget lb exceeds the true min"
                if st_b == ST_MIN:
                    assert abs(val_b - m_true) < 1e-9
                n_budget_lb += 1

            # (5) sector verdict vs brute-force nondeg-ML truth
            cap_true = m_true + 0.5  # any valid upper bound works as the cap
            cert = certify_sector(
                mk_sector(), s.astype(np.int8), lam0, cap_true,
                true_budget=1_000_000, wrong_budget=1_000_000)
            brute_margin = m_wrong - m_true
            if cert["verdict"] == V_FAIL:
                assert brute_margin < 0, "certified FAIL but ML succeeds"
            elif cert["verdict"] == V_SUCCESS:
                assert brute_margin > 0, "certified SUCCESS but ML fails"
            if abs(brute_margin) > 1e-6:  # decidable -> must certify
                assert cert["verdict"] in (V_FAIL, V_SUCCESS), \
                    f"decidable shot (margin {brute_margin:.3g}) left uncertified"
            n_verdicts += 1

            # (6) numba kernel vs pure-python reference oracle
            if trial < 5:
                for mi, cap_r in ((0, m_true + 0.5), (1, m_true + 0.5)):
                    b_r = b_true.copy()
                    if mi:
                        b_r[m] ^= 1
                    st_r, val_r, lb_r, _, _ = reference_min(
                        A_full, m, masks[mi], b_r, w, cap_r, 300_000)
                    st_n, val_n, lb_n, _, _ = eng.run(masks[mi], b_r, cap_r,
                                                      300_000)
                    if st_r == ST_BUDGET and st_n == ST_BUDGET:
                        n_ref += 1  # tie-order may differ mid-run; both valid
                        continue
                    assert st_r == st_n, f"status mismatch {st_r} vs {st_n}"
                    if st_n == ST_MIN:
                        assert abs(val_r - val_n) < 1e-9, "value mismatch"
                    else:
                        assert abs(lb_r - lb_n) < 1e-9 or \
                            (np.isinf(lb_r) and np.isinf(lb_n)), "lb mismatch"
                    n_ref += 1

            # (7) NEW optimization layers vs brute force -----------------------
            # (7a) adaptive parity-cut LP: an admissible lower bound; an integral
            #      optimum is the EXACT coset minimum with a feasible witness.
            for mi, brute_mi in ((0, m_true), (1, minima[(np.arange(1 << k) & 1)
                                                         != (lam0_int & 1)].min())):
                b_mi = b_true.copy()
                if mi:
                    b_mi[m] ^= 1
                alp_lb, alp_wit, alp_cuts = _adaptive_lp(A_full, masks[mi],
                                                         b_mi, w)
                if np.isfinite(brute_mi):
                    assert alp_lb <= brute_mi + 1e-6, \
                        f"adaptive LP bound {alp_lb} exceeds true min {brute_mi}"
                if alp_wit is not None:
                    xa = np.zeros(n, dtype=np.int64); xa[alp_wit] = 1
                    active = masks[mi].astype(bool)
                    lhs = (A_full.astype(np.int64) @ xa % 2)
                    assert np.array_equal(lhs[active], b_mi.astype(np.int64)[active]), \
                        "adaptive-LP integral witness infeasible"
                    assert abs(float(w @ xa) - brute_mi) < 1e-9, \
                        "adaptive-LP integral witness not the exact minimum"
                n_lp += 1

            # (7b) exact integer ILP: proves the coset minimum (or infeasibility);
            #      recycled parity cuts and banned-column restriction must not
            #      change the exact minimum (cuts are valid for the binary
            #      coset; the below-cap argmin never uses a banned column)
            if trial < 10:
                for mi, brute_mi in ((0, m_true),
                                     (1, minima[(np.arange(1 << k) & 1)
                                                != (lam0_int & 1)].min())):
                    b_mi = b_true.copy()
                    if mi:
                        b_mi[m] ^= 1
                    st_il, val_il, lb_il, wit_il = _ilp_exact(A_full, masks[mi], b_mi, w)
                    if np.isinf(brute_mi):
                        assert st_il == ST_PROVED, "exact ILP missed infeasibility"
                    else:
                        assert st_il == ST_MIN and abs(val_il - brute_mi) < 1e-9, \
                            f"exact ILP min {val_il} vs brute {brute_mi}"
                        xi = np.zeros(n, dtype=np.int64); xi[wit_il] = 1
                        active = masks[mi].astype(bool)
                        lhs = (A_full.astype(np.int64) @ xi % 2)
                        assert np.array_equal(lhs[active],
                                              b_mi.astype(np.int64)[active]), \
                            "exact ILP witness infeasible"
                        # recycle the adaptive LP's cuts: same exact minimum
                        _, _, cuts_mi = _adaptive_lp(A_full, masks[mi], b_mi, w)
                        st_c, val_c, _, wit_c = _ilp_exact(
                            A_full, masks[mi], b_mi, w, cuts=cuts_mi)
                        assert st_c == ST_MIN and abs(val_c - brute_mi) < 1e-9, \
                            "cut-recycled ILP changed the exact minimum"
                        # ban columns above cap = brute_mi + 0.5: a below-cap
                        # argmin can never contain a banned column (its cost
                        # would exceed Phi + w'_j > cap), so the restricted
                        # minimum must be unchanged
                        dl = (_lp_dual_charges(mk_sector(), np.flatnonzero(s))
                              if s.any() else None)
                        if dl is not None:
                            phi_t, wred_t = dl
                            ban_t = (float(phi_t.sum()) + wred_t
                                     > brute_mi + 0.5 + BAN_SLACK)
                            assert not ban_t[wit_il].any(), \
                                "ban removed a below-cap argmin column"
                            keep_t = np.flatnonzero(~ban_t).astype(np.int64)
                            st_k, val_k, _, wit_k = _ilp_exact(
                                A_full, masks[mi], b_mi, w, keep=keep_t,
                                cuts=cuts_mi)
                            assert st_k == ST_MIN and \
                                abs(val_k - brute_mi) < 1e-9, \
                                "column-banned ILP changed the exact minimum"
                    n_ilp += 1

            # (7c) seed-driven verdict: hand certify_sector the true brute-force
            #      wrong-class argmin as a witness; it must reproduce the verdict.
            key_s = int(s @ powers_m)
            wm = (syn_int == key_s) & (lam_int != lam0_int)
            if wm.any():
                idxs = np.flatnonzero(wm)
                jbest = int(idxs[np.argmin(cost[idxs])])
                x_seed = ((jbest >> np.arange(n, dtype=np.int64)) & 1).astype(np.int64)
                cert_s = certify_sector(
                    mk_sector(), s.astype(np.int8), lam0, m_true + 0.5,
                    seed_wrong=[x_seed],
                    true_budget=1_000_000, wrong_budget=1_000_000)
                if m_wrong < m_true - 1e-6:
                    assert cert_s["verdict"] == V_FAIL, \
                        "seed did not certify a real FAIL"
                elif m_wrong > m_true + 1e-6:
                    assert cert_s["verdict"] == V_SUCCESS, \
                        "seed corrupted a real SUCCESS"
                fail_only = certify_sector(
                    mk_sector(), s.astype(np.int8), lam0, m_true + 0.5,
                    seed_wrong=[x_seed], true_budget=1_000_000,
                    wrong_budget=1_000_000,
                    cfg=DEFAULT_CFG.replace(fail_only=True))
                if m_wrong < m_true - 1e-6:
                    assert fail_only["verdict"] == V_FAIL, \
                        "true-class-only pass missed a seeded FAIL"
                else:
                    assert fail_only["verdict"] != V_FAIL, \
                        "true-class-only pass manufactured a FAIL"
                n_seed += 1

            # (8) UNIFIED single-coset decider vs brute force ------------------
            # One syndrome-only search must recover the exact global minimum, its
            # winning class, and the SUCCESS/FAIL/tie verdict -- matching brute
            # force AND the decomposition (block 5).
            gmin_brute = float(np.min(minima))
            (dst, gmin, cg1, mth, mwh, wcls, frontier, dnodes, dsupp,
             m2, c2) = \
                eng.decide(s.astype(np.int8), lam0_int, m_true + 0.5, 1_000_000)
            if dst == DEC_OK:
                assert abs(gmin - gmin_brute) < 1e-9, \
                    f"unified global-min {gmin} vs brute {gmin_brute}"
                assert abs(minima[cg1] - gmin_brute) < 1e-9, \
                    "unified winner class is not a global-min class"
                # m_second: exact min over classes != class_g1 whenever that
                # min ties the global min within eps (the cluster path's tie
                # detector); otherwise it must not report a value <= gmin+eps
                other = np.delete(minima, cg1)
                second_brute = float(other.min()) if other.size else np.inf
                if second_brute <= gmin_brute + EPS_TIE:
                    assert abs(m2 - second_brute) < 1e-9, \
                        f"m_second {m2} vs brute second-class min {second_brute}"
                    assert abs(minima[c2] - second_brute) < 1e-9, \
                        "class_second is not a second-class argmin"
                else:
                    assert not np.isfinite(m2) or m2 > gmin_brute + EPS_TIE, \
                        "m_second reported below a non-degenerate frontier"
                # verify the returned global-min witness in numpy
                if dsupp is not None:
                    xv = np.zeros(n, dtype=np.int64); xv[dsupp] = 1
                    assert (Hdec.astype(np.int64) @ xv % 2 == s).all(), \
                        "unified witness violates the syndrome"
                    assert abs(float(w @ xv) - gmin) < 1e-9, \
                        "unified witness cost mismatch"
                    cls_v = int((H_log.astype(np.int64) @ xv % 2)
                                @ (1 << np.arange(k, dtype=np.int64)))
                    assert cls_v == cg1, "unified witness class mismatch"
                # derive and check the verdict against brute truth
                loser = mwh if cg1 == lam0_int else mth
                if cg1 == lam0_int:
                    uv = V_TIE if loser <= gmin + EPS_TIE else V_SUCCESS
                else:
                    uv = V_TIE if loser <= gmin + EPS_TIE else V_FAIL
                margin = m_wrong - m_true
                truth = (V_SUCCESS if margin > EPS_TIE else
                         V_FAIL if margin < -EPS_TIE else V_TIE)
                assert uv == truth, \
                    f"unified verdict {uv} != brute truth {truth} (margin {margin:.3g})"
                # full-path integration: certify_sector with the flag ON
                if trial < 6:
                    # full-path integration: certify_sector with the unified
                    # single-coset pre-pass enabled via a cfg override (replaces
                    # the old module-global flip).
                    cu = certify_sector(mk_sector(), s.astype(np.int8), lam0,
                                        m_true + 0.5, true_budget=1_000_000,
                                        wrong_budget=1_000_000,
                                        cfg=DEFAULT_CFG.replace(use_unified_search=True))
                    if abs(margin) > 1e-6:
                        assert cu["verdict"] == truth, \
                            f"certify_sector(unified) {cu['verdict']} != {truth}"
                n_uni += 1

            # (8b) phi-dual heuristic + column ban in the A* kernels ---------
            # The covering-LP dual must lower-bound the global coset minimum;
            # with row_phi as an extra admissible bound and provably irrelevant
            # columns banned, exact minima, cap proofs, and budget lower bounds
            # must all be preserved bit-exactly.
            if s.any():
                dphi = _lp_dual_charges(mk_sector(), np.flatnonzero(s))
                if dphi is not None:
                    phi_t, wred_t = dphi
                    gmin_brute = float(np.min(minima))
                    assert float(phi_t.sum()) <= gmin_brute + 1e-6, \
                        "covering dual exceeds the global coset minimum"
                    rphi = np.zeros(m + k, dtype=np.float64)
                    rphi[np.flatnonzero(s)] = phi_t
                    cap_b = m_true + 0.5
                    ban_u = np.ascontiguousarray(
                        float(phi_t.sum()) + wred_t > cap_b + BAN_SLACK,
                        dtype=np.uint8)
                    st_p, val_p, lb_p, _, sol_p = eng.run(
                        masks[0], b_true, cap_b, 1_000_000,
                        row_phi=rphi, col_banned=ban_u)
                    assert st_p == ST_MIN and abs(val_p - m_true) < 1e-9, \
                        f"phi/ban search min {val_p} vs brute {m_true}"
                    xv = np.zeros(n, dtype=np.int64); xv[sol_p] = 1
                    assert (Hdec.astype(np.int64) @ xv % 2 == s).all()
                    assert (H_log.astype(np.int64) @ xv % 2 == lam0).all()
                    # wrong-bit-0 at the true cap: exact when within cap,
                    # else a certified above-cap proof
                    b_0 = b_true.copy(); b_0[m] ^= 1
                    brute_0 = minima[(np.arange(1 << k) & 1)
                                     != (lam0_int & 1)].min()
                    st_w, val_w, lb_w, _, _ = eng.run(
                        masks[1], b_0, cap_b, 1_000_000,
                        row_phi=rphi, col_banned=ban_u)
                    if brute_0 <= cap_b:
                        assert st_w == ST_MIN and abs(val_w - brute_0) < 1e-9, \
                            "phi/ban wrong-bit min mismatch"
                    else:
                        assert st_w != ST_MIN and lb_w > cap_b + EPS_TIE, \
                            "phi/ban above-cap proof invalid"
                    # budget-limited lower bounds stay below the true minimum
                    for budget in (1, 4, 19):
                        st_b, val_b, lb_b, _, _ = eng.run(
                            masks[0], b_true, cap_b, budget,
                            row_phi=rphi, col_banned=ban_u)
                        assert lb_b <= m_true + 1e-9, \
                            "phi/ban budget lb exceeds the true min"
                    n_phi += 1
    # (9) certified cluster decomposition vs brute force ----------------------
    # Block-diagonal instances with bridge columns: the separation test must
    # merge across cheap bridges (fallback) and may split across expensive
    # ones; every decomposed verdict, witness cost and bound is checked against
    # the exhaustive 2^20 truth, and the LP dual bound against the global min.
    n_clu = n_clu_fb = n_clu_lp = 0
    for inst in range(6):
        bridge_w = (3.0, 8.0, 25.0)[inst % 3]
        Hdec, H_log, w = _selftest_cluster_instance(rng, bridge_w)
        m, n = Hdec.shape
        k = H_log.shape[0]
        A_full = np.vstack([Hdec, H_log]).astype(np.int8)
        eng = CertEngine(A_full, m, w, pool_cap=1_500_000)
        masks = np.zeros((k + 1, m + k), dtype=np.uint8)
        masks[:, :m] = 1
        masks[0, m:] = 1
        for i in range(k):
            masks[1 + i, m + i] = 1
        sector = Sector(name="C", Hdec=Hdec, Hdec_csr=None,
                        H_log=H_log.astype(np.int64), H_aug=A_full,
                        channel_probs=np.full(n, 0.01), w=w,
                        powers=1 << np.arange(k, dtype=np.int64),
                        engine=eng, masks=masks)
        syn_int, lam_int, cost = _brute_tables(Hdec, H_log, w)
        powers_m = 1 << np.arange(m, dtype=np.int64)
        for trial in range(40):
            # one or two errors per block (so both blocks fire) + occasionally
            # a bridge column, to exercise merge and split regimes
            x0 = np.zeros(n, dtype=np.int64)
            x0[rng.integers(0, 9, size=int(rng.integers(1, 3)))] = 1
            x0[9 + rng.integers(0, 9, size=int(rng.integers(1, 3)))] = 1
            if rng.random() < 0.15:
                x0[n - 1 - int(rng.integers(0, 2))] = 1
            s = (Hdec.astype(np.int64) @ x0) % 2
            if not s.any():
                continue
            lam0 = (H_log.astype(np.int64) @ x0) % 2
            lam0_int = int(lam0 @ (1 << np.arange(k, dtype=np.int64)))
            minima = _brute_class_minima(syn_int, lam_int, cost,
                                         int(s @ powers_m), k)
            m_true = minima[lam0_int]
            m_wrong = np.delete(minima, lam0_int).min()
            gmin_brute = float(minima.min())
            dual = _lp_dual_charges(sector, np.flatnonzero(s))
            if dual is not None:  # Phi must lower-bound the global coset min
                assert float(dual[0].sum()) <= gmin_brute + 1e-6, \
                    "LP dual charge exceeds the global minimum"
                n_clu_lp += 1
            cl = _cluster_verdict(sector, s.astype(np.int8), lam0,
                                  m_true + 0.5, 1_000_000)
            if cl is None:
                n_clu_fb += 1
                continue
            margin = m_wrong - m_true
            assert cl["n_clusters"] >= 2 and cl["clustered"]
            if cl["verdict"] == V_SUCCESS:
                assert margin > EPS_TIE, "cluster SUCCESS but ML fails/ties"
                assert abs(cl["ub_true"] - m_true) < 1e-9, \
                    "cluster m_true != brute m_true"
                assert abs(cl["ub_true"] - gmin_brute) < 1e-9
                assert cl["lb_wrong"] <= m_wrong + 1e-9, \
                    "cluster lb_wrong exceeds brute m_wrong"
            elif cl["verdict"] == V_FAIL:
                assert margin < -EPS_TIE, "cluster FAIL but ML succeeds/ties"
                assert abs(cl["ub_wrong"] - m_wrong) < 1e-9, \
                    "cluster m_wrong != brute m_wrong"
                assert abs(cl["ub_wrong"] - gmin_brute) < 1e-9
                assert cl["lb_true"] <= m_true + 1e-9, \
                    "cluster lb_true exceeds brute m_true"
            else:
                raise AssertionError("cluster path returned a non-decision")
            n_clu += 1
    assert n_clu > 0, "cluster selftest never split an instance -- test broken"
    print(f"  {n_exact} exact minima, {n_proved} cap proofs, "
          f"{n_budget_lb} budget bounds, {n_verdicts} verdicts, "
          f"{n_ref} reference cross-checks -- all OK "
          f"({time.time() - t0:.1f}s)")
    print(f"  {n_lp} adaptive-LP bounds, {n_ilp} exact-ILP minima, "
          f"{n_any_wrong} all-wrong ILP checks, "
          f"{n_seed} seed-driven verdicts vs brute force -- all OK")
    print(f"  {n_uni} unified single-coset decisions vs brute force -- all OK")
    print(f"  {n_phi} phi-dual/column-ban kernel runs vs brute force -- all OK")
    print(f"  {n_clu} cluster-decomposed verdicts vs brute force "
          f"({n_clu_fb} merged/fell back, {n_clu_lp} LP dual bounds) -- all OK")
    print("Self-test passed.")
