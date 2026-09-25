"""The per-sector verdict engine.

``certify_sector`` runs the decomposed 1-true + k-wrong certified searches
(with the covering-LP dual bound / column ban, the cluster pre-pass, the
optional unified pre-pass, the adaptive-LP / MILP / exact-ILP screens, and
witness seeding) and returns the sector's SUCCESS / FAIL / TIE / BUDGET verdict
with its one-sided intervals.  ``_unified_verdict`` is the single-search
pre-pass.

All user knobs arrive on the ``cfg`` (:class:`CertConfig`) argument; the
soundness epsilons come from ``constants``.
"""

import numpy as np

from ..config import DEFAULT_CFG
from .constants import (
    EPS_TIE, BAN_SLACK, DEC_OK, ST_MIN, ST_PROVED, ST_BUDGET,
    V_SUCCESS, V_FAIL, V_TIE, V_BUDGET,
)
from .sector import pack_class, _witness_eval, _verify_witness
from .relaxations import (
    _lp_root_bound, _adaptive_lp, _milp_lower_bound, _ilp_exact,
)
from .clusters import _cluster_verdict, _lp_dual_charges

def _unified_verdict(sector, s, lam_bits, cap_true, node_budget, verify=True,
                     row_phi=None, col_banned=None):
    """Try to decide this sector with ONE unified syndrome-only search.

    Returns a full sector-verdict dict when the search conclusively decides
    (frontier cleared), else None (inconclusive -> caller runs the decomposed
    searches).  Rigor is identical to the decomposition: the winner is the exact
    global (=ML) minimum and the loser class min is certified above it.
    """
    k = len(sector.powers)
    lam0_pk = pack_class(lam_bits, sector.powers)
    (status, gmin, cg1, mth, mwh, wcls, frontier, nodes, support,
     _m2, _c2) = sector.engine.decide(s, lam0_pk, cap_true, node_budget,
                                      row_phi=row_phi, col_banned=col_banned)
    if status != DEC_OK or not np.isfinite(gmin):
        return None                           # inconclusive -> fall back

    # Re-verify the global-min witness in numpy: it must satisfy the syndrome,
    # carry class cg1, and cost exactly gmin.  This catches any kernel error
    # before the decision can enter the bracket.
    if verify and support is not None:
        x = np.zeros(sector.H_aug.shape[1], dtype=np.int64)
        x[support] = 1
        cost_w, cls_w = _witness_eval(sector, s, x, verify=True)
        assert abs(cost_w - gmin) < 1e-7 and cls_w == cg1, \
            f"{sector.name}: unified global-min witness mismatch"

    winner_is_true = (cg1 == lam0_pk)
    loser_hat = mwh if winner_is_true else mth   # min recorded in the loser class
    tie = loser_hat <= gmin + EPS_TIE
    statuses = np.full(k + 1, -1, dtype=np.int8)
    nodes_arr = np.zeros(k + 1, dtype=np.int64)
    nodes_arr[0] = nodes
    out = {
        "statuses": statuses, "nodes": nodes_arr, "nodes_total": int(nodes),
        "lb_wrong_lp": np.inf, "success_lp": False, "success_no_milp": False,
        "esc_margin": np.inf, "escalations": 0, "lp_screened": 0,
        "milp_calls": 0, "ilp_capped": 0, "wrong_witness_class": -1,
        "unified": True,   # decided by the unified single-coset search
        "clustered": False, "n_clusters": 0, "banned_cols": 0,
    }
    if tie:
        # winner == loser cost within eps: neither SUCCESS nor FAIL is provable
        out.update(verdict=V_TIE, lb_true=gmin, ub_true=cap_true,
                   lb_wrong=gmin, ub_wrong=(mwh if np.isfinite(mwh) else np.inf),
                   true_exact=winner_is_true,
                   wrong_witness_class=(int(wcls) if wcls >= 0 else -1))
        return out
    if winner_is_true:
        # m_true = gmin exactly; every wrong solution exceeds the frontier
        out.update(verdict=V_SUCCESS, lb_true=gmin, ub_true=gmin,
                   lb_wrong=min(mwh, frontier), ub_wrong=mwh,
                   true_exact=True,
                   wrong_witness_class=(int(wcls) if wcls >= 0 else -1))
    else:
        # winner is a wrong class at gmin (= m_wrong exactly); m_true > gmin
        out.update(verdict=V_FAIL, lb_true=min(mth, frontier), ub_true=cap_true,
                   lb_wrong=gmin, ub_wrong=gmin, true_exact=False,
                   wrong_witness_class=int(cg1))
    return out


def certify_sector(sector, s, lam_bits, cap_true, budget_scale=1,
                   verify=True, true_budget=None, wrong_budget=None,
                   seed_wrong=None, cfg=DEFAULT_CFG):
    """The certified searches and the sector verdict.

    Adaptive caps: wrong-bit searches run at cap UB_true while a SUCCESS
    certificate is still possible; once any of them fails to prove its bound
    (success dead) the remaining searches only hunt FAIL witnesses, which
    live strictly below LB_true -- a much smaller sphere.  The loop breaks as
    soon as a FAIL witness is found.  Every outcome is interpreted against
    the cap it ran with, so the one-sided rigor is unaffected.

    Returns a dict with the verdict, the one-sided intervals, per-search
    statuses/nodes (status -1 = search skipped), and (when FAIL) the best
    wrong-class witness's class.
    """
    eng = sector.engine
    m = sector.Hdec.shape[0]
    k = len(sector.powers)
    n_cols = sector.H_aug.shape[1]
    true_budget = cfg.true_node_budget if true_budget is None else true_budget
    wrong_budget = cfg.wrong_node_budget if wrong_budget is None else wrong_budget
    b_true = np.concatenate([s.astype(np.int8), lam_bits.astype(np.int8)])
    lam0_pk = pack_class(lam_bits, sector.powers)
    ilp_calls = 0        # exact integer-x ILP solves spent this sector
    ilp_capped = 0       # searches left budget-stuck because the ILP cap was hit

    # Verified covering-LP dual of this shot (one solve, shared by every layer):
    #   * row_phi  -- per-detector charges, an LP-strength admissible A* bound;
    #   * col_banned/keep -- columns with Phi + w'_j > cap_true + BAN_SLACK
    #     provably appear in no solution any verdict compares against (every
    #     verdict margin sits at cap_true + eps or below, eps << BAN_SLACK),
    #     so they are excluded from the searches and the LP/ILP solves;
    #   * the same dual drives the cluster separation test.
    dual = None
    row_phi = None
    col_banned = None
    keep = None
    n_banned = 0
    D_fired = np.flatnonzero(np.asarray(s) != 0).astype(np.int64)
    if (cfg.use_dual_bounds or cfg.use_clusters) and D_fired.size:
        dual = _lp_dual_charges(sector, D_fired)
    if cfg.use_dual_bounds and dual is not None:
        phi_d, wred_d = dual
        row_phi = np.zeros(m + k, dtype=np.float64)
        row_phi[D_fired] = phi_d
        ban = (float(phi_d.sum()) + wred_d
               > float(cap_true) + BAN_SLACK)
        n_banned = int(ban.sum())
        if n_banned:
            col_banned = np.ascontiguousarray(ban, dtype=np.uint8)
            keep = np.flatnonzero(~ban).astype(np.int64)

    # Certified cluster decomposition: when the fired detectors provably split
    # into non-interacting clusters (see _cluster_labels_core), decide each
    # cluster with its own tiny exact search and compose the verdict.  Ties,
    # budget-outs and single-cluster shots return None and fall through, so
    # this can only ever accelerate, never change a verdict.
    if cfg.use_clusters:
        cl = _cluster_verdict(sector, s, lam_bits, cap_true,
                              cfg.cluster_node_budget * budget_scale, verify,
                              dual=dual, row_phi=row_phi,
                              col_banned=col_banned)
        if cl is not None:
            cl["banned_cols"] = n_banned
            return cl

    # Unified single-coset decider (opt-in): one syndrome-only search that decides
    # the whole sector.  On a conclusive verdict use it directly; otherwise fall
    # through to the decomposed 1-true + k-wrong searches (which never lose
    # coverage), so this can only ever accelerate, never corrupt.
    if cfg.use_unified_search:
        uni = _unified_verdict(sector, s, lam_bits, cap_true,
                               cfg.unified_node_budget * budget_scale, verify,
                               row_phi=row_phi, col_banned=col_banned)
        if uni is not None:
            uni["banned_cols"] = n_banned
            return uni

    def _dense(supp):
        x = np.zeros(n_cols, dtype=np.int64)
        x[supp] = 1
        return x

    def _screen(mask, b, cap):
        """(bound, witness, cuts) LP screen.  Adaptive parity-cut LP when
        enabled (its integral optimum is an exact coset witness and its cuts
        are recycled into any later ILP on the same system), else the cheap
        root LP."""
        if cfg.use_adaptive_lp:
            return _adaptive_lp(sector.H_aug, mask, b, sector.w,
                                max_rounds=cfg.adaptive_lp_rounds, keep=keep)
        if cfg.use_lp_screen:
            return _lp_root_bound(sector.H_aug, mask, b, sector.w), None, None
        return 0.0, None, None

    st_t, val_t, lb_t, nodes_t, sol_t = eng.run(
        sector.masks[0], b_true, cap_true, true_budget * budget_scale,
        row_phi=row_phi, col_banned=col_banned)
    assert st_t != ST_PROVED, (
        f"{sector.name}: true-class search proved min > its own upper bound "
        f"(cap={cap_true}) -- certificate invariant broken")
    if st_t == ST_MIN:
        lb_true = ub_true = val_t
        if verify:
            _verify_witness(sector, b_true, sector.masks[0], sol_t, val_t, "true")
    else:
        lb_true, ub_true = lb_t, cap_true
        # B&B exhausted its budget on the true class.  An LP relaxation of the
        # true-coset system is an independent admissible lower bound on m_lam0;
        # its integral optimum (if any) is exact.  A higher lb_true sharpens the
        # FAIL side (raises the wrong-bit caps and the val_i < lb_true test); a
        # lower ub_true sharpens the SUCCESS side (lowers every wrong cap).
        lp_lb_true, lp_wit_true, cuts_true = _screen(
            sector.masks[0], b_true, cap_true)
        if lp_wit_true is not None:
            cost_w, _ = _witness_eval(sector, s, _dense(lp_wit_true), verify)
            lb_true = ub_true = cost_w          # exact m_lam0 from the LP witness
        else:
            if np.isfinite(lp_lb_true) and lp_lb_true > lb_true:
                lb_true = min(lp_lb_true, ub_true)
            # Exact ILP pins m_lam0: status-0 gives lb_true = ub_true = m_lam0; a
            # time-limit still tightens both sides (dual bound raises lb_true, a
            # verified true-class incumbent lowers ub_true below the OSD-0 anchor).
            if cfg.use_exact_ilp and ilp_calls < cfg.exact_ilp_max_calls:
                st_il, val_il, lb_il, wit_il = _ilp_exact(
                    sector.H_aug, sector.masks[0], b_true, sector.w, keep=keep,
                    time_limit=cfg.exact_ilp_time_limit,
                    cuts=cuts_true if cfg.use_cut_recycle else None)
                ilp_calls += 1
                if st_il == ST_MIN:
                    cost_w, _ = _witness_eval(sector, s, _dense(wit_il), verify)
                    lb_true = ub_true = cost_w
                elif st_il == ST_BUDGET:
                    if lb_il > lb_true:
                        lb_true = min(lb_il, ub_true)
                    if wit_il is not None:
                        cost_w, cls_w = _witness_eval(
                            sector, s, _dense(wit_il), verify)
                        if cls_w == lam0_pk and cost_w < ub_true:
                            ub_true = cost_w

    statuses = np.full(k + 1, -1, dtype=np.int8)
    nodes = np.zeros(k + 1, dtype=np.int64)
    statuses[0], nodes[0] = st_t, nodes_t
    ub_wrong = np.inf       # best (smallest) achieved wrong-class cost -> UB on m_wrong
    lb_wrong = np.inf       # certified lower bound on m_wrong over searches run
    lb_wrong_lp = np.inf    # LP-only lower bound on the wrong min (no A*/ILP) --
                            # for the LP-only bracket (ml_ub_lp)
    wrong_witness_class = -1
    success_alive = True    # no wrong solution <= ub_true has been exhibited yet
    fail_found = False
    worst_block = np.inf    # min certified lb among proof-blocking searches
    lp_screened = 0         # wrong-bit A* searches skipped by an LP bound >= cap

    # Seed witnesses: verified wrong-class solutions from the production decoders
    # (e.g. BP+OSD / Tesseract beam), each satisfying Hdec x = s.  Such a point
    # proves m_wrong <= cost(x); if cost(x) < lb_true it proves FAIL outright (no
    # search needed), and if cost(x) <= ub_true it kills SUCCESS (the remaining
    # searches then run at the cheaper FAIL-hunt cap lb_true).  Pure acceleration
    # -- seeds only ever supply UPPER bounds on m_wrong, never touch lb_wrong.
    if seed_wrong:
        for xw in seed_wrong:
            cw, clsw = _witness_eval(sector, s, xw, verify)
            if clsw == lam0_pk:
                continue                        # not actually a wrong class
            if cw < ub_wrong:
                ub_wrong, wrong_witness_class = cw, clsw
            if cw <= ub_true + EPS_TIE:
                success_alive = False   # a wrong solution <= ub_true exists
            if cw < lb_true - EPS_TIE:
                fail_found = True        # ... and strictly below the true LB => FAIL

    for i in range(0 if cfg.fail_only else k):
        if fail_found:
            break                               # a FAIL witness already decides
        b_i = b_true.copy()
        b_i[m + i] ^= 1
        cap_i = ub_true if success_alive else lb_true

        # Screen: an admissible LP bound on this wrong coset's minimum, kept
        # regardless of outcome.  An integral LP optimum IS the exact wrong
        # minimum (decide directly); otherwise a bound >= cap_i proves the coset
        # cannot undercut the true class and A* is skipped.
        lp_lb, lp_wit, lp_cuts = _screen(sector.masks[1 + i], b_i, cap_i)
        lb_wrong_lp = min(lb_wrong_lp, lp_lb)
        if lp_wit is not None:
            val_i, clsw = _witness_eval(sector, s, _dense(lp_wit), verify)
            statuses[1 + i] = ST_MIN
            if val_i < ub_wrong:
                ub_wrong, wrong_witness_class = val_i, clsw
            lb_wrong = min(lb_wrong, val_i)
            if val_i < lb_true - EPS_TIE:
                fail_found = True
                break
            success_alive = success_alive and (val_i > ub_true + EPS_TIE)
            continue
        if lp_lb >= cap_i + EPS_TIE:
            statuses[1 + i] = ST_PROVED
            lb_wrong = min(lb_wrong, lp_lb)
            lp_screened += 1
            continue

        st_i, val_i, lb_i, nodes_i, sol_i = eng.run(
            sector.masks[1 + i], b_i, cap_i, wrong_budget * budget_scale,
            row_phi=row_phi, col_banned=col_banned)
        statuses[1 + i], nodes[1 + i] = st_i, nodes_i
        if st_i == ST_MIN:
            if verify:
                _verify_witness(sector, b_i, sector.masks[1 + i], sol_i, val_i,
                                f"wrong-bit-{i}")
            if val_i < ub_wrong:
                ub_wrong = val_i
                lam_w = sector.H_log @ np.bincount(
                    sol_i, minlength=n_cols).astype(np.int64) % 2
                wrong_witness_class = pack_class(lam_w, sector.powers)
            lb_wrong = min(lb_wrong, val_i)
            if val_i < lb_true - EPS_TIE:
                fail_found = True
                break           # one witness decides FAIL; skip the rest
            success_alive = False  # a wrong min <= cap_i <= UB_true exists
        elif st_i == ST_PROVED:
            lb_wrong = min(lb_wrong, lb_i)  # lb_i > cap_i + eps
        else:  # ST_BUDGET: A* could not decide.  lb_i (with the LP bound) is a
               # certified lower bound on this wrong coset's minimum.
            lb_eff = max(lb_i, lp_lb)
            # Cheap continuous-z MILP bound first (~ms): it reactivates the even-
            # parity rows the LP screen drops and frequently already clears the
            # cap, so it both dominates the pure-LP bound and saves the expensive
            # exact ILP.  Kept always-on (it can only raise an admissible LB).
            if cfg.use_milp_fallback:
                milp_lb, _ = _milp_lower_bound(
                    sector.H_aug, sector.masks[1 + i], b_i, sector.w,
                    time_limit=cfg.milp_time_limit,
                    keep=keep, cuts=lp_cuts if cfg.use_cut_recycle else None)
                if milp_lb > lb_eff:
                    lb_eff = milp_lb
            # Would an exact solve still change the verdict?  In the SUCCESS regime
            # only if the current bound has not yet cleared ub_true; in the FAIL
            # hunt only if a witness could still lie below lb_true.
            need_ilp = (lb_eff <= ub_true + EPS_TIE if success_alive
                        else lb_eff < lb_true - EPS_TIE)
            # Exact ILP: decisive in BOTH regimes.  status-0 yields the exact
            # m_wrong_i with a re-verified witness (proves FAIL if < lb_true, or
            # kills SUCCESS / proves > ub_true otherwise); a time-limit yields a
            # certified dual bound plus, if present, a verified incumbent.
            if cfg.use_exact_ilp and need_ilp and ilp_calls < cfg.exact_ilp_max_calls:
                st_il, val_il, lb_il, wit_il = _ilp_exact(
                    sector.H_aug, sector.masks[1 + i], b_i, sector.w,
                    time_limit=cfg.exact_ilp_time_limit,
                    keep=keep, cuts=lp_cuts if cfg.use_cut_recycle else None)
                ilp_calls += 1
                if st_il == ST_MIN:
                    val_x, clsw = _witness_eval(sector, s, _dense(wit_il), verify)
                    statuses[1 + i] = ST_MIN
                    if val_x < ub_wrong:
                        ub_wrong, wrong_witness_class = val_x, clsw
                    lb_wrong = min(lb_wrong, val_x)
                    if val_x < lb_true - EPS_TIE:
                        fail_found = True
                        break
                    success_alive = success_alive and (val_x > ub_true + EPS_TIE)
                    continue
                if st_il == ST_PROVED:
                    statuses[1 + i] = ST_PROVED   # infeasible coset: m_wrong_i=+inf
                    continue
                # ST_BUDGET: fold the certified dual bound and any incumbent
                if lb_il > lb_eff:
                    lb_eff = lb_il
                if wit_il is not None:
                    cwit, clsw = _witness_eval(sector, s, _dense(wit_il), verify)
                    if clsw != lam0_pk:
                        if cwit < ub_wrong:
                            ub_wrong, wrong_witness_class = cwit, clsw
                        if cwit < lb_true - EPS_TIE:
                            fail_found = True
                            break
                        if cwit <= ub_true + EPS_TIE:
                            success_alive = False
            elif cfg.use_exact_ilp and need_ilp:
                ilp_capped += 1               # cap hit: this search stays stuck
            if lb_eff > ub_true + EPS_TIE:
                statuses[1 + i] = ST_PROVED
            lb_wrong = min(lb_wrong, lb_eff)
            if success_alive and lb_eff <= ub_true + EPS_TIE:
                success_alive = False
                worst_block = min(worst_block, lb_eff)
            elif not success_alive and lb_eff <= lb_true:
                worst_block = min(worst_block, lb_eff)  # FAIL side also blocked

    fail = fail_found
    # A verified FAIL witness (from a search, ILP, or seed) is a rigorous proof
    # and always dominates: it can never coexist with a valid SUCCESS proof, so
    # exclude it explicitly (a seed/incumbent FAIL supplies only an UPPER bound on
    # m_wrong and so does not lower lb_wrong on its own).
    success = (not cfg.fail_only and success_alive and (not fail_found)
               and lb_wrong > ub_true + EPS_TIE)
    # LP-only sector verdict: would the LP screen alone -- with the anchor/seed
    # anchor (cap_true) as the only upper bound and no A*/ILP -- already certify
    # SUCCESS?  True iff every wrong coset's LP bound clears the anchor.
    success_lp = bool((cfg.use_adaptive_lp or cfg.use_lp_screen) and not fail_found
                      and lb_wrong_lp > cap_true + EPS_TIE)
    # Would this sector still be SUCCESS without any expensive integer solve?  An
    # ILP/MILP call fires only on a budget-stuck search whose cheap bound was
    # <= ub_true, so it would otherwise have killed SUCCESS.  Hence
    # success-without-solver == SUCCESS reached with zero ILP/MILP solves.
    success_no_milp = bool(success and ilp_calls == 0)
    assert not (fail and success), \
        f"{sector.name}: FAIL and SUCCESS certified simultaneously"
    if fail:
        verdict = V_FAIL
    elif success:
        verdict = V_SUCCESS
    else:
        budget_limited = (cfg.fail_only or st_t == ST_BUDGET or np.isfinite(worst_block)
                          or ilp_capped > 0)
        verdict = V_BUDGET if budget_limited else V_TIE
    return {
        "verdict": verdict,
        "lb_true": lb_true, "ub_true": ub_true,
        "lb_wrong": lb_wrong, "ub_wrong": ub_wrong,
        "lb_wrong_lp": lb_wrong_lp, "success_lp": success_lp,
        "success_no_milp": success_no_milp,
        "statuses": statuses, "nodes": nodes,
        "nodes_total": int(nodes.sum()),
        "wrong_witness_class": wrong_witness_class,
        # how far the blocked proofs were from deciding (drives escalation)
        "esc_margin": (ub_true + EPS_TIE - worst_block)
                      if np.isfinite(worst_block) else np.inf,
        "true_exact": bool(lb_true == ub_true),
        "escalations": 0,
        "lp_screened": lp_screened,   # wrong-bit searches skipped by an LP bound
        "milp_calls": ilp_calls,      # expensive integer-solver solves this sector
        "ilp_capped": ilp_capped,     # searches left stuck by the per-sector ILP cap
        "unified": False, "clustered": False, "n_clusters": 0,
        "banned_cols": n_banned,      # columns proven irrelevant below the cap
    }
