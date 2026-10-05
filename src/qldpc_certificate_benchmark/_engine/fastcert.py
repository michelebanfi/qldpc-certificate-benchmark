"""Cheap first pass and heavy last resort around ``certify_sector``.

``fast_unified``
    The unified single-coset decider with NO true-class cap (cap = +inf) and the
    verified covering-LP dual charges as the A* bound.  The cap only prunes
    pushes before the first goal is found (``certified_decide_core`` then
    tightens it to the global minimum), and every dropped node stays in the
    certified frontier, so the verdict is exactly as rigorous as the anchored
    search -- it just skips the true-class anchor, the seed folding and the
    column ban.  On [[144,12,12]] circuit noise it decides a sector in 2-4 ms
    against 80-226 ms for the default decomposition (A/B identical verdicts on
    6000 sectors).  With a large ``node_budget`` (and a matching ``pool_cap``;
    pushes outnumber expansions ~10-20x) the same search is also a cheap
    re-try for sectors the full certificate left undecided.

``global_route``
    Exact global ILPs for sectors every search left undecided: a syndrome-only
    ILP supplies a re-verified witness of cost c and class ell; then ONE
    infeasibility proof that no competing class has a solution of cost
    <= c + margin.  If ell is the true class the competitor is "any wrong
    class" (``_ilp_any_wrong_below``, SUCCESS); otherwise it is the true class
    (``_ilp_class_below``, FAIL).  Neither proof relies on an ILP optimality
    claim, only on infeasibility, so there is no MIP gap to trust.  About
    1-2 min per hard d=12 sector at p=0.004-0.006.

Both return a sector-verdict dict in the ``certify_sector`` format, or ``None``.
No decoder output enters either, so the verdicts are independent of every
benchmarked decoder.
"""

import numpy as np
from scipy.sparse import csr_matrix

from .clusters import _lp_dual_charges
from .constants import EPS_TIE, ST_BUDGET, ST_MIN, ST_PROVED, V_FAIL, V_SUCCESS
from .relaxations import (_Bounds, _LC, _ilp_any_wrong_below, _ilp_exact,
                          _sp_hstack, _sp_milp, _speye)
from .sector import _witness_eval, pack_class
from .verdicts import _unified_verdict

# From the loosest margin down: a larger one is cheaper to prove; the small
# ones decide near-ties (true/wrong gaps of 1e-3 nats are common).  All stay
# >= 1e-6, far above EPS_TIE and HiGHS's feasibility tolerances.
GLOBAL_MARGINS = (0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6)


def _dual_row_phi(sector, s):
    """Per-detector covering-LP dual charges (as in ``certify_sector``), or None."""
    fired = np.flatnonzero(np.asarray(s) != 0).astype(np.int64)
    if not fired.size:
        return None
    dual = _lp_dual_charges(sector, fired)
    if dual is None:
        return None
    row_phi = np.zeros(sector.Hdec.shape[0] + len(sector.powers), dtype=np.float64)
    row_phi[fired] = dual[0]
    return row_phi


def fast_unified(sector, s, lam_bits, node_budget, verify=True):
    """Cap-free unified search; a verdict dict or None when inconclusive."""
    out = _unified_verdict(sector, s, lam_bits, np.inf, int(node_budget),
                           verify=verify, row_phi=_dual_row_phi(sector, s),
                           col_banned=None)
    if out is not None:
        out["route"] = "fast"
    return out


def _ilp_class_below(H_aug, b, w, cap, time_limit):
    """Is there x in {0,1}^n with H_aug x = b (mod 2) and w.x <= cap?

    A pure feasibility problem (zero objective), so no MIP optimality gap is
    involved: ST_PROVED = infeasible = every member of this coset costs more
    than cap; ST_MIN = a verified witness at or below cap; else ST_BUDGET.
    Integer-z bounds are floored and the cost row has a finite lower bound,
    the HiGHS presolve safeguards of ``_ilp_exact``.
    """
    A = csr_matrix(np.asarray(H_aug, dtype=np.float64))
    R, n = A.shape
    b = np.asarray(b, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    max_z = np.floor(np.asarray(A.sum(axis=1)).ravel() / 2.0)
    A_eq = _sp_hstack([A, _speye(R, format="csr") * (-2.0)], format="csr")
    cost = csr_matrix(np.concatenate([w, np.zeros(R)]).reshape(1, -1))
    try:
        res = _sp_milp(np.zeros(n + R), integrality=np.ones(n + R),
                       bounds=_Bounds(np.zeros(n + R), np.concatenate([np.ones(n), max_z])),
                       constraints=[_LC(A_eq, b, b), _LC(cost, 0.0, float(cap))],
                       options={"time_limit": float(time_limit), "mip_rel_gap": 0.0})
    except Exception:
        return ST_BUDGET, None
    if res.status == 2:
        return ST_PROVED, None
    if res.x is not None and np.all(np.isfinite(res.x)):
        x = np.rint(res.x[:n]).astype(np.int64)
        if (np.all((x == 0) | (x == 1))
                and np.array_equal(A.astype(np.int64) @ x % 2, b.astype(np.int64))
                and float(w @ x) <= cap + 1e-7):
            return ST_MIN, np.flatnonzero(x)
    return ST_BUDGET, None


def global_route(sector, s, lam_bits, time_limit, margins=GLOBAL_MARGINS):
    """SUCCESS / FAIL from exact global ILPs, else None."""
    m = sector.Hdec.shape[0]
    true_cls = pack_class(lam_bits, sector.powers)
    st, _, _, supp = _ilp_exact(sector.Hdec, np.ones(m, dtype=bool), s, sector.w,
                                time_limit=time_limit)
    if supp is None:
        return None
    x = np.zeros(sector.Hdec.shape[1], dtype=np.int64)
    x[supp] = 1
    cost, cls = _witness_eval(sector, s, x)          # re-verifies the syndrome
    b_true = np.concatenate([s, lam_bits.astype(np.int8)])
    for margin in margins:
        cap = cost + margin
        if cls == true_cls:
            st2, _ = _ilp_any_wrong_below(sector.Hdec, sector.H_log, s, lam_bits,
                                          sector.w, cap, time_limit=time_limit)
            if st2 == ST_PROVED:
                assert cap > cost + EPS_TIE
                return _global_out(V_SUCCESS, cost, cost, cap, np.inf, -1,
                                   margin, True)
        else:
            st2, _ = _ilp_class_below(sector.H_aug, b_true, sector.w, cap, time_limit)
            if st2 == ST_PROVED:
                assert cap > cost + EPS_TIE
                return _global_out(V_FAIL, cap, np.inf, cost, cost, int(cls),
                                   margin, False)
        if st2 == ST_BUDGET:
            return None                     # timed out: smaller caps are harder
    return None


def _global_out(verdict, lb_true, ub_true, lb_wrong, ub_wrong, wcls, margin,
                true_exact):
    return {"verdict": verdict, "lb_true": float(lb_true), "ub_true": float(ub_true),
            "lb_wrong": float(lb_wrong), "ub_wrong": float(ub_wrong),
            "wrong_witness_class": wcls, "true_exact": true_exact,
            "global_margin": float(margin), "route": "global",
            "nodes_total": 0, "clustered": False, "unified": False}
