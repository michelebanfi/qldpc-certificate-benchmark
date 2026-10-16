"""LP / ILP relaxations and exact integer solves (scipy HiGHS).

Every routine here returns an *admissible* bound or a re-verifiable witness:
- ``_lp_root_bound`` / ``_adaptive_lp``: lower bounds on the coset minimum
  (the adaptive parity-cut LP's integral optima ARE the exact minimum);
- ``_milp_lower_bound``: integer-z relaxation lower bound;
- ``_ilp_exact``: full integer program -- an exact GF(2)-coset minimum.

The ``time_limit`` / ``max_rounds`` defaults below match the historical module
constants; the runner threads the live ``CertConfig`` values in explicitly.

HiGHS presolve gotcha (see ``_cut_constraint``): models mixing -inf inequality
bounds with fractional upper bounds on integer variables can return WRONG
status-0 optima.  Integer bounds are floored and cut rows given finite lower
bounds to avoid it -- keep it that way.
"""

import numpy as np
from scipy.optimize import milp as _sp_milp, LinearConstraint as _LC, Bounds as _Bounds
from scipy.sparse import csr_matrix, eye as _speye, hstack as _sp_hstack, vstack as _sp_vstack

from .constants import ST_MIN, ST_PROVED, ST_BUDGET

# A status-0 MIP whose dual bound trails the primal by more than this is treated
# as unproved (see _ilp_exact).  Float round-off alone leaves ~1e-15; 1e-10 is 10x
# below EPS_TIE, so an optimum overstated by at most this much still leaves
# every strict EPS_TIE-margin verdict true.
_GAP_TOL = 1e-10


# --- max-packing LP (the covering dual of every shot) -------------------------
# The covering dual is solved once per shot and is small (~10-50 variables), so
# linprog(method="highs") spends 50-70% of it in scipy's wrapper (input checks,
# per-option validation).  _max_packing hands HiGHS the model linprog builds,
# with the options linprog sets, and applies linprog's validity check: the
# solution is bit-identical, 2-3x faster.  It uses scipy's vendored highspy
# (scipy >= 1.15, a private module), so it is self-checked against linprog once
# per process; linprog is used whenever that module is missing or misbehaves.
# Soundness never depends on it: callers re-verify the returned vector.
_HIGHS = None        # None = not probed yet; False = use linprog; else (module, options)
_LINPROG_TOL = np.sqrt(1e-9) * 10    # linprog's _check_result tolerance


def _max_packing_linprog(A, b):
    from scipy.optimize import linprog
    try:
        res = linprog(-np.ones(A.shape[1]), A_ub=A, b_ub=b, bounds=(0, None),
                      method="highs")
    except Exception:
        return None
    if not res.success or res.x is None:
        return None
    return np.asarray(res.x, dtype=np.float64)


def _max_packing_highs(h, opts, A, b):
    n_rows, n = A.shape
    cols, rows = np.nonzero(A.T)               # column-major order = CSC of A
    start = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(np.bincount(cols, minlength=n), out=start[1:])
    highs = h._Highs()
    if highs.passOptions(opts) == h.HighsStatus.kError:
        return None
    # flat passModel(num_col, num_row, num_nz, a_format=colwise, sense=minimize,
    #                offset, cost, col_lower, col_upper, row_lower, row_upper,
    #                a_start, a_index, a_value, integrality)
    if highs.passModel(n, n_rows, int(rows.size), 1, 1, 0.0,
                       np.full(n, -1.0), np.zeros(n), np.full(n, np.inf),
                       np.full(n_rows, -np.inf), b, start, rows.astype(np.int32),
                       A.T[A.T != 0], np.zeros(n, dtype=np.int32)) \
            == h.HighsStatus.kError:
        return None
    if highs.run() == h.HighsStatus.kError:
        return None
    if highs.getModelStatus() != h.HighsModelStatus.kOptimal:
        return None
    sol = highs.getSolution()
    x = np.array(sol.col_value, dtype=np.float64)
    slack = b - np.array(sol.row_value, dtype=np.float64)
    fun = highs.getInfo().objective_function_value
    if (np.isnan(x).any() or np.isnan(fun) or np.isnan(slack).any()
            or not np.all(x >= -_LINPROG_TOL) or (slack < -_LINPROG_TOL).any()):
        return None                            # where linprog reports status 4
    return x


def _highs_direct():
    global _HIGHS
    if _HIGHS is None:
        _HIGHS = False
        try:
            from scipy.optimize._highspy import _core as h
            opts = h.HighsOptions()
            opts.presolve = "on"
            opts.highs_debug_level = 0
            opts.log_to_console = False
            opts.output_flag = False
            opts.simplex_strategy = 1          # dual simplex, as linprog sets it
            # degenerate (many optima) check LP: same vertex as linprog or bust
            A = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1],
                          [1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
            A = np.asfortranarray(A)
            b = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 0.7, 0.9])
            ref = _max_packing_linprog(A, b)
            got = _max_packing_highs(h, opts, A, b)
            if ref is not None and got is not None and np.array_equal(ref, got):
                _HIGHS = (h, opts)
        except Exception:
            pass
    return _HIGHS


def _max_packing(A, b):
    """max sum(x) s.t. A x <= b, x >= 0, for a 0/1 matrix A (dense float64).

    Returns x, or None when the LP is not solved to optimality -- exactly the
    result of linprog(-1, A_ub=A, b_ub=b, bounds=(0, None), method="highs")."""
    direct = _highs_direct()
    if direct:
        try:
            return _max_packing_highs(*direct, A, b)
        except Exception:
            pass
    return _max_packing_linprog(A, b)


def _lp_root_bound(H_aug, row_active, b, w):
    """LP-relaxation lower bound on min{w^T x : A x = b (mod 2), x in {0,1}^n}.

    Encoding: replace each GF(2) row  Σ_j A_ij x_j ≡ b_i (mod 2)  with the
    equality  Σ_j A_ij x_j - 2 z_i = b_i  and relax x in {0,1} to x in [0,1],
    z_i >= 0 (continuous).  This is an admissible bound: every binary feasible
    point is LP-feasible, so LP_optimal <= binary_optimal.

    Returns LP optimal (float), or +inf when the LP is infeasible (implies the
    binary problem is also infeasible).  Returns 0.0 on unexpected solver error
    (safe: weights are non-negative so 0 is always an under-estimate).
    """
    mask = row_active.astype(bool)
    A_act = H_aug[mask].astype(np.float64)   # R x n
    b_act = b[mask].astype(np.float64)
    R, n = A_act.shape
    if R == 0:
        return 0.0

    # z_i upper bound: Σ x_j <= row_weight, so z_i = (Σ x_j - b_i)/2 <= row_weight/2
    max_z = A_act.sum(axis=1) / 2.0
    c = np.concatenate([w.astype(np.float64), np.zeros(R)])

    # Constraint matrix [A | -2I] (x-block sparse LDPC; z-block diagonal)
    A_lp = _sp_hstack(
        [csr_matrix(A_act), _speye(R, format="csr") * (-2.0)], format="csr"
    )
    try:
        res = _sp_milp(
            c,
            constraints=_LC(A_lp, b_act, b_act),
            bounds=_Bounds(
                lb=np.concatenate([np.zeros(n), np.zeros(R)]),
                ub=np.concatenate([np.ones(n), max_z]),
            ),
        )
        return float(res.fun) if res.success else np.inf
    except Exception:
        return 0.0   # safe under-estimate; never corrupts a certificate


def _cut_constraint(cuts, keep, n_x, R):
    """Parity cuts (full column space) as a LinearConstraint over the (x, z)
    variable block of the [A | -2I] encoding.

    Entries on columns outside `keep` are dropped with the rhs unchanged: the
    restricted problem only contains binary points with those variables at 0,
    where the dropped terms contribute nothing -- the truncated inequality is
    the original one evaluated on exactly those points, hence still valid.
    (In practice screen and ILP share the same `keep`, so nothing drops.)

    HiGHS GOTCHA: the cut rows get FINITE lower bounds (the exact minimum of
    each row's LHS over the box: the sum of its negative coefficients).  The
    vendored HiGHS presolve returns a WRONG claimed-optimal on models mixing
    -inf inequality bounds with fractional upper bounds on integer variables
    (caught by --selftest against brute force); finite bounds -- combined with
    the floored z bounds below -- avoid the trigger without changing the
    binary feasible set."""
    rows, cols_full, data, rhs, n_cuts = cuts
    if keep is not None:
        pos = np.searchsorted(keep, cols_full)
        pos_c = np.minimum(pos, keep.size - 1)
        ok = keep[pos_c] == cols_full
        rows, data, cols = rows[ok], data[ok], pos_c[ok]
    else:
        cols = cols_full
    A_cut = csr_matrix((data, (rows, cols)), shape=(int(n_cuts), n_x + R))
    lb = np.bincount(rows.astype(np.int64), weights=np.minimum(data, 0.0),
                     minlength=int(n_cuts))
    return _LC(A_cut, lb, rhs)


def _milp_lower_bound(H_aug, row_active, b, w, time_limit=2.0,
                      keep=None, cuts=None):
    """Integer-z MILP lower bound on min{w^T x : A x = b (mod 2), x in {0,1}^n}.

    Same [A | -2I] encoding as _lp_root_bound but with z constrained INTEGER
    (x stays continuous in [0,1]).  Forcing z integer makes A x land on an
    integer of the correct parity, which reactivates the even-parity (b_i=0)
    rows the continuous LP drops -- a strictly tighter relaxation, and HiGHS
    layers its own cuts on top.  Still admissible (every binary x is feasible),
    so the result is always a valid lower bound on the binary coset minimum.

    Returns (bound, exact).  status 0 -> proved optimal (bound = exact min,
    exact=True).  Time/iteration limit -> mip_dual_bound, the CERTIFIED lower
    bound (never the incumbent res.fun, which is an upper bound), exact=False.
    Infeasible -> (+inf, True): the coset is empty.  Solver error / no usable
    bound -> (0.0, False): a safe under-estimate that never corrupts a cert.
    """
    mask = row_active.astype(bool)
    A_act = H_aug[mask].astype(np.float64)
    if keep is not None:
        A_act = np.ascontiguousarray(A_act[:, keep])
    b_act = b[mask].astype(np.float64)
    R, n = A_act.shape
    if R == 0:
        return 0.0, True
    # z = (Ax - b)/2 is integral for every binary-feasible point, so flooring
    # its bound is exact -- and fractional upper bounds on integer variables
    # are one half of a HiGHS presolve wrong-optimal trigger (see
    # _cut_constraint), so they must not be emitted.
    max_z = np.floor(A_act.sum(axis=1) / 2.0)
    c = np.concatenate([(w if keep is None else w[keep]).astype(np.float64),
                        np.zeros(R)])
    A_lp = _sp_hstack(
        [csr_matrix(A_act), _speye(R, format="csr") * (-2.0)], format="csr")
    integrality = np.concatenate([np.zeros(n), np.ones(R)])  # x cont., z integer
    constraints = [_LC(A_lp, b_act, b_act)]
    if cuts is not None:
        constraints.append(_cut_constraint(cuts, keep, n, R))
    try:
        res = _sp_milp(
            c,
            constraints=constraints,
            bounds=_Bounds(
                lb=np.concatenate([np.zeros(n), np.zeros(R)]),
                ub=np.concatenate([np.ones(n), max_z]),
            ),
            integrality=integrality,
            # HiGHS otherwise stops at a 1e-4 RELATIVE gap and still reports
            # status 0; res.fun is then an upper bound, not the minimum.
            options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
        )
    except Exception:
        return 0.0, False
    if res.status == 0:                 # proved optimal
        # the proved bound is the dual bound; it equals res.fun when HiGHS
        # closed the gap exactly (the normal case) and is the safe side if not
        db = getattr(res, "mip_dual_bound", None)
        if db is not None and np.isfinite(db) and db < float(res.fun) - _GAP_TOL:
            return float(db), False
        return float(res.fun), True
    if res.status == 2:                 # infeasible -> empty coset
        return np.inf, True
    db = getattr(res, "mip_dual_bound", None)   # time/iteration limit hit
    if db is None or not np.isfinite(db):
        return 0.0, False
    return float(db), False


def _adaptive_lp(H_aug, row_active, b, w, max_rounds=12,
                 cut_tol=1e-6, keep=None):
    """Feldman LP-decoding lower bound via adaptively separated parity cuts.

    Relaxes x in {0,1}^n to [0,1]^n (no auxiliary variables) and repeatedly adds
    the most-violated valid forbidden-set inequality for each active parity row
    until none is violated (or max_rounds).  The returned bound is an admissible
    LOWER bound on min{w^T x : A x = b (mod 2), x in {0,1}^n}; +inf means the LP
    (hence the binary coset) is infeasible.

    keep: optional int64 index array of columns to solve over (the caller's
    banned columns removed).  Because every banned column is proven absent from
    all solutions any verdict compares against (cost > cap_true + BAN_SLACK,
    while every verdict margin sits at cap_true + eps or below), the restricted
    bound is valid wherever the unrestricted one is used.

    Returns (bound, witness, cuts).  witness is the support (int64, FULL column
    space) of the LP optimum iff that optimum is integral AND binary-feasible
    on the active rows -- the EXACT coset minimum, re-verified by the caller.
    cuts is (rows, cols_full, data, rhs, n_cuts) -- the separated forbidden-set
    inequalities in full column space, valid for the binary coset, for recycling
    into the exact-ILP solve of the same system -- or None if none were added.
    """
    from scipy.optimize import linprog
    mask = row_active.astype(bool)
    A_act = np.ascontiguousarray(H_aug[mask])
    if keep is not None:
        A_act = np.ascontiguousarray(A_act[:, keep])
    b_act = (b[mask].astype(np.int64) & 1)
    R, n = A_act.shape
    if R == 0:
        return 0.0, np.empty(0, dtype=np.int64), None
    supports = [np.flatnonzero(A_act[i]).astype(np.int64) for i in range(R)]
    c = np.ascontiguousarray(w if keep is None else w[keep], dtype=np.float64)
    bnds = [(0.0, 1.0)] * n
    rows_i, cols_i, data, rhs = [], [], [], []
    n_cuts = 0
    bound = 0.0
    x = np.zeros(n)
    for _ in range(max_rounds):
        A_ub = (csr_matrix((data, (rows_i, cols_i)), shape=(n_cuts, n))
                if n_cuts else None)
        b_ub = np.asarray(rhs, dtype=np.float64) if n_cuts else None

        def _cuts_out():
            if n_cuts == 0:
                return None
            cols_full = (np.asarray(cols_i, dtype=np.int64) if keep is None
                         else keep[np.asarray(cols_i, dtype=np.int64)])
            return (np.asarray(rows_i, dtype=np.int64), cols_full,
                    np.asarray(data, dtype=np.float64),
                    np.asarray(rhs, dtype=np.float64), n_cuts)

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bnds, method="highs")
        except Exception:
            return bound, None, _cuts_out()  # keep last valid (admissible) bound
        if not res.success:
            # HiGHS status 2 == infeasible: no LP point, so no binary point
            # either (all cuts are valid) -> the coset is empty.
            if getattr(res, "status", None) == 2:
                return np.inf, None, _cuts_out()
            return bound, None, _cuts_out()
        x = res.x
        bound = float(res.fun)
        added = 0
        for i in range(R):
            S = supports[i]
            if S.size == 0:
                continue
            bi = int(b_act[i])
            xs = x[S]
            in_T = xs > 0.5
            if (int(in_T.sum()) & 1) == bi:
                # parity of T matches b_i -> not a forbidden set; flip the bit
                # closest to 0.5 (smallest violation loss) to make it wrong-parity
                in_T[int(np.argmin(np.abs(xs - 0.5)))] ^= True
            T = S[in_T]
            notT = S[~in_T]
            viol = float(xs[in_T].sum() - xs[~in_T].sum() - (T.size - 1))
            if viol > cut_tol:
                # VALIDITY GUARD: the inequality is valid for the binary coset
                # iff |T| has the wrong parity for this row.  This protects the
                # certificate -- an invalid cut could lift the LB above the true
                # minimum and falsely certify SUCCESS.
                assert (T.size & 1) != bi, "adaptive-LP: invalid parity cut"
                for j in T:
                    rows_i.append(n_cuts); cols_i.append(int(j)); data.append(1.0)
                for j in notT:
                    rows_i.append(n_cuts); cols_i.append(int(j)); data.append(-1.0)
                rhs.append(float(T.size - 1))
                n_cuts += 1
                added += 1
        if added == 0:
            break
    witness = None
    xr = np.rint(x)
    if np.all(np.abs(x - xr) < 1e-6):
        xb = xr.astype(np.int64)
        # parity of every active row from its support (no dense matvec)
        par = np.array([int(xb[s].sum()) & 1 for s in supports], dtype=np.int64)
        if np.array_equal(par, b_act):
            witness = np.flatnonzero(xb).astype(np.int64)
            if keep is not None:
                witness = keep[witness]
    cols_full = (np.asarray(cols_i, dtype=np.int64) if keep is None
                 else keep[np.asarray(cols_i, dtype=np.int64)])
    cuts = ((np.asarray(rows_i, dtype=np.int64), cols_full,
             np.asarray(data, dtype=np.float64),
             np.asarray(rhs, dtype=np.float64), n_cuts)
            if n_cuts else None)
    return bound, witness, cuts


def _ilp_exact(H_aug, row_active, b, w, time_limit=2.0,
               keep=None, cuts=None):
    """Exact integer program for a GF(2) coset minimum (x AND z integer).

    keep restricts to non-banned columns (sound wherever the result is used --
    see _adaptive_lp); cuts recycles the adaptive LP's asserted-valid parity
    cuts, which never change the binary feasible set, so status-0 minima and
    dual bounds keep their exact meaning.  Witnesses are returned in FULL
    column space.

    Same [A | -2I] encoding as _milp_lower_bound but with BOTH x binary and z
    non-negative integer.  A x - 2 z = b with x in {0,1}, z integer >= 0 is an
    exact bijection with {x in {0,1}^n : A x = b (mod 2)} (z = (Ax-b)/2 is forced
    non-negative because Ax and b share parity), so this is the exact coset
    problem, not a relaxation.

    Returns (status, value, lb, witness):
      ST_MIN    : proved optimal.  value == lb == exact minimum; witness = its
                  support (int64), re-verified by the caller.
      ST_PROVED : infeasible coset.  value = lb = +inf, witness = None.
      ST_BUDGET : time/iteration limit.  lb = mip_dual_bound (certified LOWER
                  bound); value = incumbent objective (a valid UPPER bound) or
                  +inf; witness = the incumbent's support (re-verified) or None.
    """
    mask = row_active.astype(bool)
    A_act = H_aug[mask].astype(np.float64)
    if keep is not None:
        A_act = np.ascontiguousarray(A_act[:, keep])
    b_act = b[mask].astype(np.float64)
    R, n = A_act.shape
    if R == 0:
        return ST_MIN, 0.0, 0.0, np.empty(0, dtype=np.int64)
    w_eff = np.asarray(w if keep is None else w[keep], dtype=np.float64)
    # floored: exact for integral z, avoids the HiGHS presolve trigger
    # documented in _cut_constraint
    max_z = np.floor(A_act.sum(axis=1) / 2.0)
    c = np.concatenate([w_eff, np.zeros(R)])
    A_lp = _sp_hstack(
        [csr_matrix(A_act), _speye(R, format="csr") * (-2.0)], format="csr")
    integrality = np.ones(n + R)          # x binary AND z integer
    constraints = [_LC(A_lp, b_act, b_act)]
    if cuts is not None:
        constraints.append(_cut_constraint(cuts, keep, n, R))
    try:
        res = _sp_milp(
            c, constraints=constraints,
            bounds=_Bounds(lb=np.zeros(n + R),
                           ub=np.concatenate([np.ones(n), max_z])),
            integrality=integrality,
            # exact gap: at HiGHS's default 1e-4 relative gap, status 0 would
            # not mean the incumbent is the minimum (see _milp_lower_bound)
            options={"time_limit": float(time_limit), "mip_rel_gap": 0.0})
    except Exception:
        return ST_BUDGET, np.inf, 0.0, None
    if res.status == 2:                    # infeasible -> empty coset
        return ST_PROVED, np.inf, np.inf, None
    # Extract an integral incumbent witness (support of the x-block) if present.
    witness, ub_val = None, np.inf
    if res.x is not None and np.all(np.isfinite(res.x)):
        xb = np.rint(res.x[:n]).astype(np.int64)
        if np.all((xb == 0) | (xb == 1)):
            witness = np.flatnonzero(xb).astype(np.int64)
            ub_val = float(w_eff @ xb)
            if keep is not None:
                witness = keep[witness]
    db = getattr(res, "mip_dual_bound", None)
    if res.status == 0:                    # proved optimal
        if db is not None and np.isfinite(db) and db < float(res.fun) - _GAP_TOL:
            # "optimal" only up to HiGHS's absolute gap tolerance: keep what is
            # proved -- the dual lower bound and the verified incumbent
            return ST_BUDGET, ub_val, float(db), witness
        return ST_MIN, float(res.fun), float(res.fun), witness
    # time/iteration limit hit
    lb = float(db) if (db is not None and np.isfinite(db)) else 0.0
    return ST_BUDGET, ub_val, lb, witness


def _ilp_any_wrong_below(Hdec, H_log, syndrome, true_logical, w, cap,
                         time_limit=60.0):
    """Decide whether *any* wrong logical class has cost at most ``cap``.

    This is one exact integer feasibility problem rather than one solve for
    each of the 2**k - 1 wrong classes.  Binary ``y`` is the logical class of
    ``x``: H_log x - 2q - y = 0.  The last row requires at least one bit of y
    to differ from ``true_logical``.  Status ``ST_PROVED`` means infeasible,
    hence every wrong class costs strictly more than ``cap``; ``ST_MIN``
    supplies a re-verified wrong-class witness at or below it.  A timeout or
    solver error remains ``ST_BUDGET``.

    All auxiliary parity-variable bounds are integral, and every inequality
    has a finite lower bound, preserving the HiGHS soundness safeguard used
    by ``_ilp_exact`` and ``_cut_constraint``.
    """
    Hdec = np.asarray(Hdec, dtype=np.int8)
    H_log = np.asarray(H_log, dtype=np.int8)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    true_logical = np.asarray(true_logical, dtype=np.int8)
    w = np.asarray(w, dtype=np.float64)
    m, n = Hdec.shape
    k = H_log.shape[0]
    assert H_log.shape[1] == n and syndrome.shape == (m,)
    assert true_logical.shape == (k,) and w.shape == (n,)
    assert np.all((Hdec == 0) | (Hdec == 1))
    assert np.all((H_log == 0) | (H_log == 1))
    assert np.all((syndrome == 0) | (syndrome == 1))
    assert np.all((true_logical == 0) | (true_logical == 1))
    assert np.all(w >= 0) and cap >= 0
    zero = lambda a, b: csr_matrix((a, b), dtype=np.float64)
    det = _sp_hstack([csr_matrix(Hdec, dtype=np.float64),
                      -2 * _speye(m, format="csr"), zero(m, k), zero(m, k)],
                     format="csr")
    log = _sp_hstack([csr_matrix(H_log, dtype=np.float64), zero(k, m),
                      -2 * _speye(k, format="csr"),
                      -_speye(k, format="csr")], format="csr")
    signs = (1 - 2 * true_logical).astype(np.float64)
    differ = _sp_hstack([zero(1, n), zero(1, m), zero(1, k),
                         csr_matrix(signs.reshape(1, -1))], format="csr")
    cost = _sp_hstack([csr_matrix(w.reshape(1, -1)), zero(1, m),
                       zero(1, k), zero(1, k)], format="csr")
    A = _sp_vstack([det, log, differ, cost], format="csr")
    ones_true = int(true_logical.sum())
    lower = np.r_[syndrome, np.zeros(k), 1 - ones_true, 0.0]
    upper = np.r_[syndrome, np.zeros(k), k - ones_true, float(cap)]
    zmax = np.floor(Hdec.sum(axis=1) / 2.0)
    qmax = np.floor(H_log.sum(axis=1) / 2.0)
    var_upper = np.r_[np.ones(n), zmax, qmax, np.ones(k)]
    nvar = n + m + 2 * k
    try:
        res = _sp_milp(np.zeros(nvar), integrality=np.ones(nvar),
                       bounds=_Bounds(np.zeros(nvar), var_upper),
                       constraints=_LC(A, lower, upper),
                       options={"time_limit": float(time_limit),
                                "mip_rel_gap": 0.0})
    except Exception:
        return ST_BUDGET, None
    if res.status == 2:
        return ST_PROVED, None
    if res.x is not None and np.all(np.isfinite(res.x)):
        x = np.rint(res.x[:n]).astype(np.int8)
        if (np.all((x == 0) | (x == 1))
                and np.array_equal(csr_matrix(Hdec, dtype=np.int64)
                                   @ x.astype(np.int64) % 2, syndrome)
                and not np.array_equal(np.asarray(H_log, dtype=np.int64)
                                       @ x.astype(np.int64) % 2, true_logical)
                and float(w @ x) <= cap + 1e-7):
            return ST_MIN, np.flatnonzero(x).astype(np.int64)
    return ST_BUDGET, None
