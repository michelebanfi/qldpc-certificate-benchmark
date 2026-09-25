"""Soundness constants and status/verdict enums for the certification engine.

These are NOT user knobs -- they are either baked into the Numba kernels at
compile time (``MAX_DEPTH``, ``ST_*``, ``DEC_*``) or fixed soundness margins on
which the certificate proofs rely (``EPS_TIE``, ``BAN_SLACK``, ``CLUSTER_SLACK``).
User-tunable parameters live in :class:`qldpc_certificate_benchmark.CertConfig`.

If you change any value here, clear the cached kernels
(delete the ``__pycache__/*.nbc`` files next to ``kernels.py``): the ``ST_*``/``DEC_*``/``MAX_DEPTH``
values are frozen into the compiled ``kernels.py`` and Numba will not detect the
change on its own.
"""

# --- A* search depth guard (baked into the kernels; also sizes CertEngine
#     scratch arrays).  >> cap / min_w for every real instance.
MAX_DEPTH = 64

# --- Strict-margin epsilon on every certifying comparison.  Passed as a kernel
#     argument, but a fixed soundness epsilon -- never a user knob.
EPS_TIE = 1e-9

# --- Column-ban safety slack (nats).  Must exceed EPS_TIE so every verdict
#     comparison (which sits at cap + EPS_TIE) stays inside the ban margin.
BAN_SLACK = 1e-6

# --- Cluster separation-test merge-side slack (nats).  Covers float rounding,
#     errs toward merging (= always sound).  Equal to BAN_SLACK so the cluster
#     path's cap+slack clamps stay valid.
CLUSTER_SLACK = 1e-6

# --- certified_min_core engine statuses (baked as kernel return literals)
ST_MIN = 0      # exact minimum found (value == lb)
ST_PROVED = 1   # proved min > cap + eps (lb = +inf -> system infeasible)
ST_BUDGET = 2   # node/pool budget exhausted (lb = certified one-sided bound)

# --- certified_decide_core (unified decider) statuses (baked kernel literals)
DEC_OK = 0       # frontier cleared: the (class_g1, m_*_hat) decision is valid
DEC_BUDGET = 2   # inconclusive: node/pool budget hit before the frontier cleared

# --- Sector verdicts
V_SUCCESS, V_FAIL, V_TIE, V_BUDGET = 0, 1, 2, 3
VERDICT_NAMES = ("success", "fail", "tie-uncert", "budget-uncert")
