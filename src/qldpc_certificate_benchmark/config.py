"""Certifier configuration.

:class:`CertConfig` is a frozen dataclass holding every *user-tunable* knob of
the certification engine: search budgets, escalation, the bracket-tightening
layers, and solver time limits.  None of these knobs can make a certificate
wrong.  They only trade wall-time against the fraction of shots left
*uncertified* (``budget-uncert``).

Soundness constants (``EPS_TIE``, ``MAX_DEPTH``, the slacks) are not knobs.
They live in ``qldpc_certificate_benchmark._engine.constants``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class CertConfig:
    # --- internal true-class anchor ----------------------------------------
    # A short BP followed by OSD-0 (ldpc) on [H; L] gives a feasible member of
    # the true logical class.  Its cost caps every search.  It uses no
    # benchmarked decoder's output, so the bracket stays independent of the
    # decoder under test.
    anchor_bp_iters: int = 3
    anchor_bp_schedule: str = "serial"          # "serial" | "parallel"

    # --- cheap first pass ----------------------------------------------------
    # Before the anchor and the 1-true + k-wrong decomposition, try the unified
    # single-coset search with NO true-class cap (budget
    # ``unified_node_budget``).  Exact: the kernel tightens the cap to the
    # global minimum at its first goal.  It decides most shots in a few ms (vs
    # ~100 ms for the decomposition on [[144,12,12]] circuit noise); when it is
    # inconclusive the full certificate runs as before.
    use_fast_pass: bool = True

    # --- certified A* search budgets (nodes) --------------------------------
    true_node_budget: int = 50_000
    wrong_node_budget: int = 25_000
    pool_cap: int = 1_000_000
    unified_node_budget: int = 60_000
    cluster_node_budget: int = 30_000

    # --- escalation ---------------------------------------------------------
    # A shot left BUDGET-uncertified but close to a decision (within
    # ``esc_margin`` nats) is re-run once with every budget x escalate_factor.
    escalate: bool = True
    escalate_factor: int = 10
    esc_margin: float = 8.0

    # --- last resorts for shots the full certificate left undecided -------
    # ``fast_retry_node_budget``: re-run the cap-free unified search with this
    # many nodes (``pool_cap`` must be ~10-20x larger: pushes outnumber
    # expansions).  ``None`` disables it.
    fast_retry_node_budget: int | None = None
    # ``use_global_route``: exact global ILPs -- a syndrome-only ILP witness,
    # then one infeasibility proof that no competing class costs <= its cost +
    # margin (margins 0.1 .. 1e-6).  No MIP optimality claim is trusted.
    # Expensive (~1-2 min per hard d=12 sector), so off by default.
    use_global_route: bool = False
    global_time_limit: float = 60.0

    # --- bracket-tightening layers (all admissible / re-verified) -----------
    use_lp_screen: bool = True
    use_seeds: bool = True
    use_milp_fallback: bool = True
    use_adaptive_lp: bool = True
    use_unified_search: bool = False
    use_clusters: bool = True
    use_dual_bounds: bool = True
    use_cut_recycle: bool = True
    use_exact_ilp: bool = True
    # Targeted re-certification: only try to PROVE FAIL (from a verified
    # wrong-class witness + a true-class lower bound).  Leave off for sweeps.
    fail_only: bool = False

    # --- solver knobs (HiGHS via scipy) --------------------------------------
    milp_time_limit: float = 2.0
    adaptive_lp_rounds: int = 12
    exact_ilp_time_limit: float = 2.0
    exact_ilp_max_calls: int = 30

    def __post_init__(self):
        for name in ("true_node_budget", "wrong_node_budget", "pool_cap",
                     "unified_node_budget", "cluster_node_budget",
                     "exact_ilp_max_calls", "adaptive_lp_rounds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.anchor_bp_iters < 1:
            raise ValueError("anchor_bp_iters must be >= 1")
        if self.anchor_bp_schedule not in ("serial", "parallel"):
            raise ValueError(f"anchor_bp_schedule must be 'serial' or 'parallel', "
                             f"got {self.anchor_bp_schedule!r}")
        if self.escalate_factor < 1:
            raise ValueError(f"escalate_factor must be >= 1, got {self.escalate_factor!r}")
        if self.esc_margin < 0:
            raise ValueError(f"esc_margin must be >= 0, got {self.esc_margin!r}")
        if self.fast_retry_node_budget is not None and self.fast_retry_node_budget <= 0:
            raise ValueError(f"fast_retry_node_budget must be positive or None, "
                             f"got {self.fast_retry_node_budget!r}")
        if (self.milp_time_limit <= 0 or self.exact_ilp_time_limit <= 0
                or self.global_time_limit <= 0):
            raise ValueError("solver time limits must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CertConfig":
        return cls(**d)

    def replace(self, **changes) -> "CertConfig":
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)


DEFAULT_CFG = CertConfig()
