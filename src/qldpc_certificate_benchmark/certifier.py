"""Per-shot certification of the maximum-likelihood decision.

For one shot ``(syndrome s, true observables lam)``, the certifier decides
*provably* whether the non-degenerate ML decoder (the minimum-weight solution
of ``H e = s`` under weights ``log((1-p)/p)``) predicts ``lam`` correctly:

``success``
    Every solution in a wrong logical class costs strictly more than the true
    class optimum.  ML decodes this shot correctly.
``fail``
    A solution in a wrong class costs strictly less than every true-class
    solution.  ML decodes this shot incorrectly.
``tie``
    The optima are equal within ``EPS_TIE``.  ML's answer depends on tie-breaking.
``undecided``
    The search budget ran out before a proof was found.

Every proof is exact: an A* search with admissible bounds, LP/ILP relaxations
whose witnesses are re-verified in NumPy, and cluster decompositions with a
certified separation test.  Averaged over shots, the verdicts give a
certified bracket on the ML logical error rate:
``fail/N <= LER_ML <= (fail + tie + undecided)/N``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CertConfig, DEFAULT_CFG
from .decoders import Role, as_decoder
from ._engine.constants import V_SUCCESS, V_FAIL, V_TIE, V_BUDGET
from ._engine.sector import make_sector, pack_class, _witness_eval, hdec_mod2, haug_mod2
from ._engine.verdicts import certify_sector
from ._engine.fastcert import fast_unified, global_route

__all__ = ["Certifier", "Certificate", "VERDICTS", "ROUTES"]

VERDICTS = ("success", "fail", "tie", "undecided")
# Which stage produced the verdict: the cap-free unified pass, the anchored
# full certificate, the large-budget cap-free retry, or the global ILP route.
ROUTES = ("fast", "full", "fast_retry", "global")


@dataclass
class Certificate:
    """The certified verdict for one shot.

    ``lb_true``/``ub_true``: bounds on the cheapest true-class solution cost.
    ``lb_wrong``/``ub_wrong``: bounds on the cheapest wrong-class solution cost.
    Costs are in nats, ``sum_j e_j log((1-p_j)/p_j)``.
    """
    verdict: str
    lb_true: float
    ub_true: float
    lb_wrong: float
    ub_wrong: float
    nodes: int
    escalated: bool
    details: dict = field(repr=False, default_factory=dict)

    @property
    def ml_correct(self):
        """True / False when proven, None when tie or undecided."""
        return {"success": True, "fail": False}.get(self.verdict)

    @property
    def route(self) -> str:
        """The stage that decided the shot (see :data:`ROUTES`)."""
        return self.details.get("route", "full")

    @property
    def code(self) -> int:
        return VERDICTS.index(self.verdict)


class Certifier:
    """Certifies the ML decision on individual shots of a :class:`DecodingProblem`.

    The certifier owns large search scratch buffers.  Make one per process and
    reuse it across shots.
    """

    def __init__(self, problem, config: CertConfig = DEFAULT_CFG):
        self.problem = problem
        self.cfg = config
        H = problem.H
        H_full = np.vstack([H, problem.L])
        self.sector = make_sector(problem.name, H, H_full, H.shape[0],
                                  problem.priors, problem.num_observables,
                                  pool_cap=config.pool_cap, verbose=False)
        # True-class anchor: short BP + OSD-0 (ldpc) on the augmented system
        # [H; L] e = [s; lam].  Any feasible point gives a valid cap, so the
        # anchor affects speed only, never a verdict.
        from ldpc import BpOsdDecoder
        from scipy.sparse import csr_matrix
        from .decoders import _ldpc_channel
        self._anchor_dec = BpOsdDecoder(
            csr_matrix(self.sector.H_aug.astype(np.uint8)),
            error_channel=_ldpc_channel(problem.priors),
            max_iter=config.anchor_bp_iters, bp_method="product_sum",
            schedule=config.anchor_bp_schedule, osd_method="osd0")

    # ------------------------------------------------------------ public API
    def certify(self, syndrome, observables, seeds=None) -> Certificate:
        """Certify the ML decision for one shot.

        Parameters
        ----------
        syndrome : (m,) 0/1 array
        observables : (k,) 0/1 array -- the TRUE logical observables of the shot.
        seeds : iterable of (n,) 0/1 corrections, optional
            Candidate solutions from any decoder.  Each one is re-verified
            against ``H`` and used only to tighten the search.  A seed can never
            make a proof wrong, but a decoder that seeds its own bracket is no
            longer independently benchmarked.
        """
        s, lam_bits = self._prep(syndrome, observables)
        return self._to_certificate(self._run(s, lam_bits, seeds or ()))

    # ------------------------------------------------------------ internals
    def _prep(self, syndrome, observables):
        sec = self.sector
        s = np.ascontiguousarray(np.asarray(syndrome).ravel() & 1, dtype=np.int8)
        lam = np.asarray(observables).ravel().astype(np.int64) & 1
        if s.size != sec.Hdec.shape[0]:
            raise ValueError(f"syndrome has length {s.size}, expected {sec.Hdec.shape[0]}")
        if lam.size != len(sec.powers):
            raise ValueError(f"observables has length {lam.size}, expected "
                             f"{len(sec.powers)}")
        return s, lam

    def _anchor(self, s, lam_bits):
        """Decoder-agnostic true-class anchor: BP -> OSD-0 on [H; L] = [s; lam].
        Its cost is a valid cap on the true-class optimum."""
        sec = self.sector
        b_true = np.concatenate([s, lam_bits.astype(np.int8)]).astype(np.uint8)
        E = np.asarray(self._anchor_dec.decode(b_true), dtype=np.int64)
        if not np.array_equal(haug_mod2(sec, E), b_true.astype(np.int64)):
            raise RuntimeError("anchor solution violates [H; L] e = [s; lam]: the "
                               "syndrome/observables are inconsistent with the model")
        return float(sec.w @ E), []

    def _fold_seeds(self, s, lam_bits, cap_true, seed_wrong, seeds):
        if not self.cfg.use_seeds:
            return cap_true, seed_wrong
        ell_true = pack_class(lam_bits, self.sector.powers)
        for corr in seeds:
            corr = np.asarray(corr).ravel().astype(np.int64) & 1
            if not np.array_equal(hdec_mod2(self.sector, corr), s.astype(np.int64)):
                continue                      # not a coset member: ignore
            cost, cls = _witness_eval(self.sector, s, corr, verify=True)
            if cls == ell_true:
                cap_true = min(cap_true, cost)
            else:
                seed_wrong.append(corr)
        return cap_true, seed_wrong

    def _run(self, s, lam_bits, seeds):
        """Fast pass -> anchored full certificate (+ escalation) -> last resorts.

        Every stage is exact on its own, so the order only affects speed:
        ``route`` records which stage decided the shot."""
        cfg, sec = self.cfg, self.sector
        if cfg.use_fast_pass:
            out = fast_unified(sec, s, lam_bits, cfg.unified_node_budget)
            if out is not None:
                out["escalations"] = 0
                return out
        cap_true, seed_wrong = self._anchor(s, lam_bits)
        cap_true, seed_wrong = self._fold_seeds(s, lam_bits, cap_true, seed_wrong,
                                                seeds)
        cert = certify_sector(sec, s, lam_bits, cap_true,
                              seed_wrong=seed_wrong, cfg=cfg)
        cert["escalations"] = 0
        if cfg.escalate and cert["verdict"] == V_BUDGET:
            close_success = cert["true_exact"] and cert["esc_margin"] <= cfg.esc_margin
            close_fail = (not cert["true_exact"]
                          and cert["ub_wrong"] - cert["lb_true"] <= cfg.esc_margin)
            if close_success or close_fail:
                cert = certify_sector(sec, s, lam_bits, cap_true,
                                      budget_scale=cfg.escalate_factor,
                                      seed_wrong=seed_wrong, cfg=cfg)
                cert["escalations"] = 1
        cert.setdefault("route", "full")
        if cert["verdict"] != V_BUDGET:
            return cert
        nodes, esc = int(cert["nodes_total"]), cert["escalations"]
        out = None
        if cfg.fast_retry_node_budget:
            out = fast_unified(sec, s, lam_bits, cfg.fast_retry_node_budget)
            if out is not None:
                out["route"] = "fast_retry"
        if out is None and cfg.use_global_route:
            out = global_route(sec, s, lam_bits, cfg.global_time_limit)
        if out is None:
            return cert
        out["nodes_total"] = nodes + int(out["nodes_total"])
        out["escalations"] = esc
        return out

    @staticmethod
    def _to_certificate(cert):
        v = {V_SUCCESS: "success", V_FAIL: "fail", V_TIE: "tie",
             V_BUDGET: "undecided"}[cert["verdict"]]
        return Certificate(
            verdict=v, lb_true=float(cert["lb_true"]), ub_true=float(cert["ub_true"]),
            lb_wrong=float(cert["lb_wrong"]), ub_wrong=float(cert["ub_wrong"]),
            nodes=int(cert["nodes_total"]), escalated=bool(cert["escalations"]),
            details=cert)

    def _decode_and_certify(self, s, lam_bits, decoders, shot_idx):
        """Benchmark path: run every decoder, seed ONLY from SEED-role decoders,
        certify.  A PROBE's answer never reaches the bracket (structural
        independence: the seed loop iterates only SEED specs)."""
        sec = self.sector
        ell_true = pack_class(lam_bits, sec.powers)
        results = {d.name: d._run(sec, s, shot_idx) for d in decoders}
        seeds = [results[d.name].correction for d in decoders
                 if d.role is Role.SEED and results[d.name].correction is not None]
        cert = self._run(s, lam_bits, seeds)
        out = {
            "verdict": int(cert["verdict"]),
            "nodes": int(cert["nodes_total"]),
            "escalated": bool(cert["escalations"]),
            "clustered": bool(cert.get("clustered", False)),
            "route": ROUTES.index(cert.get("route", "full")),
            "lb_true": float(cert["lb_true"]), "ub_true": float(cert["ub_true"]),
            "lb_wrong": float(cert["lb_wrong"]), "ub_wrong": float(cert["ub_wrong"]),
        }
        for name, r in results.items():
            ok = (r.ell == ell_true) if (r.ran and r.ell is not None) else False
            out[f"ok:{name}"] = bool(ok)
            out[f"ran:{name}"] = bool(r.ran)
            out[f"lowconf:{name}"] = bool(r.low_confidence)
            out[f"invalid:{name}"] = bool(r.invalid)
        return out

    def setup_decoders(self, decoders):
        decs = [as_decoder(d) for d in decoders]
        for d in decs:
            d._setup(self.problem, self.sector)
        return decs
