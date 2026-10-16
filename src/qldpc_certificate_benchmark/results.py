"""Benchmark results: certified ML bracket, decoder LERs, paired tests.

Confidence bounds are exact Clopper-Pearson limits.  :meth:`BenchmarkResult.ml_bracket`
bounds the population ML logical error rate from one run;
:func:`population_bracket` pools several runs of the same problem -- plain runs
and fault-count-band runs (``benchmark(..., fault_band=(lo, hi))``) -- into one
joint bracket.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.stats import beta as _beta, binomtest as _binomtest

from ._engine.constants import V_SUCCESS, V_FAIL, V_TIE, V_BUDGET
from .certifier import ROUTES

__all__ = ["BenchmarkResult", "clopper_pearson", "clopper_pearson_lower",
           "clopper_pearson_upper", "population_bracket"]


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided (1-alpha) binomial confidence interval for k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def clopper_pearson_lower(k, n, alpha=0.05):
    """One-sided (1-alpha) Clopper-Pearson lower limit: P(theta >= limit) >= 1-alpha."""
    if n == 0:
        return 0.0
    return float(_beta.ppf(alpha, k, n - k + 1)) if k > 0 else 0.0


def clopper_pearson_upper(k, n, alpha=0.05):
    """One-sided (1-alpha) Clopper-Pearson upper limit: P(theta <= limit) >= 1-alpha."""
    if n == 0:
        return 1.0
    return float(_beta.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.0


def _one_sided_p(b, c):
    """P(X >= b) for X ~ Bin(b+c, 1/2): exact one-sided McNemar."""
    if b + c == 0:
        return 1.0
    return float(_binomtest(b, b + c, 0.5, alternative="greater").pvalue)


class BenchmarkResult:
    """Per-shot certificate verdicts and decoder outcomes, with summaries.

    Attributes
    ----------
    verdict : (N,) int8 -- 0 success, 1 fail, 2 tie, 3 undecided (ML, certified)
    ok, ran, lowconf, invalid : dict name -> (N,) bool arrays
    decoders : list of (name, role)
    """

    def __init__(self, verdict, ok, ran, lowconf, invalid, nodes, times,
                 bounds, escalated, clustered, decoders, problem_info, config, elapsed,
                 meta, route=None, fault_counts=None):
        self.verdict = np.asarray(verdict, dtype=np.int8)
        self.ok, self.ran, self.lowconf, self.invalid = ok, ran, lowconf, invalid
        self.nodes = np.asarray(nodes, dtype=np.int64)
        self.times = np.asarray(times, dtype=np.float64)
        self.bounds = bounds
        self.escalated = np.asarray(escalated, dtype=bool)
        self.clustered = np.asarray(clustered, dtype=bool)
        n = self.verdict.size
        # route: index into ROUTES (which certifier stage decided the shot)
        self.route = (np.full(n, ROUTES.index("full"), dtype=np.int8) if route is None
                      else np.asarray(route, dtype=np.int8))
        # number of fired mechanisms per shot (-1: unknown, e.g. certify_shots)
        self.fault_counts = (np.full(n, -1, dtype=np.int64) if fault_counts is None
                             else np.asarray(fault_counts, dtype=np.int64))
        self.decoders = list(decoders)
        self.problem_info = problem_info
        self.config = config
        self.elapsed = float(elapsed)
        self.meta = dict(meta)

    @classmethod
    def from_rows(cls, rows, decoders, problem, config, elapsed, meta,
                  fault_counts=None):
        names = [n for n, _ in decoders]
        col = lambda key, dt: np.array([r[key] for r in rows], dtype=dt)
        return cls(
            verdict=col("verdict", np.int8),
            ok={n: col(f"ok:{n}", bool) for n in names},
            ran={n: col(f"ran:{n}", bool) for n in names},
            lowconf={n: col(f"lowconf:{n}", bool) for n in names},
            invalid={n: col(f"invalid:{n}", bool) for n in names},
            nodes=col("nodes", np.int64), times=col("time", np.float64),
            bounds={b: col(b, np.float64)
                    for b in ("lb_true", "ub_true", "lb_wrong", "ub_wrong")},
            escalated=col("escalated", bool), clustered=col("clustered", bool),
            decoders=decoders,
            problem_info={"name": problem.name, "detectors": problem.num_detectors,
                          "mechanisms": problem.num_mechanisms,
                          "observables": problem.num_observables,
                          "fingerprint": problem.fingerprint},
            config=config.to_dict(), elapsed=elapsed, meta=meta,
            route=np.array([r.get("route", ROUTES.index("full")) for r in rows],
                           dtype=np.int8),
            fault_counts=fault_counts)

    # ---------------------------------------------------------------- basics
    @property
    def shots(self) -> int:
        return int(self.verdict.size)

    def counts(self) -> dict:
        v = self.verdict
        return {"success": int((v == V_SUCCESS).sum()), "fail": int((v == V_FAIL).sum()),
                "tie": int((v == V_TIE).sum()), "undecided": int((v == V_BUDGET).sum())}

    @property
    def fault_band(self):
        """``(lo, hi)`` for a fault-count-band run, else None."""
        b = self.meta.get("fault_band")
        return None if b is None else (int(b[0]), int(b[1]))

    def route_counts(self) -> dict:
        """How many shots each certifier stage decided (see ``ROUTES``)."""
        return {r: int((self.route == i).sum()) for i, r in enumerate(ROUTES)}

    def ml_bracket(self, alpha=0.05) -> dict:
        """Certified bracket on the ML logical error rate.

        ``lb = #fail / N`` and ``ub = (#fail + #tie + #undecided) / N`` hold
        exactly on the sampled shots.  ``lb_ci``/``ub_ci`` bound the
        *population* ML LER, jointly with confidence at least ``1 - alpha``:
        one-sided Clopper-Pearson limits at ``alpha/2`` each (a lower limit on
        the certified-failure rate, an upper limit on the not-proven-success
        rate).

        For a fault-count-band run (``benchmark(..., fault_band=(lo, hi))``)
        ``lb``/``ub`` are conditional on the band, and the population bounds
        are ``P(S) * CP_lower`` and ``P(S) * CP_upper + (1 - P(S))`` -- the
        upper one is only informative for a band holding almost all the mass.
        Pool band and plain runs with :func:`population_bracket`.
        """
        c, N = self.counts(), self.shots
        f, u = c["fail"], c["fail"] + c["tie"] + c["undecided"]
        mass = float(self.meta.get("band_mass", 1.0))
        lo = clopper_pearson_lower(f, N, alpha / 2)
        hi = clopper_pearson_upper(u, N, alpha / 2)
        return {"lb": f / N if N else float("nan"),
                "ub": u / N if N else float("nan"),
                "lb_ci": mass * lo,
                "ub_ci": min(1.0, mass * hi + (1.0 - mass)),
                "confidence": 1.0 - alpha,
                "fault_band": self.meta.get("fault_band"), "band_mass": mass,
                "uncertified_frac": (c["tie"] + c["undecided"]) / N if N else float("nan")}

    def ler(self, name, alpha=0.05) -> dict:
        """Logical error rate of a decoder over the shots it ran on."""
        ran = self.ran[name]
        n = int(ran.sum())
        k = int((~self.ok[name] & ran).sum())
        lo, hi = clopper_pearson(k, n, alpha)
        return {"ler": k / n if n else float("nan"), "failures": k, "shots": n,
                "ci": (lo, hi), "low_confidence": int(self.lowconf[name].sum()),
                "invalid": int(self.invalid[name].sum())}

    def decomposition(self, name) -> dict:
        """Split a decoder's failures by the certified ML verdict on the same shot.

        ``intrinsic``   -- ML also provably fails.  No decoder could fix these.
        ``recoverable`` -- ML provably succeeds.  The decoder is provably
                           suboptimal on these shots.
        ``unknown``     -- the ML verdict is a tie or undecided.
        ``lucky``       -- the decoder succeeds where ML provably fails.
        """
        ran = self.ran[name]
        fail = ran & ~self.ok[name]
        v = self.verdict
        return {"intrinsic": int((fail & (v == V_FAIL)).sum()),
                "recoverable": int((fail & (v == V_SUCCESS)).sum()),
                "unknown": int((fail & (v >= V_TIE)).sum()),
                "lucky": int((ran & self.ok[name] & (v == V_FAIL)).sum())}

    def gap_to_ml(self, name) -> dict:
        """Certified test that decoder ``name`` is worse than ML on this problem.

        Paired, one-sided exact McNemar test.  Uncertified shots are resolved
        against the claim: ``b`` counts shots where the decoder fails and ML
        provably succeeds; ``c`` counts shots where the decoder succeeds and ML
        is not proven to succeed.  A small ``p_value`` is therefore a
        certified statement that the decoder is suboptimal, whatever the
        uncertified shots hide.
        """
        ran = self.ran[name]
        v = self.verdict[ran]
        ok = self.ok[name][ran]
        b = int((~ok & (v == V_SUCCESS)).sum())
        c = int((ok & (v != V_SUCCESS)).sum())
        return {"b_dec_fail_ml_ok": b, "c_dec_ok_ml_not_ok": c,
                "p_value": _one_sided_p(b, c)}

    def compare(self, a, b) -> dict:
        """Paired exact McNemar test between decoders ``a`` and ``b`` on shared shots.

        Returns the discordant counts and the two-sided p-value, plus one-sided
        p-values for "a worse than b" and "b worse than a".
        """
        both = self.ran[a] & self.ran[b]
        oa, ob = self.ok[a][both], self.ok[b][both]
        n_a_fail_b_ok = int((~oa & ob).sum())
        n_b_fail_a_ok = int((oa & ~ob).sum())
        tot = n_a_fail_b_ok + n_b_fail_a_ok
        p2 = float(_binomtest(n_a_fail_b_ok, tot, 0.5).pvalue) if tot else 1.0
        return {"a_fail_b_ok": n_a_fail_b_ok, "b_fail_a_ok": n_b_fail_a_ok,
                "p_two_sided": p2,
                "p_a_worse": _one_sided_p(n_a_fail_b_ok, n_b_fail_a_ok),
                "p_b_worse": _one_sided_p(n_b_fail_a_ok, n_a_fail_b_ok)}

    def crosscheck_contradictions(self) -> dict:
        """For every CROSSCHECK decoder (exact MLE): shots where it disagrees
        with a *proven* verdict.  Expected to be 0."""
        out = {}
        for name, role in self.decoders:
            if role != "crosscheck":
                continue
            ran = self.ran[name]
            v, ok = self.verdict[ran], self.ok[name][ran]
            out[name] = int((((v == V_SUCCESS) & ~ok) | ((v == V_FAIL) & ok)).sum())
        return out

    # ------------------------------------------------------------ reporting
    def summary(self) -> dict:
        d = {"problem": self.problem_info, "shots": self.shots,
             "elapsed_s": self.elapsed, "counts": self.counts(),
             "ml_bracket": self.ml_bracket(),
             "nodes_median": float(np.median(self.nodes)) if self.shots else 0.0,
             "escalated": int(self.escalated.sum()),
             "clustered_frac": float(self.clustered.mean()) if self.shots else 0.0,
             "routes": self.route_counts(),
             "seeders": list(self.meta.get("seeders", [])),
             "decoders": {}, "crosscheck_contradictions": self.crosscheck_contradictions()}
        for name, role in self.decoders:
            e = {"role": role, **self.ler(name)}
            if self.ran[name].any():
                e["decomposition"] = self.decomposition(name)
                e["gap_to_ml"] = self.gap_to_ml(name)
            d["decoders"][name] = e
        return d

    def __str__(self):
        s = self.summary()
        br, c = s["ml_bracket"], s["counts"]
        lines = [
            f"Certified ML benchmark -- {self.problem_info['name']} "
            f"(m={self.problem_info['detectors']}, n={self.problem_info['mechanisms']}, "
            f"k={self.problem_info['observables']})",
            f"  shots: {self.shots}   wall: {self.elapsed:.1f}s   "
            f"verdicts: {c['success']} success / {c['fail']} fail / "
            f"{c['tie']} tie / {c['undecided']} undecided",
            f"  ML LER certified bracket: [{br['lb']:.3e}, {br['ub']:.3e}]"
            + ("   (conditional on the fault band)" if self.fault_band else ""),
            f"  population ML LER, {100 * br['confidence']:g}% joint (Clopper-Pearson): "
            f"[{br['lb_ci']:.3e}, {br['ub_ci']:.3e}]"
            + (f"   band K in [{self.fault_band[0]}, {self.fault_band[1]}], "
               f"P(S) = {br['band_mass']:.4g}" if self.fault_band else ""),
            "",
            f"  {'decoder':<14}{'role':<11}{'LER':>10}  {'95% CI':<23}"
            f"{'recov.':>7}{'intrin.':>8}{'unkn.':>6}  {'p(worse than ML)':>16}",
        ]
        for name, role in self.decoders:
            e = s["decoders"][name]
            if e["shots"] == 0:
                lines.append(f"  {name:<14}{role:<11}{'not run':>10}")
                continue
            dc, g = e["decomposition"], e["gap_to_ml"]
            lines.append(
                f"  {name:<14}{role:<11}{e['ler']:>10.3e}  "
                f"[{e['ci'][0]:.2e}, {e['ci'][1]:.2e}]  "
                f"{dc['recoverable']:>7}{dc['intrinsic']:>8}{dc['unknown']:>6}  "
                f"{g['p_value']:>16.3g}")
        cc = s["crosscheck_contradictions"]
        if cc:
            lines.append(f"\n  exact-MLE contradictions with proven verdicts: {cc} "
                         f"(expected 0)")
        if s["seeders"]:
            lines.append(f"\n  certificate seeders (not scored, run only where the "
                         f"fast pass left a shot open): {', '.join(s['seeders'])}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "config": self.config, "meta": self.meta,
                "created_at": datetime.now(timezone.utc).isoformat()}

    def save(self, path):
        """Save everything: ``<path>.json`` (summary) and ``<path>.npz`` (per-shot)."""
        path = str(path)
        base, ext = os.path.splitext(path)
        if ext not in (".json", ".npz"):
            base = path
        with open(base + ".json", "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=float)
        arrays = {"verdict": self.verdict, "nodes": self.nodes, "time": self.times,
                  "escalated": self.escalated, "clustered": self.clustered,
                  "route": self.route, "fault_counts": self.fault_counts}
        arrays.update({k: v for k, v in self.bounds.items()})
        for name, _ in self.decoders:
            for tag, dct in (("ok", self.ok), ("ran", self.ran),
                             ("lowconf", self.lowconf), ("invalid", self.invalid)):
                arrays[f"{tag}__{name}"] = dct[name]
        np.savez_compressed(base + ".npz", **arrays)
        return base + ".json", base + ".npz"


# ------------------------------------------------------------------ pooling
def population_bracket(results, alpha=0.05, alpha_lower=None, alpha_upper=None):
    """Joint Clopper-Pearson bracket on the population ML LER from several runs.

    ``results`` are :class:`BenchmarkResult` s of the SAME problem from
    :func:`benchmark`, each a fixed-size run with its own seed: plain runs and
    fault-count-band runs (``fault_band=(lo, hi)``, bands pairwise disjoint).

    lower
        For every band ``S``: its band runs pooled with the plain shots whose
        fault count lies in ``S``.  Conditional on ``K in S`` every such shot
        fails ML with the same probability, so the certified-failure count
        ``F_S`` of ``n_S`` shots is exactly binomial and
        ``sum_S P(S) * CP_lower(F_S, n_S; alpha_lower / #bands)`` is a lower
        bound on the ML LER (Bonferroni over bands).  With no band run the
        plain shots alone give ``CP_lower(F, N; alpha_lower)``.
    upper
        Every plain shot: ``CP_upper(N - n_success, N; alpha_upper)``, a tie or
        undecided shot counting as a failure.

    Missed failure proofs only make the lower bound conservative, and missed
    success proofs the upper one.  Coverage is at least
    ``1 - alpha_lower - alpha_upper`` (default ``alpha/2`` each).
    """
    results = list(results)
    if not results:
        raise ValueError("no results to pool")
    a_lo = alpha / 2 if alpha_lower is None else float(alpha_lower)
    a_up = alpha / 2 if alpha_upper is None else float(alpha_upper)
    fps = {r.problem_info.get("fingerprint") for r in results}
    if len(fps) != 1 or None in fps:
        raise ValueError("results come from different (or unfingerprinted) problems")
    runs = set()
    for r in results:
        if r.meta.get("mode") != "benchmark":
            raise ValueError("population_bracket needs runs sampled by benchmark(): "
                             "the sampling law of certify_shots input is unknown")
        key = (repr(r.meta.get("seed")), repr(r.meta.get("fault_band")))
        if r.meta.get("seed") is None or key in runs:
            raise ValueError(f"runs must use distinct explicit seeds, got {key[0]} "
                             f"twice (identical shots are not independent)")
        runs.add(key)
        if r.meta.get("stopped_early") or r.meta.get("target_failures") is not None:
            warnings.warn("a run used target_failures: Clopper-Pearson assumes a "
                          "fixed number of shots, so its coverage is not exact",
                          RuntimeWarning, stacklevel=2)
    plain = [r for r in results if r.fault_band is None]
    banded = [r for r in results if r.fault_band is not None]

    # ---- upper: plain shots -------------------------------------------------
    n_up = sum(r.shots for r in plain)
    s_up = sum(r.counts()["success"] for r in plain)
    f_up = sum(r.counts()["fail"] for r in plain)
    upper = clopper_pearson_upper(n_up - s_up, n_up, a_up) if n_up else 1.0

    # ---- lower: bands (pooled with in-band plain shots) or plain alone -----
    bands = defaultdict(lambda: {"shots": 0, "fail_shots": 0, "parts": []})
    for r in banded:
        b = bands[r.fault_band]
        b["mass"] = float(r.meta["band_mass"])
        f = r.counts()["fail"]
        b["shots"] += r.shots
        b["fail_shots"] += f
        b["parts"].append({"seed": r.meta["seed"], "shots": r.shots, "fail_shots": f})
    keys = sorted(bands)
    for x, y in zip(keys, keys[1:]):
        if y[0] <= x[1]:
            raise ValueError(f"fault bands {list(x)} and {list(y)} overlap")
    for (lo, hi) in keys:
        for r in plain:
            if (r.fault_counts < 0).any():
                raise ValueError("a plain run has no fault counts")
            inb = (r.fault_counts >= lo) & (r.fault_counts <= hi)
            f = int((inb & (r.verdict == V_FAIL)).sum())
            bands[(lo, hi)]["shots"] += int(inb.sum())
            bands[(lo, hi)]["fail_shots"] += f
            bands[(lo, hi)]["parts"].append({"seed": r.meta["seed"], "plain": True,
                                             "shots": int(inb.sum()), "fail_shots": f})
    rows = []
    if keys:
        a_each = a_lo / len(keys)
        for key in keys:
            b = bands[key]
            cond = clopper_pearson_lower(b["fail_shots"], b["shots"], a_each)
            rows.append({"band": list(key), "mass": b["mass"], "shots": b["shots"],
                         "fail_shots": b["fail_shots"],
                         "estimate": b["mass"] * b["fail_shots"] / b["shots"],
                         "lower": b["mass"] * cond, "parts": b["parts"]})
        lower = sum(x["lower"] for x in rows)
    else:
        lower = clopper_pearson_lower(f_up, n_up, a_lo)
    return {"lower": lower, "upper": upper,
            "joint_confidence": 1.0 - a_lo - a_up,
            "alpha_lower": a_lo, "alpha_upper": a_up,
            "band_rows": rows,
            "plain_counts": {"shots": n_up, "success_shots": s_up, "fail_shots": f_up}}
