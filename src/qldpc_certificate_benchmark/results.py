"""Benchmark results: certified ML bracket, decoder LERs, paired tests."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy.stats import beta as _beta, binomtest as _binomtest

from ._engine.constants import V_SUCCESS, V_FAIL, V_TIE, V_BUDGET

__all__ = ["BenchmarkResult", "clopper_pearson"]


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided (1-alpha) binomial confidence interval for k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


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
                 meta):
        self.verdict = np.asarray(verdict, dtype=np.int8)
        self.ok, self.ran, self.lowconf, self.invalid = ok, ran, lowconf, invalid
        self.nodes = np.asarray(nodes, dtype=np.int64)
        self.times = np.asarray(times, dtype=np.float64)
        self.bounds = bounds
        self.escalated = np.asarray(escalated, dtype=bool)
        self.clustered = np.asarray(clustered, dtype=bool)
        self.decoders = list(decoders)
        self.problem_info = problem_info
        self.config = config
        self.elapsed = float(elapsed)
        self.meta = dict(meta)

    @classmethod
    def from_rows(cls, rows, decoders, problem, config, elapsed, meta):
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
                          "observables": problem.num_observables},
            config=config.to_dict(), elapsed=elapsed, meta=meta)

    # ---------------------------------------------------------------- basics
    @property
    def shots(self) -> int:
        return int(self.verdict.size)

    def counts(self) -> dict:
        v = self.verdict
        return {"success": int((v == V_SUCCESS).sum()), "fail": int((v == V_FAIL).sum()),
                "tie": int((v == V_TIE).sum()), "undecided": int((v == V_BUDGET).sum())}

    def ml_bracket(self, alpha=0.05) -> dict:
        """Certified bracket on the ML logical error rate over these shots.

        ``lb = #fail / N`` and ``ub = (#fail + #tie + #undecided) / N`` hold
        exactly on the sampled shots.  ``lb_ci``/``ub_ci`` widen them by the
        one-sided sampling error: a (1-alpha) Clopper-Pearson lower bound on
        ``lb`` and upper bound on ``ub``.
        """
        c, N = self.counts(), self.shots
        f, u = c["fail"], c["fail"] + c["tie"] + c["undecided"]
        return {"lb": f / N if N else float("nan"),
                "ub": u / N if N else float("nan"),
                "lb_ci": clopper_pearson(f, N, 2 * alpha)[0],
                "ub_ci": clopper_pearson(u, N, 2 * alpha)[1],
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
            f"  ML LER certified bracket: [{br['lb']:.3e}, {br['ub']:.3e}]   "
            f"(95% CI-widened: [{br['lb_ci']:.3e}, {br['ub_ci']:.3e}])",
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
        if any(r == "seed" for _, r in self.decoders):
            lines.append("\n  note: 'seed' decoders helped build the bracket, so their "
                         "comparison to ML is not independent.")
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
                  "escalated": self.escalated, "clustered": self.clustered}
        arrays.update({k: v for k, v in self.bounds.items()})
        for name, _ in self.decoders:
            for tag, dct in (("ok", self.ok), ("ran", self.ran),
                             ("lowconf", self.lowconf), ("invalid", self.invalid)):
                arrays[f"{tag}__{name}"] = dct[name]
        np.savez_compressed(base + ".npz", **arrays)
        return base + ".json", base + ".npz"
