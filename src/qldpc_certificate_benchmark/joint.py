"""Joint versus separate X/Z decoding, certified on identical shots.

For a CSS code the usual decoders treat the X-error and Z-error sectors as two
independent problems, each with the *marginal* fault model of its sector.  A
correlated fault (a ``Y``, or a two-qubit fault after a CNOT) is then priced as
two independent faults, although it is one event of the joint model.  The
joint ML decoder uses the joint model directly.

:func:`compare_joint_separate` samples shots from a joint model, then certifies

* the JOINT ML decision (one problem, ``2k`` observables), and
* the SEPARATE ML decision: each sector's marginal model
  (:meth:`DecodingProblem.css_split`) certified on that sector's rows of the
  same shot.  A shot is a separate-ML failure iff either sector fails.

Both verdicts are exact, so the two certified brackets and the paired test
compare the two ML decoders themselves, not approximations of them.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ._engine.constants import V_SUCCESS, V_FAIL, V_TIE, V_BUDGET
from .benchmark import certify_shots
from .config import CertConfig, DEFAULT_CFG
from .results import BenchmarkResult, _one_sided_p, clopper_pearson, \
    clopper_pearson_lower, clopper_pearson_upper

__all__ = ["compare_joint_separate", "JointComparison", "combine_sector_verdicts"]


def combine_sector_verdicts(vx, vz):
    """Separate-decoding verdict of a shot from its two sector verdicts.

    ``fail`` if either sector provably fails; ``success`` only if both provably
    succeed; ``undecided`` if a sector is undecided; otherwise ``tie``."""
    vx = np.asarray(vx, dtype=np.int8)
    vz = np.asarray(vz, dtype=np.int8)
    out = np.full(vx.shape, V_TIE, dtype=np.int8)
    out[(vx == V_BUDGET) | (vz == V_BUDGET)] = V_BUDGET
    out[(vx == V_SUCCESS) & (vz == V_SUCCESS)] = V_SUCCESS
    out[(vx == V_FAIL) | (vz == V_FAIL)] = V_FAIL
    return out


def _bracket(verdict, alpha):
    v = np.asarray(verdict)
    N = v.size
    f = int((v == V_FAIL).sum())
    u = int((v != V_SUCCESS).sum())
    return {"lb": f / N if N else float("nan"), "ub": u / N if N else float("nan"),
            "lb_ci": clopper_pearson_lower(f, N, alpha / 2),
            "ub_ci": clopper_pearson_upper(u, N, alpha / 2),
            "confidence": 1.0 - alpha, "fail": f, "not_success": u, "shots": N}


def _paired(va, vb):
    """Certified one-sided test that decision ``a`` fails less often than ``b``.

    Uncertified shots count against the claim: ``wins`` are shots where ``a``
    provably succeeds and ``b`` provably fails; ``losses`` are shots where
    ``a`` is not proven to succeed and ``b`` is not proven to fail."""
    wins = int(((va == V_SUCCESS) & (vb == V_FAIL)).sum())
    losses = int(((va != V_SUCCESS) & (vb != V_FAIL)).sum())
    return {"wins": wins, "losses": losses, "p_value": _one_sided_p(wins, losses)}


class JointComparison:
    """Certified joint vs separate ML (and decoders) on the same shots.

    Attributes
    ----------
    joint : BenchmarkResult -- the joint model on every shot.
    x, z : BenchmarkResult -- the two marginal sector models on the same shots
        (row ``i`` of all three is shot ``i``).
    separate_verdict : (N,) int8 -- :func:`combine_sector_verdicts` of x and z.
    """

    def __init__(self, joint: BenchmarkResult, x: BenchmarkResult, z: BenchmarkResult,
                 partition, meta=None):
        if not joint.shots == x.shots == z.shots:
            raise ValueError("joint, x and z results must cover the same shots")
        self.joint, self.x, self.z = joint, x, z
        self.partition = partition
        self.meta = dict(meta or {})
        self.separate_verdict = combine_sector_verdicts(x.verdict, z.verdict)

    @property
    def shots(self) -> int:
        return self.joint.shots

    def ml_brackets(self, alpha=0.05) -> dict:
        """Certified ML LER brackets of joint and separate decoding.

        ``lb``/``ub`` hold exactly on the shots; ``lb_ci``/``ub_ci`` bound the
        population rates jointly at ``1 - alpha`` (as
        :meth:`BenchmarkResult.ml_bracket`)."""
        return {"joint": _bracket(self.joint.verdict, alpha),
                "separate": _bracket(self.separate_verdict, alpha)}

    def paired_test(self) -> dict:
        """One-sided certified paired tests in both directions:
        ``joint_better`` (joint ML fails less often than separate ML) and
        ``separate_better``.  A small p-value is a certified statement whatever
        the uncertified shots hide."""
        return {"joint_better": _paired(self.joint.verdict, self.separate_verdict),
                "separate_better": _paired(self.separate_verdict, self.joint.verdict)}

    # ------------------------------------------------------------- decoders
    def joint_decoders(self):
        return [n for n, _ in self.joint.decoders]

    def separate_decoders(self):
        zs = {n for n, _ in self.z.decoders}
        return [n for n, _ in self.x.decoders if n in zs]

    def decoder_ler(self, name, mode, alpha=0.05) -> dict:
        """LER of decoder ``name`` used jointly (``mode="joint"``) or separately
        (``mode="separate"``: correct iff both sectors are)."""
        if mode == "joint":
            return self.joint.ler(name, alpha)
        if mode != "separate":
            raise ValueError("mode must be 'joint' or 'separate'")
        ran = self.x.ran[name] & self.z.ran[name]
        ok = self.x.ok[name] & self.z.ok[name]
        n = int(ran.sum())
        k = int((~ok & ran).sum())
        lo, hi = clopper_pearson(k, n, alpha)
        return {"ler": k / n if n else float("nan"), "failures": k, "shots": n,
                "ci": (lo, hi)}

    def _decoder_ok(self, name, mode):
        if mode == "joint":
            return self.joint.ok[name], self.joint.ran[name]
        return (self.x.ok[name] & self.z.ok[name]), (self.x.ran[name] & self.z.ran[name])

    def decoder_vs_ml(self, name, mode, ml="separate") -> dict:
        """Certified one-sided test that decoder ``name`` (used in ``mode``)
        fails LESS often than ``ml`` ("joint" or "separate") ML.  Uncertified
        shots count against the claim.  E.g. a joint decoder beating the
        certified separate-ML bracket."""
        ok, ran = self._decoder_ok(name, mode)
        v = (self.joint.verdict if ml == "joint" else self.separate_verdict)[ran]
        ok = ok[ran]
        wins = int((ok & (v == V_FAIL)).sum())
        losses = int((~ok & (v != V_FAIL)).sum())
        return {"wins": wins, "losses": losses, "p_value": _one_sided_p(wins, losses)}

    # ------------------------------------------------------------ reporting
    def summary(self, alpha=0.05) -> dict:
        d = {"shots": self.shots, "ml_brackets": self.ml_brackets(alpha),
             "paired": self.paired_test(),
             "counts": {"joint": self.joint.counts(),
                        "x": self.x.counts(), "z": self.z.counts()},
             "routes": {"joint": self.joint.route_counts(),
                        "x": self.x.route_counts(), "z": self.z.route_counts()},
             "elapsed_s": {"joint": self.joint.elapsed, "x": self.x.elapsed,
                           "z": self.z.elapsed},
             "decoders": {}}
        for mode, names in (("joint", self.joint_decoders()),
                            ("separate", self.separate_decoders())):
            for name in names:
                e = self.decoder_ler(name, mode, alpha)
                if e["shots"]:
                    e["beats_separate_ml"] = self.decoder_vs_ml(name, mode, "separate")
                    e["beats_joint_ml"] = self.decoder_vs_ml(name, mode, "joint")
                d["decoders"][f"{mode}:{name}"] = e
        d["meta"] = self.meta
        return d

    def __str__(self):
        s = self.summary()
        br, pt = s["ml_brackets"], s["paired"]
        info = self.joint.problem_info
        lines = [f"Joint vs separate ML -- {info['name']} (joint m={info['detectors']}, "
                 f"n={info['mechanisms']}, 2k={info['observables']}; "
                 f"{self.shots} shots)"]
        for key in ("separate", "joint"):
            b = br[key]
            lines.append(f"  {key + ' ML':<12} certified [{b['lb']:.3e}, {b['ub']:.3e}]   "
                         f"population {100 * b['confidence']:g}%: "
                         f"[{b['lb_ci']:.3e}, {b['ub_ci']:.3e}]")
        jb, sb = pt["joint_better"], pt["separate_better"]
        lines.append(f"  paired: joint better {jb['wins']} v {jb['losses']} "
                     f"(p = {jb['p_value']:.3g});  separate better "
                     f"{sb['wins']} v {sb['losses']} (p = {sb['p_value']:.3g})")
        if s["decoders"]:
            lines += ["", f"  {'decoder':<22}{'LER':>10}  {'95% CI':<23}"
                          f"{'beats sep. ML (p)':>18}{'beats joint ML (p)':>19}"]
            for key, e in s["decoders"].items():
                if not e["shots"]:
                    lines.append(f"  {key:<22}{'not run':>10}")
                    continue
                lines.append(
                    f"  {key:<22}{e['ler']:>10.3e}  [{e['ci'][0]:.2e}, {e['ci'][1]:.2e}]"
                    f"  {e['beats_separate_ml']['p_value']:>16.3g}"
                    f"  {e['beats_joint_ml']['p_value']:>17.3g}")
        return "\n".join(lines)

    def save(self, path):
        """``<path>.json`` (summary) plus the three per-shot results
        ``<path>_joint``, ``<path>_x``, ``<path>_z`` (see BenchmarkResult.save)."""
        base = os.path.splitext(str(path))[0]
        files = [base + ".json"]
        with open(base + ".json", "w") as fh:
            json.dump(self.summary(), fh, indent=2, default=float)
        for tag, r in (("joint", self.joint), ("x", self.x), ("z", self.z)):
            files += list(r.save(f"{base}_{tag}"))
        return files


def compare_joint_separate(problem, shots: int = 1000, *, partition=None, seed=0,
                           joint_decoders=None, separate_decoders=None,
                           seeders="auto", config: CertConfig = DEFAULT_CFG,
                           num_workers=None, progress: bool = True,
                           mp_context=None) -> JointComparison:
    """Certify joint and separate ML decoding on the same shots.

    Parameters
    ----------
    problem : DecodingProblem
        A JOINT X/Z model, e.g. from :meth:`DecodingProblem.from_css_code_pauli`,
        :meth:`~DecodingProblem.from_css_sectors`, or a Stim circuit DEM.
    shots : int
        Shots sampled from ``problem`` (the joint, correlated noise).
    partition : CSSPartition, optional
        Defaults to ``problem.css``; required for a model built otherwise.
    joint_decoders : decoders run on the joint model (same forms as
        :func:`benchmark`).
    separate_decoders : decoders run once per sector, on that sector's rows
        and marginal model; a shot is correct iff both sectors are.  Each one
        is set up separately for the X and the Z problem, so pass built-in
        names, :class:`Decoder` objects, or ``FunctionDecoder(factory=...)``
        (a bare function cannot tell the two sectors apart).
    seeders : "auto", list, or None
        Bracket seeders, applied to each of the three problems (see
        :func:`benchmark`).  They are measured too, in both modes.
    seed, config, num_workers, progress, mp_context : as in :func:`benchmark`.

    Returns
    -------
    JointComparison
    """
    part = problem.css if partition is None else partition
    x_prob, z_prob = problem.css_split(part)
    xd, zd, xo, zo = part.arrays()
    syn, obs, K = problem.sample(shots, seed=seed, return_fault_counts=True)
    kw = dict(seeders=seeders, config=config, num_workers=num_workers,
              progress=progress, mp_context=mp_context)
    joint = certify_shots(problem, syn, obs, decoders=joint_decoders, **kw)
    joint.fault_counts = np.asarray(K, dtype=np.int64)
    rx = certify_shots(x_prob, np.ascontiguousarray(syn[:, xd]),
                       np.ascontiguousarray(obs[:, xo]), decoders=separate_decoders, **kw)
    rz = certify_shots(z_prob, np.ascontiguousarray(syn[:, zd]),
                       np.ascontiguousarray(obs[:, zo]), decoders=separate_decoders, **kw)
    meta = {"mode": "compare_joint_separate", "seed": seed, "shots": int(shots),
            "x_problem": rx.problem_info, "z_problem": rz.problem_info}
    return JointComparison(joint, rx, rz, part, meta=meta)
