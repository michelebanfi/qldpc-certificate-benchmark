"""Every proven verdict must agree with exhaustive enumeration."""
import numpy as np
import pytest

import qldpc_certificate_benchmark as qcb
from conftest import brute_class_minima

EPS = 1e-9


def _truth(problem, s, lam):
    minima = brute_class_minima(problem.H, problem.L, problem.priors, s)
    true_cls = int(lam @ (1 << np.arange(len(lam))))
    m_true = minima[true_cls]
    m_wrong = np.delete(minima, true_cls).min()
    if m_true < m_wrong - EPS:
        return "success"
    if m_wrong < m_true - EPS:
        return "fail"
    return "tie"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_certifier_matches_bruteforce(small_problem, seed):
    cert = qcb.Certifier(small_problem)
    syn, obs = small_problem.sample(60, seed=seed)
    n_decided = 0
    for s, lam in zip(syn, obs):
        c = cert.certify(s, lam)
        truth = _truth(small_problem, s.astype(np.int64), lam.astype(np.int64))
        if c.verdict in ("success", "fail", "tie"):
            assert c.verdict == truth, (c, truth)
            n_decided += 1
    assert n_decided >= 50


def test_bogus_seeds_are_harmless(small_problem):
    """Random (mostly invalid) seeds may never change a proven verdict."""
    cert = qcb.Certifier(small_problem)
    rng = np.random.default_rng(9)
    syn, obs = small_problem.sample(30, seed=5)
    for s, lam in zip(syn, obs):
        seeds = [(rng.random(small_problem.num_mechanisms) < 0.3).astype(np.uint8)
                 for _ in range(5)]
        a = cert.certify(s, lam).verdict
        b = cert.certify(s, lam, seeds=seeds).verdict
        truth = _truth(small_problem, s.astype(np.int64), lam.astype(np.int64))
        for v in (a, b):
            assert v == "undecided" or v == truth


def test_benchmark_bracket_contains_truth(small_problem):
    res = qcb.benchmark(small_problem, {"osd0": qcb.BPOSD(osd_method="osd0", name="osd0", role="probe")}, shots=120,
                        seed=4, num_workers=1, progress=False)
    syn, obs = small_problem.sample(120, seed=4)
    truth = [_truth(small_problem, s.astype(np.int64), l.astype(np.int64))
             for s, l in zip(syn, obs)]
    ml_fail = np.mean([t == "fail" for t in truth])
    ml_notok = np.mean([t != "success" for t in truth])
    br = res.ml_bracket()
    assert br["lb"] <= ml_fail + 1e-12 and ml_notok <= br["ub"] + 1e-12
