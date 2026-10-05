"""Fault-count band sampling, Clopper-Pearson bounds, and the certifier stages."""

import itertools
import warnings

import numpy as np
import pytest
from scipy.stats import binom

import qldpc_certificate_benchmark as qcb
from conftest import brute_class_minima


def _brute_band(priors, lo, hi):
    """Exact P(K in S) and P(e | K in S) by enumerating all 2^n patterns."""
    n = priors.size
    X = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.int64)
    pr = np.prod(np.where(X == 1, priors, 1 - priors), axis=1)
    K = X.sum(axis=1)
    inb = (K >= lo) & (K <= hi)
    return float(pr[inb].sum()), X[inb], pr[inb] / pr[inb].sum()


def test_band_mass_and_pmf_exact():
    rng = np.random.default_rng(0)
    pri = rng.uniform(0.01, 0.4, size=10)
    prob = qcb.DecodingProblem(np.eye(10, dtype=np.uint8), np.ones((1, 10)), pri)
    for lo, hi in [(0, 10), (3, 5), (7, 7), (0, 0)]:
        mass, _, _ = _brute_band(pri, lo, hi)
        assert abs(prob.band_mass(lo, hi) - mass) < 1e-12
    # uniform priors: binomial, and a deep tail where naive products underflow
    n, p = 20000, 1e-3
    prob = qcb.DecodingProblem(np.ones((1, n)), np.ones((1, n)), p)
    pmf = prob.fault_count_pmf(400)
    ref = binom.pmf(np.arange(401), n, p)
    ok = ref > 1e-300
    assert ref[300] < 1e-200 and np.allclose(pmf[ok], ref[ok], rtol=1e-9, atol=0)


def test_band_sampler_matches_conditional_law():
    rng = np.random.default_rng(1)
    n = 8
    pri = rng.uniform(0.05, 0.45, size=n)
    H = np.eye(n, dtype=np.uint8)
    prob = qcb.DecodingProblem(H, np.ones((1, n)), pri)
    lo, hi = 2, 4
    _, X, cond = _brute_band(pri, lo, hi)
    shots = 200_000
    _, _, E, K = prob.sample(shots, seed=5, fault_band=(lo, hi), return_errors=True,
                             return_fault_counts=True)
    assert np.array_equal(E.sum(axis=1), K) and K.min() >= lo and K.max() <= hi
    code = E.astype(np.int64) @ (1 << np.arange(n))
    freq = np.bincount(code, minlength=1 << n) / shots
    expect = np.zeros(1 << n)
    expect[X @ (1 << np.arange(n))] = cond
    # chi-square-style check on every pattern with enough mass
    sd = np.sqrt(expect * (1 - expect) / shots)
    big = expect > 1e-4
    assert np.all(np.abs(freq[big] - expect[big]) < 5 * sd[big])
    assert freq[~(expect > 0)].sum() == 0
    # determinism
    a = prob.sample(100, seed=9, fault_band=(lo, hi))
    b = prob.sample(100, seed=9, fault_band=(lo, hi))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_plain_sampling_unchanged():
    # the v0.1 sampler: fault counts are an extra output, the draws are the same
    Hx, Hz, Lx, Lz = qcb.codes.toric_code(3)
    prob = qcb.DecodingProblem.from_code(Hz, Lz, 0.08)
    s1, o1, e1 = prob.sample(50, seed=3, return_errors=True)
    rng = np.random.default_rng(3)
    e = (rng.random((50, prob.num_mechanisms)) < prob.priors).astype(np.uint8)
    assert np.array_equal(e, e1) and np.array_equal(s1, prob.syndrome(e))
    _, _, K = prob.sample(50, seed=3, return_fault_counts=True)
    assert np.array_equal(K, e.sum(axis=1))


def test_clopper_pearson_one_sided():
    from scipy.stats import binomtest
    for k, n in [(0, 50), (3, 50), (50, 50), (7, 1000)]:
        lo = qcb.clopper_pearson_lower(k, n, 0.025)
        hi = qcb.clopper_pearson_upper(k, n, 0.025)
        two = binomtest(k, n).proportion_ci(0.95, method="exact")
        assert abs(lo - two.low) < 1e-10 and abs(hi - two.high) < 1e-10
    assert qcb.clopper_pearson_upper(0, 100, 0.05) == pytest.approx(1 - 0.05 ** (1 / 100))


def test_ml_bracket_joint_and_population_bracket(small_problem):
    kw = dict(seeders=None, num_workers=1, progress=False)
    plain = qcb.benchmark(small_problem, shots=150, seed=1, **kw)
    br = plain.ml_bracket(0.05)
    c = plain.counts()
    u = c["fail"] + c["tie"] + c["undecided"]
    assert br["lb_ci"] == pytest.approx(qcb.clopper_pearson_lower(c["fail"], 150, 0.025))
    assert br["ub_ci"] == pytest.approx(qcb.clopper_pearson_upper(u, 150, 0.025))
    assert br["lb_ci"] <= br["lb"] <= br["ub"] <= br["ub_ci"]

    band = qcb.benchmark(small_problem, shots=100, seed=2, fault_band=(4, 7), **kw)
    assert band.fault_counts.min() >= 4 and band.fault_counts.max() <= 7
    mass = small_problem.band_mass(4, 7)
    pb = qcb.population_bracket([plain, band])
    row = pb["band_rows"][0]
    in_band = (plain.fault_counts >= 4) & (plain.fault_counts <= 7)
    assert row["shots"] == 100 + int(in_band.sum())
    f = band.counts()["fail"] + int((in_band & (plain.verdict == 1)).sum())
    assert row["fail_shots"] == f
    assert pb["lower"] == pytest.approx(
        mass * qcb.clopper_pearson_lower(f, row["shots"], 0.025))
    assert pb["upper"] == pytest.approx(br["ub_ci"])
    assert pb["joint_confidence"] == pytest.approx(0.95)
    # misuse is rejected
    with pytest.raises(ValueError):
        qcb.population_bracket([plain, plain])                    # same seed
    other = qcb.benchmark(small_problem, shots=20, seed=3, fault_band=(6, 9), **kw)
    with pytest.raises(ValueError):
        qcb.population_bracket([plain, band, other])              # overlapping bands
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        qcb.population_bracket([plain])


def test_stages_agree_with_brute_force(small_problem):
    """Fast pass, full certificate and global ILP route: same verdicts, all
    consistent with exhaustive enumeration."""
    H, L, pri = small_problem.H, small_problem.L, small_problem.priors
    syn, obs = small_problem.sample(60, seed=11)
    cfgs = {
        "fast": qcb.CertConfig(),
        "full": qcb.CertConfig(use_fast_pass=False),
        # starve every search so the global route has to decide
        "global": qcb.CertConfig(use_fast_pass=False, use_clusters=False,
                                 use_lp_screen=False, use_milp_fallback=False,
                                 use_exact_ilp=False, escalate=False,
                                 true_node_budget=1, wrong_node_budget=1,
                                 use_global_route=True, global_time_limit=30.0),
    }
    certs = {k: qcb.Certifier(small_problem, c) for k, c in cfgs.items()}
    routes = set()
    for s, lam in zip(syn, obs):
        mins = brute_class_minima(H, L, pri, s)
        true_cls = int(lam @ (1 << np.arange(L.shape[0])))
        wrong = np.delete(mins, true_cls).min()
        for name, cert in certs.items():
            c = cert.certify(s, lam)
            routes.add(c.route)
            if c.verdict == "success":
                assert mins[true_cls] < wrong, name
            elif c.verdict == "fail":
                assert wrong < mins[true_cls], name
    assert {"fast", "full", "global"} <= routes
