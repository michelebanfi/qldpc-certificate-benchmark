"""Joint X/Z models: exact construction, exact marginals, certified verdicts."""
import itertools

import numpy as np
import pytest

import qldpc_certificate_benchmark as qcb
from qldpc_certificate_benchmark.problem import _independent_pauli
from conftest import brute_class_minima

EPS = 1e-9

# [[4,2,2]]: the smallest CSS code with logical qubits -- a joint model has
# 3 * 4 = 12 mechanisms, small enough for exhaustive enumeration
H422 = np.array([[1, 1, 1, 1]], dtype=np.uint8)


def _channel_of(qx, qy, qz):
    """Pauli channel produced by independent X, Y, Z mechanisms on one qubit."""
    P = {"I": 0.0, "X": 0.0, "Y": 0.0, "Z": 0.0}
    mul = {("I", "I"): "I"}
    for a, b, c in itertools.product((0, 1), repeat=3):
        prob = ((qx if a else 1 - qx) * (qy if b else 1 - qy) * (qz if c else 1 - qz))
        xpart, zpart = (a + b) % 2, (b + c) % 2          # X^a Y^b Z^c
        P[{(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}[(xpart, zpart)]] += prob
    return P


def test_independent_pauli_is_exact():
    """Exact whenever the channel is a product of independent flips, refused
    exactly when it is not (px(1-p) >= py*pz and cyclically)."""
    rng = np.random.default_rng(0)
    n_ok = n_refused = 0
    for _ in range(400):
        px, py, pz = rng.uniform(0.001, 0.12, 3)
        p = px + py + pz
        representable = (px * (1 - p) >= py * pz and py * (1 - p) >= px * pz
                         and pz * (1 - p) >= px * py)
        if not representable:
            with pytest.raises(ValueError):
                _independent_pauli(px, py, pz)
            n_refused += 1
            continue
        qx, qy, qz = (float(v) for v in _independent_pauli(px, py, pz))
        P = _channel_of(qx, qy, qz)
        assert abs(P["X"] - px) < 1e-12 and abs(P["Y"] - py) < 1e-12
        assert abs(P["Z"] - pz) < 1e-12
        n_ok += 1
    assert n_ok > 100 and n_refused > 10     # both branches exercised
    q = float(_independent_pauli(0.03, 0.03, 0.03)[0])
    assert abs(q - (1 - np.sqrt(1 - 4 * 0.09 / 3)) / 2) < 1e-15
    with pytest.raises(ValueError):          # Y and Z but no X: not a product
        _independent_pauli(0.0, 0.05, 0.05)
    with pytest.raises(ValueError):          # a marginal >= 1/2
        _independent_pauli(0.2, 0.3, 0.1)


@pytest.mark.parametrize("code", ["toric3", "bb72"])
def test_depolarizing_marginals_are_the_separate_models(code):
    Hx, Hz, Lx, Lz = (qcb.codes.toric_code(3) if code == "toric3"
                      else qcb.codes.bb_72_12_6())
    p = 0.06
    J = qcb.DecodingProblem.from_css_code_depolarizing(Hx, Hz, p)
    n = Hx.shape[1]
    assert J.num_mechanisms == 3 * n and J.num_observables == 2 * Lx.shape[0]
    X, Z = J.css_split()
    Xs = qcb.DecodingProblem.from_css_code(Hx, Hz, 2 * p / 3, error_type="X")
    Zs = qcb.DecodingProblem.from_css_code(Hx, Hz, 2 * p / 3, error_type="Z")
    for got, want in ((X, Xs), (Z, Zs)):
        assert np.array_equal(got.H, want.H) and np.array_equal(got.L, want.L)
        assert np.allclose(got.priors, want.priors, rtol=0, atol=1e-15)


def test_from_css_sectors_reproduces_the_pauli_builder():
    Hx, Hz, Lx, Lz = qcb.codes.toric_code(3)
    n = Hx.shape[1]
    rng = np.random.default_rng(1)
    px, py, pz = (rng.uniform(0.01, 0.05, n) for _ in range(3))
    J = qcb.DecodingProblem.from_css_code_pauli(Hx, Hz, px, py, pz)
    qx, qy, qz = _independent_pauli(px, py, pz)
    q = np.arange(n)
    mech = ([(i, -1, qx[i]) for i in q] + [(-1, i, qz[i]) for i in q]
            + [(i, i, qy[i]) for i in q])
    J2 = qcb.DecodingProblem.from_css_sectors(
        qcb.DecodingProblem(Hz, Lz, 0.1), qcb.DecodingProblem(Hx, Lx, 0.1), mech)
    assert np.array_equal(J.H, J2.H) and np.array_equal(J.L, J2.L)
    assert np.allclose(J.priors, J2.priors, rtol=0, atol=1e-15)
    assert J.css == J2.css


def _law(H, L, priors, rows_H=None, rows_L=None):
    """Exact distribution of (H e, L e) restricted to the given rows."""
    n = H.shape[1]
    Hr = H if rows_H is None else H[rows_H]
    Lr = L if rows_L is None else L[rows_L]
    E = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int64)
    pe = np.prod(np.where(E == 1, priors, 1 - priors), axis=1)
    key = np.hstack([(E @ Hr.T.astype(np.int64)) % 2, (E @ Lr.T.astype(np.int64)) % 2])
    law = {}
    for k, v in zip(map(bytes, key.astype(np.uint8)), pe):
        law[k] = law.get(k, 0.0) + v
    return law


def test_css_split_is_the_exact_marginal_law():
    """The split sector model has exactly the joint model's law of the
    sector's (syndrome, observables) -- including merged mechanisms."""
    rng = np.random.default_rng(2)
    for trial in range(6):
        mx, mz, nx, nz, k = 3, 3, 5, 5, 1
        HX = (rng.random((mx, nx)) < 0.5).astype(np.uint8)
        HZ = (rng.random((mz, nz)) < 0.5).astype(np.uint8)
        for H in (HX, HZ):
            for j in range(H.shape[1]):
                if not H[:, j].any():
                    H[rng.integers(H.shape[0]), j] = 1
        LX = (rng.random((k, nx)) < 0.5).astype(np.uint8)
        LZ = (rng.random((k, nz)) < 0.5).astype(np.uint8)
        xs, zs = qcb.DecodingProblem(HX, LX, 0.1), qcb.DecodingProblem(HZ, LZ, 0.1)
        mech = [(i, -1, rng.uniform(0.02, 0.2)) for i in range(nx)]
        mech += [(-1, i, rng.uniform(0.02, 0.2)) for i in range(nz)]
        mech += [(int(rng.integers(nx)), int(rng.integers(nz)), rng.uniform(0.02, 0.2))
                 for _ in range(4)]
        J = qcb.DecodingProblem.from_css_sectors(xs, zs, mech)
        X, Z = J.css_split()
        xd, zd, xo, zo = J.css.arrays()
        for sec, d, o in ((X, xd, xo), (Z, zd, zo)):
            joint_law = _law(J.H, J.L, J.priors, d, o)
            sec_law = _law(sec.H, sec.L, sec.priors)
            assert set(joint_law) == set(sec_law)
            for key in joint_law:
                assert abs(joint_law[key] - sec_law[key]) < 1e-12


def test_partition_validation():
    Hx, Hz, Lx, Lz = qcb.codes.toric_code(3)
    J = qcb.DecodingProblem.from_css_code_depolarizing(Hx, Hz, 0.05)
    m, k = J.num_detectors, J.num_observables
    with pytest.raises(ValueError):           # overlapping sectors
        qcb.CSSPartition(range(m // 2 + 1), range(m // 2, m), range(k // 2),
                         range(k // 2, k)).validate(m, k)
    with pytest.raises(ValueError):           # a detector in neither sector
        qcb.CSSPartition(range(m // 2), range(m // 2, m - 1), range(k // 2),
                         range(k // 2, k)).validate(m, k)
    with pytest.raises(ValueError):           # no observable in a sector
        qcb.CSSPartition(range(m // 2), range(m // 2, m), range(k), ()).validate(m, k)
    plain = qcb.DecodingProblem(J.H, J.L, J.priors)
    with pytest.raises(ValueError):           # no partition known
        plain.css_split()
    X, Z = plain.css_split(J.css)             # an explicit partition works
    assert X.num_observables == Z.num_observables == k // 2
    with pytest.raises(ValueError):           # observables in the wrong sector:
        # an X-sector observable assigned to Z is flipped by Z-sector columns
        # that no Z detector sees -> an "undetectable logical" in the split
        swap = qcb.CSSPartition(J.css.x_detectors, J.css.z_detectors,
                                J.css.z_observables, J.css.x_observables)
        plain.css_split(swap)
    with pytest.raises(TypeError):
        qcb.DecodingProblem(J.H, J.L, J.priors, css=("not", "a", "partition"))


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


def test_joint_and_separate_verdicts_match_bruteforce():
    """[[4,2,2]] under a non-uniform Pauli channel: every proven joint and
    sector verdict equals exhaustive ML, and the separate verdict is the
    combination of the two sector verdicts."""
    rng = np.random.default_rng(4)
    px, py, pz = (rng.uniform(0.03, 0.15, 4) for _ in range(3))
    J = qcb.DecodingProblem.from_css_code_pauli(H422, H422, px, py, pz)
    X, Z = J.css_split()
    xd, zd, xo, zo = J.css.arrays()
    cj, cx, cz = qcb.Certifier(J), qcb.Certifier(X), qcb.Certifier(Z)
    syn, obs = J.sample(150, seed=7)
    tally = {"success": 0, "fail": 0, "tie": 0}
    names = ("success", "fail", "tie", "undecided")
    for s, lam in zip(syn, obs):
        vj = cj.certify(s, lam).verdict
        tj = _truth(J, s.astype(np.int64), lam.astype(np.int64))
        assert vj in ("undecided", tj)
        vx = cx.certify(s[xd], lam[xo]).verdict
        vz = cz.certify(s[zd], lam[zo]).verdict
        tx = _truth(X, s[xd].astype(np.int64), lam[xo].astype(np.int64))
        tz = _truth(Z, s[zd].astype(np.int64), lam[zo].astype(np.int64))
        assert vx in ("undecided", tx) and vz in ("undecided", tz)
        comb = qcb.combine_sector_verdicts([names.index(vx)], [names.index(vz)])[0]
        if "fail" in (vx, vz):
            assert names[comb] == "fail"
        elif vx == vz == "success":
            assert names[comb] == "success"
        tally[tj] += 1
    assert tally["success"] and tally["fail"]       # the test exercised both


def test_compare_joint_separate_smoke():
    Hx, Hz, Lx, Lz = qcb.codes.toric_code(3)
    rng = np.random.default_rng(5)
    n = Hx.shape[1]
    # a non-uniform channel avoids the exact weight ties of uniform noise
    J = qcb.DecodingProblem.from_css_code_pauli(Hx, Hz, rng.uniform(0.01, 0.04, n),
                                                rng.uniform(0.01, 0.04, n),
                                                rng.uniform(0.01, 0.04, n))
    cmp = qcb.compare_joint_separate(J, shots=40, seed=1, seeders=None,
                                     joint_decoders=["bp_osd"],
                                     separate_decoders=["bp_osd"],
                                     num_workers=1, progress=False)
    assert cmp.shots == cmp.x.shots == cmp.z.shots == 40
    assert np.array_equal(cmp.separate_verdict,
                          qcb.combine_sector_verdicts(cmp.x.verdict, cmp.z.verdict))
    br = cmp.ml_brackets()
    for key in ("joint", "separate"):
        assert 0 <= br[key]["lb"] <= br[key]["ub"] <= 1
    pt = cmp.paired_test()
    assert 0 <= pt["joint_better"]["p_value"] <= 1
    assert cmp.joint_decoders() == ["bp_osd"] and cmp.separate_decoders() == ["bp_osd"]
    e = cmp.decoder_ler("bp_osd", "separate")
    assert e["shots"] == 40
    assert cmp.joint.fault_counts.min() >= 0          # recorded from the joint sampler
    assert "Joint vs separate ML" in str(cmp)
