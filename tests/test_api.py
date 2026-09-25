import numpy as np
import pytest

import qldpc_certificate_benchmark as qcb


def test_problem_validation():
    H = np.array([[1, 1, 0], [0, 1, 1]])
    L = np.array([[1, 0, 0]])
    qcb.DecodingProblem(H, L, 0.1)
    with pytest.raises(ValueError):
        qcb.DecodingProblem(H, L, 0.6)                      # p >= 0.5
    with pytest.raises(ValueError):
        qcb.DecodingProblem(H, np.array([[1, 0]]), 0.1)     # shape mismatch
    with pytest.raises(ValueError):                          # undetectable logical
        qcb.DecodingProblem(np.array([[1, 1, 0]]), np.array([[0, 0, 1]]), 0.1)


def test_codes_are_valid():
    for Hx, Hz, Lx, Lz, k in [(*qcb.codes.toric_code(3), 2), (*qcb.codes.bb_72_12_6(), 12)]:
        assert not (Hx.astype(int) @ Hz.T % 2).any()
        assert not (Hx.astype(int) @ Lz.T % 2).any()
        assert not (Hz.astype(int) @ Lx.T % 2).any()
        assert Lx.shape[0] == Lz.shape[0] == k
        # Lx . Lz^T must be invertible mod 2
        assert qcb.codes.gf2_rank(Lx.astype(int) @ Lz.T % 2) == k


def test_from_css_code_matches_explicit_logicals():
    Hx, Hz, Lx, Lz = qcb.codes.bb_72_12_6()
    a = qcb.DecodingProblem.from_css_code(Hx, Hz, 0.05, error_type="X")
    b = qcb.DecodingProblem.from_code(Hz, Lz, 0.05)
    assert np.array_equal(a.H, b.H) and a.num_observables == 12
    # a different logical basis only relabels classes: identical verdicts
    rng = np.random.default_rng(0)
    A = np.eye(12, dtype=np.int64)
    A[0] ^= A[3]; A[5] ^= A[1]                       # invertible basis change
    stab = rng.integers(0, 2, (12, Hz.shape[0])) @ Hz % 2   # + stabilizer parts
    c = qcb.DecodingProblem(Hz, (A @ Lz + stab) % 2, 0.05)
    syn, obs_b = b.sample(40, seed=1, return_errors=True)[:2]
    _, _, e = b.sample(40, seed=1, return_errors=True)
    ra = qcb.certify_shots(b, syn, b.observables(e), seeders=None, num_workers=1,
                           progress=False)
    rc = qcb.certify_shots(c, syn, c.observables(e), seeders=None, num_workers=1,
                           progress=False)
    assert np.array_equal(ra.verdict, rc.verdict)
    with pytest.raises(ValueError):
        qcb.DecodingProblem.from_css_code(Hx, Lx, 0.05)       # not a CSS pair


def test_user_decoders_and_roles(small_problem):
    H, L = small_problem.H, small_problem.L

    def zero_correction(s):            # always returns the trivial (invalid) correction
        return np.zeros(H.shape[1], dtype=np.uint8)

    def obs_guess(s):                  # predicts "no logical flip"
        return np.zeros(L.shape[0], dtype=np.uint8)

    res = qcb.benchmark(small_problem,
                        {"zero": qcb.FunctionDecoder(zero_correction, output="correction"),
                         "guess": qcb.FunctionDecoder(obs_guess, output="observables"),
                         "osd0": qcb.BPOSD(osd_method="osd0", name="osd0", role="probe")},
                        shots=80, seed=1, num_workers=1, progress=False)
    roles = dict(res.decoders)
    assert roles["zero"] == "probe" and roles["bp_osd"] == "seed"
    # zero correction is invalid on every shot with a nonzero syndrome
    syn, obs = small_problem.sample(80, seed=1)
    assert res.invalid["zero"].sum() == int(syn.any(axis=1).sum())
    assert res.ler("guess")["failures"] == int(obs.any(axis=1).sum())
    s = res.summary()
    assert s["decoders"]["osd0"]["shots"] == 80
    str(res)


def test_parallel_matches_serial(small_problem):
    kw = dict(decoders=[qcb.BPOSD(osd_method="osd0", name="osd0", role="probe")],
              shots=64, seed=7, progress=False, seeders=["bp_osd"])
    a = qcb.benchmark(small_problem, num_workers=1, **kw)
    b = qcb.benchmark(small_problem, num_workers=3, **kw)
    assert np.array_equal(a.verdict, b.verdict)
    assert np.array_equal(a.ok["osd0"], b.ok["osd0"])


def test_certify_shots_matches_benchmark(small_problem):
    syn, obs = small_problem.sample(50, seed=2)
    pred = np.zeros_like(obs)
    res = qcb.certify_shots(small_problem, syn, obs, predictions={"zero": pred},
                            num_workers=1, progress=False)
    ref = qcb.benchmark(small_problem, [], shots=50, seed=2, num_workers=1,
                        progress=False)
    assert np.array_equal(res.verdict, ref.verdict)
    assert res.ler("zero")["failures"] == int(obs.any(axis=1).sum())


def test_target_failures_prefix(small_problem):
    res = qcb.benchmark(small_problem, [qcb.BPOSD(osd_method="osd0", max_iter=2,
                                                  name="weak", role="probe")],
                        shots=400, seed=0, num_workers=2,
                        target_failures=5, progress=False)
    assert res.shots < 400
    assert (~res.ok["weak"]).sum() >= 5


def test_save_roundtrip(tmp_path, small_problem):
    res = qcb.benchmark(small_problem, [], shots=20, num_workers=1, progress=False)
    j, z = res.save(tmp_path / "run")
    d = np.load(z)
    assert np.array_equal(d["verdict"], res.verdict)
