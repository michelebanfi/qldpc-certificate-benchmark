import numpy as np
import pytest

import qldpc_certificate_benchmark as qcb

stim = pytest.importorskip("stim")


def _circuit(d=3, rounds=3, p=0.004):
    return stim.Circuit.generated("surface_code:rotated_memory_x", distance=d,
                                  rounds=rounds, after_clifford_depolarization=p,
                                  before_measure_flip_probability=p,
                                  after_reset_flip_probability=p)


def test_from_dem_consistency():
    c = _circuit()
    dem = c.detector_error_model(decompose_errors=True)
    prob = qcb.DecodingProblem.from_dem(dem)
    assert prob.num_detectors == dem.num_detectors
    assert prob.num_observables == dem.num_observables
    # merged model: identical columns are merged
    cols = {tuple(np.r_[prob.H[:, j], prob.L[:, j]]) for j in range(prob.num_mechanisms)}
    assert len(cols) == prob.num_mechanisms
    # text and circuit constructors agree
    p2 = qcb.DecodingProblem.from_dem(str(dem))
    assert np.array_equal(prob.H, p2.H)
    p3 = qcb.DecodingProblem.from_stim_circuit(c)
    assert p3.num_mechanisms == prob.num_mechanisms


def test_stim_samples_certify():
    c = _circuit()
    prob = qcb.DecodingProblem.from_stim_circuit(c)
    syn, obs = c.compile_detector_sampler(seed=1).sample(100, separate_observables=True)
    res = qcb.certify_shots(prob, syn, obs, predictions={"none": np.zeros_like(obs)},
                            num_workers=1, progress=False)
    c = res.counts()
    assert c["undecided"] <= 5
    assert res.ml_bracket()["lb"] <= res.ler("bp_osd")["ler"] + 0.05


@pytest.mark.skipif(not qcb.has_tesseract(), reason="tesseract-decoder not installed")
def test_exact_mle_never_contradicts():
    prob = qcb.DecodingProblem.from_stim_circuit(_circuit(p=0.008))
    res = qcb.benchmark(prob, [], shots=60, seed=3, crosscheck=60,
                        num_workers=1, progress=False)
    assert res.crosscheck_contradictions() == {"simplex": 0}
