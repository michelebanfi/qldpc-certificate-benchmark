"""Circuit-level certified benchmark from a Stim circuit.

Uses PyMatching (pip install pymatching) as the decoder under test, driven
through ``certify_shots`` with Stim's own sampler and batch decoding.
"""
import stim
import qldpc_certificate_benchmark as qcb

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x", distance=5, rounds=5,
    after_clifford_depolarization=0.005, before_measure_flip_probability=0.005,
    after_reset_flip_probability=0.005)
problem = qcb.DecodingProblem.from_stim_circuit(circuit, name="rotated surface d=5, p=0.005")

if __name__ == "__main__":
    dets, obs = circuit.compile_detector_sampler(seed=0).sample(
        2000, separate_observables=True)
    predictions = {}
    try:
        import pymatching
        m = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True))
        predictions["pymatching"] = m.decode_batch(dets)
    except ImportError:
        print("pymatching not installed; certifying BP+OSD only")
    result = qcb.certify_shots(problem, dets, obs, predictions=predictions)
    print(result)
