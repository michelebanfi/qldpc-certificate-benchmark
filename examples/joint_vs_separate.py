"""Joint vs separate ML decoding on the [[72,12,6]] code under Pauli noise.

Separate decoding gives each CSS sector its marginal noise (an X error with
probability px + py, a Z error with pz + py) and decodes the two sectors
independently.  The joint model keeps a Y as ONE mechanism that flips both
sectors.  ``compare_joint_separate`` certifies both ML decisions on the same
shots and runs the decoders in both modes.

A non-uniform channel is used on purpose: with uniform priors many solutions
tie exactly, which widens both brackets.
"""
import numpy as np

import qldpc_certificate_benchmark as qcb

Hx, Hz, Lx, Lz = qcb.codes.bb_72_12_6()
rng = np.random.default_rng(0)
n = Hx.shape[1]
px, py, pz = (rng.uniform(0.01, 0.03, n) for _ in range(3))
joint = qcb.DecodingProblem.from_css_code_pauli(Hx, Hz, px, py, pz,
                                                name="[[72,12,6]] Pauli noise")

# "your decoder" (role PROBE), run jointly and separately; the default seeders
# (BP+OSD-CS, Tesseract when installed) are reported in both modes as well
mine = qcb.BPOSD(osd_method="osd0", name="bp+osd0", role="probe")

if __name__ == "__main__":
    cmp = qcb.compare_joint_separate(joint, shots=2000, seed=0,
                                     joint_decoders=[mine], separate_decoders=[mine])
    print(cmp)
    print(cmp.paired_test())
    print(cmp.decoder_vs_ml("bp+osd0", mode="joint", ml="separate"))
