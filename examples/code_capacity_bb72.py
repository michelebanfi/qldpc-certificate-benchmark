"""Certified benchmark on the [[72,12,6]] bivariate-bicycle code, code-capacity noise.

Measures two BP+OSD configurations from Roffe's ``ldpc`` package against the
certified ML bracket:

* ``bp+osd0``   -- BP + OSD-0 (standing in for "your decoder", role PROBE)
* ``bp_osd``    -- BP + OSD-CS order 7, the default seeder (role SEED)
"""
import qldpc_certificate_benchmark as qcb

Hx, Hz, Lx, Lz = qcb.codes.bb_72_12_6()
problem = qcb.DecodingProblem.from_css_code(Hx, Hz, p=0.04, error_type="X",
                                            name="[[72,12,6]] X errors, p=0.04")
mine = qcb.BPOSD(osd_method="osd0", name="bp+osd0", role="probe")

if __name__ == "__main__":
    result = qcb.benchmark(problem, [mine], shots=2000, seed=0)
    print(result)
    print(result.compare("bp+osd0", "bp_osd"))
