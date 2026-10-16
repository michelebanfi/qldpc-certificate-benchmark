"""Command line: ``python -m qldpc_certificate_benchmark {selftest,demo,version}``."""

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qldpc-certificate-benchmark",
                                 description="Certified ML decoder benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="brute-force validation of the certifier")
    d = sub.add_parser("demo", help="certify BP+OSD variants on the [[72,12,6]] BB code")
    d.add_argument("--p", type=float, default=0.05)
    d.add_argument("--shots", type=int, default=500)
    d.add_argument("--workers", type=int, default=None)
    sub.add_parser("version")
    a = ap.parse_args(argv)

    import qldpc_certificate_benchmark as qcb
    if a.cmd == "version":
        print(qcb.__version__)
    elif a.cmd == "selftest":
        qcb.selftest()
    elif a.cmd == "demo":
        Hx, Hz, Lx, Lz = qcb.codes.bb_72_12_6()
        prob = qcb.DecodingProblem.from_css_code(Hx, Hz, a.p, name="[[72,12,6]] X-errors")
        res = qcb.benchmark(prob, decoders=[qcb.BPOSD(osd_method="osd0", name="bp+osd0"),
                                            qcb.BPOSD()],
                            shots=a.shots, num_workers=a.workers)
        print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
