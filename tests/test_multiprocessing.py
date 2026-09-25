"""Forking after scipy's HiGHS solver has run in the parent can deadlock the
workers.  The driver must detect a multi-threaded parent and use a clean start."""
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent("""
    import numpy as np
    import qldpc_certificate_benchmark as qcb
    from qldpc_certificate_benchmark.benchmark import _native_thread_count

    Hx, Hz, Lx, Lz = qcb.codes.bb_72_12_6()
    prob = qcb.DecodingProblem.from_code(Hz, Lz, 0.07)
    # small budgets force many LP / exact-ILP solver calls in every worker
    cfg = qcb.CertConfig(true_node_budget=300, wrong_node_budget=300,
                         cluster_node_budget=300, unified_node_budget=300,
                         escalate=False)
    if __name__ == "__main__":
        # serial run first: the parent process now hosts HiGHS' thread pool
        qcb.benchmark(prob, [qcb.BPOSD(osd_method="osd0", name="osd0", role="probe")], shots=20, num_workers=1, config=cfg,
                      progress=False, seeders=None)
        print("threads after serial run:", _native_thread_count())
        r = qcb.benchmark(prob, [qcb.BPOSD(osd_method="osd0", name="osd0", role="probe")], shots=160, num_workers=4, config=cfg,
                          progress=False, seeders=None)
        print("OK", r.shots)
""")


@pytest.mark.skipif(sys.platform == "win32", reason="fork-specific")
def test_no_deadlock_after_parent_used_solver(tmp_path):
    f = tmp_path / "run.py"
    f.write_text(SCRIPT)
    out = subprocess.run([sys.executable, str(f)], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK 160" in out.stdout, out.stdout + out.stderr[-2000:]
