import pytest


@pytest.mark.slow
def test_engine_selftest():
    import qldpc_certificate_benchmark as qcb
    qcb.selftest()
