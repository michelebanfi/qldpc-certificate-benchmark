import numpy as np
import pytest

import qldpc_certificate_benchmark as qcb


class Broken(qcb.Decoder):
    name, output = "broken", "observables"

    def prepare(self, problem):
        raise ValueError("boom in prepare")

    def decode(self, s):
        return np.zeros(1)


def test_worker_init_error_is_raised_not_hung(small_problem):
    with pytest.raises(RuntimeError, match="boom in prepare"):
        qcb.benchmark(small_problem, [Broken()], shots=8, num_workers=2,
                      progress=False, seeders=None)
