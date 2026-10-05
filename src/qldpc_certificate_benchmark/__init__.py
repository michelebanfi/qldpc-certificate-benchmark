"""qldpc-certificate-benchmark: certified maximum-likelihood benchmarking for quantum decoders.

Given any decoding problem (a Stim detector error model, or a parity-check
matrix with its logical operators and priors), ``qldpc_certificate_benchmark`` *proves*, shot by
shot, whether the maximum-likelihood decoder would succeed.  This yields a
certified bracket on the ML logical error rate, which any decoder can be
compared against, with paired tests that account for the uncertified shots.

Quick start::

    import qldpc_certificate_benchmark as qcb
    Hx, Hz, Lx, Lz = qcb.codes.bb_72_12_6()
    problem = qcb.DecodingProblem.from_code(Hz, Lz, p=0.05)
    result = qcb.benchmark(problem, decoders={"mine": my_decode},
                          shots=2000)
    print(result)
"""

from ._version import __version__
from .config import CertConfig
from .problem import DecodingProblem
from .certifier import Certifier, Certificate, ROUTES
from .decoders import (Role, Decoder, FunctionDecoder, BPOSD, Tesseract, Simplex,
                       has_tesseract)
from .benchmark import benchmark, certify_shots
from .results import (BenchmarkResult, clopper_pearson, clopper_pearson_lower,
                      clopper_pearson_upper, population_bracket)
from . import codes


def selftest():
    """Run the brute-force validation suite of the certification engine
    (exhaustive enumeration on small random systems).  Takes a few minutes."""
    from ._engine.selftest import selftest as _st
    _st()


__all__ = [
    "__version__", "CertConfig", "DecodingProblem", "Certifier", "Certificate",
    "Role", "Decoder", "FunctionDecoder", "BPOSD", "Tesseract",
    "Simplex", "has_tesseract", "benchmark", "certify_shots", "BenchmarkResult",
    "clopper_pearson", "clopper_pearson_lower", "clopper_pearson_upper",
    "population_bracket", "ROUTES", "codes", "selftest",
]
