import numpy as np
import pytest


def brute_class_minima(H, L, priors, s):
    """Minimum cost per logical class over all e with H e = s (n <= ~20)."""
    m, n = H.shape
    k = L.shape[0]
    w = np.log((1 - priors) / priors)
    X = ((np.arange(1 << n, dtype=np.int64)[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    syn = (X @ H.T.astype(np.int64)) % 2
    ok = (syn == s[None, :]).all(axis=1)
    lam = (X[ok] @ L.T.astype(np.int64)) % 2
    cls = lam @ (1 << np.arange(k))
    cost = X[ok].astype(np.float64) @ w
    out = np.full(1 << k, np.inf)
    np.minimum.at(out, cls, cost)
    return out


@pytest.fixture
def small_problem():
    import qldpc_certificate_benchmark as qcb
    rng = np.random.default_rng(3)
    m, n, k = 9, 16, 2
    H = (rng.random((m, n)) < 0.3).astype(np.uint8)
    for j in range(n):
        if not H[:, j].any():
            H[rng.integers(m), j] = 1
    L = (rng.random((k, n)) < 0.4).astype(np.uint8)
    priors = rng.uniform(0.02, 0.2, size=n)
    return qcb.DecodingProblem(H, L, priors, name="random-small")
