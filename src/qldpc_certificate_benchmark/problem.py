"""The decoding problem: a binary check matrix, logical matrix, and priors.

Everything the certifier needs is a *detector error model* in matrix form:

* ``H``      -- (m, n) binary matrix.  Column j is error mechanism j, row i is
                detector / parity check i.  ``H[i, j] = 1`` iff mechanism j
                flips detector i.
* ``L``      -- (k, n) binary matrix.  ``L[a, j] = 1`` iff mechanism j flips
                logical observable a.
* ``priors`` -- (n,) independent firing probabilities, each in (0, 0.5).

A shot draws each mechanism independently, and the decoder sees the syndrome
``s = H e mod 2``.  It must predict ``L e mod 2``.  This covers code-capacity
noise (``H`` = a parity-check matrix, ``L`` = its logical operators), phenomenological
noise, and full circuit-level noise via a Stim detector error model.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as _sp

__all__ = ["DecodingProblem"]

_MAX_OBSERVABLES = 62   # logical classes are packed into a signed int64


def _as_dense_binary(M, name):
    if _sp.issparse(M):
        M = M.toarray()
    M = np.asarray(M)
    if M.ndim == 1:
        M = M[None, :]
    if M.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {M.shape}")
    if M.size and not np.isin(M, (0, 1)).all():
        # allow integer matrices that are binary mod 2 only if already 0/1
        raise ValueError(f"{name} must contain only 0/1 entries")
    return np.ascontiguousarray(M, dtype=np.uint8)


class DecodingProblem:
    """An independent-error decoding problem ``(H, L, priors)``.

    Parameters
    ----------
    check_matrix : array-like or scipy.sparse, shape (m, n)
        Detector / parity-check matrix.
    logical_matrix : array-like or scipy.sparse, shape (k, n)
        Logical-observable matrix (1 <= k <= 62).
    priors : float or array-like, shape (n,)
        Independent error probability of each column.  A scalar applies to all
        columns (code-capacity noise).
    name : str, optional
        Label used in reports.

    Notes
    -----
    Columns with zero syndrome and zero logical effect have no effect on any
    decision.  They are kept but ignored by the cost model.  A column with zero
    syndrome but nonzero logical effect is an undetectable logical error
    (distance 1).  Such a model is rejected, because no decoder can do anything
    about that column.
    """

    def __init__(self, check_matrix, logical_matrix, priors, name: str = "problem"):
        H = _as_dense_binary(check_matrix, "check_matrix")
        L = _as_dense_binary(logical_matrix, "logical_matrix")
        m, n = H.shape
        if L.shape[1] != n:
            raise ValueError(f"check_matrix has {n} columns but logical_matrix has "
                             f"{L.shape[1]}")
        k = L.shape[0]
        if not 1 <= k <= _MAX_OBSERVABLES:
            raise ValueError(f"need 1 <= #logical observables <= {_MAX_OBSERVABLES}, "
                             f"got {k}")
        if m == 0 or n == 0:
            raise ValueError("check_matrix must be non-empty")
        pri = np.asarray(priors, dtype=np.float64)
        if pri.ndim == 0:
            pri = np.full(n, float(pri))
        if pri.shape != (n,):
            raise ValueError(f"priors must be a scalar or have shape ({n},), got {pri.shape}")
        if not np.all(np.isfinite(pri)) or (pri < 0).any():
            raise ValueError("priors must be finite and non-negative")

        zero_syn = ~H.any(axis=0)
        bad = zero_syn & L.any(axis=0) & (pri > 0)
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} column(s) (e.g. {np.flatnonzero(bad)[:5].tolist()}) "
                f"flip a logical observable without flipping any detector: the model "
                f"has an undetectable logical error (distance 1)")
        constrained = ~zero_syn | L.any(axis=0)
        if (pri[constrained] >= 0.5).any():
            j = int(np.flatnonzero(constrained & (pri >= 0.5))[0])
            raise ValueError(f"prior of column {j} is {pri[j]:.4g} >= 0.5: every "
                             f"constrained column needs 0 <= p < 0.5")
        # A zero-probability column can never fire.  Give it a tiny prior so its
        # weight is finite (it is clipped to 50 nats anyway).
        pri = np.where(pri <= 0.0, 1e-300, pri)

        self.H = H
        self.L = L
        self.priors = pri
        self.name = str(name)
        self._H_csr = None
        self._L_csr = None

    # ------------------------------------------------------------------ shape
    @property
    def num_detectors(self) -> int:
        return self.H.shape[0]

    @property
    def num_mechanisms(self) -> int:
        return self.H.shape[1]

    @property
    def num_observables(self) -> int:
        return self.L.shape[0]

    def __repr__(self):
        return (f"DecodingProblem(name={self.name!r}, detectors={self.num_detectors}, "
                f"mechanisms={self.num_mechanisms}, observables={self.num_observables})")

    # --------------------------------------------------------- constructors
    @classmethod
    def from_code(cls, parity_check, logicals, p, name: str = "code"):
        """Code-capacity problem: each qubit flips independently.

        For a CSS code, pass ``Hz`` with ``Lz`` to decode X errors, or ``Hx``
        with ``Lx`` to decode Z errors.  ``p`` is a scalar or a per-qubit array.
        If you do not have the logical operators, use :meth:`from_css_code`.
        """
        return cls(parity_check, logicals, p, name=name)

    @classmethod
    def from_css_code(cls, Hx, Hz, p, error_type: str = "X", name: str = "css"):
        """Code-capacity problem for a CSS code, with logicals computed for you.

        Logical operators are not a function of one check matrix alone.  To
        decide whether an X error is a stabilizer or a logical operator you need
        both ``Hz`` (which detects it) and ``Hx`` (whose row space is the
        stabilizer group).  This constructor computes a basis of Z logicals
        (``ker Hx`` mod ``rowspace Hz``) and its X analogue with GF(2) elimination.

        ``error_type="X"`` decodes X errors: ``H = Hz`` and ``L = Lz``.
        ``error_type="Z"`` decodes Z errors: ``H = Hx`` and ``L = Lx``.
        The certificate does not depend on which logical basis is chosen:
        another basis only relabels the classes.
        """
        from .codes import css_logicals
        Hx = _as_dense_binary(Hx, "Hx")
        Hz = _as_dense_binary(Hz, "Hz")
        if Hx.shape[1] != Hz.shape[1]:
            raise ValueError("Hx and Hz must have the same number of columns")
        if (Hx.astype(np.int64) @ Hz.T.astype(np.int64) % 2).any():
            raise ValueError("Hx Hz^T != 0 mod 2: not a valid CSS code")
        Lx, Lz = css_logicals(Hx, Hz)
        if Lx.shape[0] == 0:
            raise ValueError("the code encodes no logical qubits (k = 0)")
        if error_type.upper() == "X":
            return cls(Hz, Lz, p, name=name)
        if error_type.upper() == "Z":
            return cls(Hx, Lx, p, name=name)
        raise ValueError(f"error_type must be 'X' or 'Z', got {error_type!r}")

    @classmethod
    def from_dem(cls, dem, merge_duplicates: bool = True, name: str = "dem"):
        """Build from a Stim detector error model.

        ``dem`` may be a ``stim.DetectorErrorModel``, DEM text, or a path to a
        ``.dem`` file.  ``repeat`` blocks and ``shift_detectors`` are flattened.
        ``^`` decomposition separators are ignored: the targets of one
        ``error(p)`` instruction are XOR-ed into a single mechanism.  With
        ``merge_duplicates`` (the default), mechanisms with identical
        (detector, observable) support are merged into one column with the
        exact combined probability ``p1(1-p2) + p2(1-p1)``.  The merged model
        has the same syndrome and observable distribution.
        """
        try:
            import stim
        except ImportError as e:  # pragma: no cover
            raise ImportError("from_dem needs stim: pip install stim") from e
        if isinstance(dem, str):
            import os
            if os.path.exists(dem):
                with open(dem) as fh:
                    dem = fh.read()
            dem = stim.DetectorErrorModel(dem)
        if not isinstance(dem, stim.DetectorErrorModel):
            raise TypeError(f"expected a stim.DetectorErrorModel, got {type(dem)}")

        m, k = dem.num_detectors, dem.num_observables
        cols = {}      # (dets tuple, obs tuple) -> prob     (merge mode)
        col_list = []  # [(dets, obs, prob)]                 (no-merge mode)
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            p = float(inst.args_copy()[0])
            dets, obs = set(), set()
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    dets ^= {t.val}
                elif t.is_logical_observable_id():
                    obs ^= {t.val}
            if not dets and not obs or p == 0.0:
                continue
            key = (tuple(sorted(dets)), tuple(sorted(obs)))
            if merge_duplicates:
                q = cols.get(key, 0.0)
                cols[key] = q * (1 - p) + p * (1 - q)
            else:
                col_list.append((key[0], key[1], p))
        if merge_duplicates:
            col_list = [(d, o, p) for (d, o), p in cols.items()]
        n = len(col_list)
        if n == 0:
            raise ValueError("the DEM contains no error mechanisms")
        H = np.zeros((m, n), dtype=np.uint8)
        L = np.zeros((k, n), dtype=np.uint8)
        pri = np.empty(n, dtype=np.float64)
        for j, (d, o, p) in enumerate(col_list):
            H[list(d), j] = 1
            L[list(o), j] = 1
            pri[j] = p
        return cls(H, L, pri, name=name)

    @classmethod
    def from_stim_circuit(cls, circuit, merge_duplicates: bool = True,
                          name: str = "circuit"):
        """Build from a ``stim.Circuit`` (or circuit text / path) via its DEM."""
        import stim
        if isinstance(circuit, str):
            import os
            if os.path.exists(circuit):
                with open(circuit) as fh:
                    circuit = fh.read()
            circuit = stim.Circuit(circuit)
        dem = circuit.detector_error_model(decompose_errors=False,
                                           approximate_disjoint_errors=True)
        return cls.from_dem(dem, merge_duplicates=merge_duplicates, name=name)

    # ------------------------------------------------------------ utilities
    def _csr(self):
        if self._H_csr is None:
            self._H_csr = _sp.csr_matrix(self.H, dtype=np.int64)
            self._L_csr = _sp.csr_matrix(self.L, dtype=np.int64)
        return self._H_csr, self._L_csr

    def syndrome(self, errors):
        """``H e mod 2`` for one error vector (n,) or a batch (shots, n)."""
        Hc, _ = self._csr()
        e = np.asarray(errors)
        return (np.asarray(Hc @ e.T).T % 2).astype(np.uint8)

    def observables(self, errors):
        """``L e mod 2`` for one error vector (n,) or a batch (shots, n)."""
        _, Lc = self._csr()
        e = np.asarray(errors)
        return (np.asarray(Lc @ e.T).T % 2).astype(np.uint8)

    def sample(self, shots: int, seed=None, return_errors: bool = False):
        """Draw ``shots`` independent shots from the priors.

        Returns ``(syndromes, observables)`` as uint8 arrays of shape
        ``(shots, m)`` and ``(shots, k)``, plus ``errors`` (shots, n) when
        ``return_errors`` is set.  Deterministic given ``seed``.
        """
        rng = np.random.default_rng(seed)
        n = self.num_mechanisms
        syn = np.empty((shots, self.num_detectors), dtype=np.uint8)
        obs = np.empty((shots, self.num_observables), dtype=np.uint8)
        errs = np.empty((shots, n), dtype=np.uint8) if return_errors else None
        chunk = max(1, min(shots, 4_000_000 // max(n, 1)))
        for a in range(0, shots, chunk):
            b = min(shots, a + chunk)
            e = (rng.random((b - a, n)) < self.priors).astype(np.uint8)
            syn[a:b] = self.syndrome(e)
            obs[a:b] = self.observables(e)
            if return_errors:
                errs[a:b] = e
        if return_errors:
            return syn, obs, errs
        return syn, obs
