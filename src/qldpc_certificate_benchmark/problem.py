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

Joint X/Z decoding: for a CSS code the X and Z parts of a fault are detected by
different checks, but a ``Y`` (or a correlated two-qubit fault after a CNOT)
flips both at once.  A *joint* model keeps such a fault as ONE mechanism with
both signatures; *separate* decoding sees two independent faults, one per
sector, with marginal probabilities.  :class:`CSSPartition` records which
detectors and observables belong to each sector, :meth:`DecodingProblem.css_split`
returns the two marginal (separate) models of a joint one, and
:meth:`~DecodingProblem.from_css_code_pauli` /
:meth:`~DecodingProblem.from_css_sectors` build joint models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse as _sp

__all__ = ["DecodingProblem", "CSSPartition"]

_MAX_OBSERVABLES = 62   # logical classes are packed into a signed int64


@dataclass(frozen=True)
class CSSPartition:
    """Which detectors and observables belong to each CSS sector of a joint model.

    The *X sector* is what separate decoding of X-type errors sees: the
    detectors an X error can flip (Z-type checks) and the observables an X
    error can flip.  The *Z sector* is the same for Z-type errors.  Every
    detector and every observable must belong to exactly one sector, and each
    sector needs at least one observable.

    Index lists are stored as tuples (hashable, picklable); pass any iterables.
    """
    x_detectors: tuple
    z_detectors: tuple
    x_observables: tuple
    z_observables: tuple

    def __post_init__(self):
        for f in ("x_detectors", "z_detectors", "x_observables", "z_observables"):
            object.__setattr__(self, f, tuple(int(i) for i in np.asarray(
                getattr(self, f), dtype=np.int64).ravel()))

    @classmethod
    def blocks(cls, m_x, m_z, k_x, k_z):
        """The partition of a model whose rows are ``[X detectors; Z detectors]``
        and observables ``[X observables; Z observables]``."""
        return cls(range(m_x), range(m_x, m_x + m_z), range(k_x), range(k_x, k_x + k_z))

    def arrays(self):
        """``(x_detectors, z_detectors, x_observables, z_observables)`` as int64 arrays."""
        return tuple(np.asarray(getattr(self, f), dtype=np.int64) for f in
                     ("x_detectors", "z_detectors", "x_observables", "z_observables"))

    def validate(self, num_detectors, num_observables):
        xd, zd, xo, zo = self.arrays()
        for name, a, b, total in (("detectors", xd, zd, num_detectors),
                                  ("observables", xo, zo, num_observables)):
            both = np.concatenate([a, b])
            if both.size and (both.min() < 0 or both.max() >= total):
                raise ValueError(f"CSSPartition: {name} index out of range [0, {total})")
            if np.unique(both).size != both.size:
                raise ValueError(f"CSSPartition: a {name[:-1]} is listed twice or in both "
                                 f"sectors")
            if both.size != total:
                raise ValueError(f"CSSPartition: {total - both.size} {name} belong to "
                                 f"neither sector (every one must be X or Z)")
        if xo.size == 0 or zo.size == 0:
            raise ValueError("CSSPartition: each sector needs at least one observable")


def _merge_columns(H, L, priors):
    """Merge identical ``(H; L)`` columns of independent mechanisms and drop
    columns with no effect.  The merged column fires iff an odd number of its
    mechanisms fire: ``1 - 2q = prod(1 - 2p)`` (exact, same law as
    ``from_dem``'s ``p1(1-p2) + p2(1-p1)``).  First-appearance order is kept."""
    sig = np.vstack([H, L]).astype(np.uint8)
    cols = np.flatnonzero(sig.any(axis=0))
    if cols.size == 0:
        return H[:, :0], L[:, :0], np.zeros(0)
    packed = np.packbits(sig[:, cols].T, axis=1)
    _, first, inv = np.unique(packed, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    order = np.argsort(first, kind="stable")          # groups by first appearance
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    g = rank[inv]                                     # group index of every column
    logs = np.zeros(order.size)
    p = np.minimum(np.asarray(priors, dtype=np.float64)[cols], 0.5)
    with np.errstate(divide="ignore"):
        np.add.at(logs, g, np.log1p(-2.0 * p))
    q = -np.expm1(logs) / 2.0
    rep = cols[first[order]]
    return H[:, rep], L[:, rep], q


def _independent_pauli(px, py, pz):
    """Probabilities ``(qx, qy, qz)`` of three INDEPENDENT X, Y, Z mechanisms
    whose combined effect is exactly the Pauli channel ``(px, py, pz)``.

    On one qubit the mechanisms multiply, so in Fourier (commutation)
    coordinates ``(1-2qy)(1-2qz) = 1-2(py+pz)`` and cyclically; solving gives
    ``1 - 2qx = sqrt(fY fZ / fX)`` with ``fX = 1-2(py+pz)``, etc.  Depolarizing
    noise ``px = py = pz = p/3`` gives ``q = (1 - sqrt(1 - 4p/3)) / 2``.

    Such ``q >= 0`` exist iff ``px (1 - p) >= py pz`` and cyclically, with
    ``p = px + py + pz`` (two independent flips combine into the third Pauli,
    so a channel with a rare Pauli and two common ones is not a product)."""
    px, py, pz = (np.asarray(v, dtype=np.float64) for v in (px, py, pz))
    if (px < 0).any() or (py < 0).any() or (pz < 0).any():
        raise ValueError("Pauli probabilities must be non-negative")
    fX, fY, fZ = 1 - 2 * (py + pz), 1 - 2 * (px + pz), 1 - 2 * (px + py)
    if (fX <= 0).any() or (fY <= 0).any() or (fZ <= 0).any():
        raise ValueError("every marginal flip probability (px+py, pz+py, px+pz) must "
                         "be < 1/2")
    A = np.sqrt(fY * fZ / fX)
    B = np.sqrt(fX * fZ / fY)
    C = np.sqrt(fX * fY / fZ)
    if (A > 1 + 1e-12).any() or (B > 1 + 1e-12).any() or (C > 1 + 1e-12).any():
        raise ValueError("this Pauli channel is not a product of independent X, Y and "
                         "Z flips: it needs px(1-p) >= py*pz and cyclically "
                         "(p = px+py+pz); build the joint model with from_css_sectors "
                         "instead")
    return tuple(np.clip((1 - v) / 2, 0.0, 0.5) for v in (A, B, C))


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
    css : CSSPartition, optional
        For a joint X/Z model: which detectors and observables belong to each
        CSS sector (enables :meth:`css_split`).  The joint constructors set it.

    Notes
    -----
    Columns with zero syndrome and zero logical effect have no effect on any
    decision.  They are kept but ignored by the cost model.  A column with zero
    syndrome but nonzero logical effect is an undetectable logical error
    (distance 1).  Such a model is rejected, because no decoder can do anything
    about that column.
    """

    def __init__(self, check_matrix, logical_matrix, priors, name: str = "problem",
                 css: CSSPartition = None):
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

        if css is not None:
            if not isinstance(css, CSSPartition):
                raise TypeError(f"css must be a CSSPartition, got {type(css)}")
            css.validate(m, k)

        self.H = H
        self.L = L
        self.priors = pri
        self.name = str(name)
        self.css = css
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

    @classmethod
    def from_css_sectors(cls, x_sector, z_sector, mechanisms, name: str = "joint"):
        """Joint X/Z model from the two sector models and the correlated faults.

        ``x_sector`` / ``z_sector`` are the models separate decoding uses: a
        :class:`DecodingProblem` (only ``H`` and ``L`` are read) whose columns
        are the X parts (resp. Z parts) a fault can have.  ``mechanisms`` lists
        every elementary fault as ``(x_col, z_col, p)``: the column of its X
        part in ``x_sector`` and of its Z part in ``z_sector`` (``-1`` for no
        part), and its independent firing probability.

        A Clifford CSS circuit propagates the X and Z parts of a Pauli fault
        independently, so the fault's joint signature is the pair of its two
        sector columns.  Rows are ``[X-sector detectors; Z-sector detectors]``
        and ``[X-sector observables; Z-sector observables]`` (the
        :class:`CSSPartition` is set).  Faults with the same signature are
        merged exactly (an odd number of them fires).  :meth:`css_split` of the
        result returns the marginal sector models implied by ``mechanisms``.
        """
        mech = np.asarray(mechanisms, dtype=np.float64)
        if mech.ndim != 2 or mech.shape[1] != 3:
            raise ValueError("mechanisms must be a sequence of (x_col, z_col, p)")
        xs = mech[:, 0].astype(np.int64)
        zs = mech[:, 1].astype(np.int64)
        p = mech[:, 2]
        if not np.array_equal(xs, mech[:, 0]) or not np.array_equal(zs, mech[:, 1]):
            raise ValueError("x_col / z_col must be integers")
        nx, nz = x_sector.num_mechanisms, z_sector.num_mechanisms
        if (xs < -1).any() or (xs >= nx).any() or (zs < -1).any() or (zs >= nz).any():
            raise ValueError("a mechanism refers to a sector column out of range "
                             "(use -1 for no part)")
        if not np.all(np.isfinite(p)) or (p < 0).any() or (p >= 0.5).any():
            raise ValueError("mechanism probabilities must lie in [0, 0.5)")
        mx, mz = x_sector.num_detectors, z_sector.num_detectors
        kx, kz = x_sector.num_observables, z_sector.num_observables
        M = xs.size
        H = np.zeros((mx + mz, M), dtype=np.uint8)
        L = np.zeros((kx + kz, M), dtype=np.uint8)
        hx, hz = xs >= 0, zs >= 0
        H[:mx, hx] = x_sector.H[:, xs[hx]]
        L[:kx, hx] = x_sector.L[:, xs[hx]]
        H[mx:, hz] = z_sector.H[:, zs[hz]]
        L[kx:, hz] = z_sector.L[:, zs[hz]]
        H, L, q = _merge_columns(H, L, p)
        if q.size == 0:
            raise ValueError("no mechanism has any effect")
        return cls(H, L, q, name=name, css=CSSPartition.blocks(mx, mz, kx, kz))

    @classmethod
    def from_css_code_pauli(cls, Hx, Hz, px, py, pz, name: str = "css_pauli"):
        """Joint code-capacity model of a CSS code under the Pauli channel
        ``(px, py, pz)`` (scalars or per-qubit arrays).

        Each qubit carries three independent mechanisms, X, Y and Z, with
        probabilities chosen so that their combined effect is EXACTLY the
        channel (not its first-order approximation).  This needs
        ``px (1 - p) >= py pz`` and cyclically (``p = px + py + pz``): true for
        depolarizing and moderately biased noise.  The two marginal models,
        ``css_split()``, are then exactly ``from_css_code(Hx, Hz, px + py, "X")``
        and ``from_css_code(Hx, Hz, pz + py, "Z")``: what separate decoding uses.
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
        if 2 * Lx.shape[0] > _MAX_OBSERVABLES:
            raise ValueError(f"a joint model has 2k = {2 * Lx.shape[0]} observables; "
                             f"at most {_MAX_OBSERVABLES} are supported")
        n = Hx.shape[1]
        qx, qy, qz = (np.broadcast_to(v, (n,)) for v in _independent_pauli(px, py, pz))
        x_sector = cls(Hz, Lz, 0.25, name=f"{name}:X")    # priors unused here
        z_sector = cls(Hx, Lx, 0.25, name=f"{name}:Z")
        qubits = np.arange(n)
        none = np.full(n, -1)
        mech = np.concatenate([
            np.column_stack([qubits, none, qx]),       # X on qubit q
            np.column_stack([none, qubits, qz]),       # Z on qubit q
            np.column_stack([qubits, qubits, qy]),     # Y on qubit q: both parts
        ])
        return cls.from_css_sectors(x_sector, z_sector, mech, name=name)

    @classmethod
    def from_css_code_depolarizing(cls, Hx, Hz, p, name: str = "css_depolarizing"):
        """Joint code-capacity model under depolarizing noise: X, Y, Z each with
        probability ``p/3`` (exactly).  Separate decoding of the same noise is
        ``from_css_code(Hx, Hz, 2p/3, ...)`` per sector, which is what
        ``css_split()`` returns."""
        p = np.asarray(p, dtype=np.float64)
        return cls.from_css_code_pauli(Hx, Hz, p / 3, p / 3, p / 3, name=name)

    def css_split(self, partition: CSSPartition = None):
        """The two marginal (separate-decoding) models of a joint X/Z model.

        Returns ``(x_problem, z_problem)``.  The X-sector model keeps only the
        X-sector detector and observable rows; its columns are the distinct X
        parts of the joint mechanisms, each with the exact probability that an
        odd number of the mechanisms sharing it fires.  Because the joint
        mechanisms are independent, so are these, and the X-sector model has
        exactly the joint model's marginal law of (X syndrome, X observables).
        Separate decoding certifies each sector on the matching rows of a joint
        shot: ``syndrome[x_detectors]`` and ``observables[x_observables]``.

        ``partition`` defaults to :attr:`css` (set by the joint constructors);
        pass one for a joint model built otherwise, e.g. from a Stim circuit.
        """
        part = self.css if partition is None else partition
        if part is None:
            raise ValueError("this problem has no CSSPartition: pass "
                             "partition=CSSPartition(x_detectors, z_detectors, "
                             "x_observables, z_observables)")
        part.validate(self.num_detectors, self.num_observables)
        xd, zd, xo, zo = part.arrays()
        out = []
        for tag, det, obs in (("X", xd, xo), ("Z", zd, zo)):
            H, L, q = _merge_columns(self.H[det], self.L[obs], self.priors)
            if q.size == 0:
                raise ValueError(f"no mechanism acts on the {tag} sector")
            try:
                out.append(DecodingProblem(H, L, q, name=f"{self.name}:{tag}"))
            except ValueError as e:
                raise ValueError(f"{tag} sector of the split: {e} -- check that the "
                                 f"partition assigns every observable to the sector "
                                 f"whose detectors see it") from e
        return tuple(out)

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

    @property
    def fingerprint(self) -> str:
        """Short hash of ``(H, L, priors)``: results pool only on equal problems."""
        import hashlib
        h = hashlib.sha1(np.ascontiguousarray(self.H).tobytes())
        h.update(np.ascontiguousarray(self.L).tobytes())
        h.update(np.ascontiguousarray(self.priors).tobytes())
        h.update(repr((self.H.shape, self.L.shape)).encode())
        return h.hexdigest()[:16]

    def fault_count_pmf(self, hi: int):
        """Exact ``P(K = c)``, ``c = 0..hi``, for ``K`` = number of fired mechanisms."""
        from ._sampling import fault_count_pmf
        return fault_count_pmf(self.priors, hi)

    def band_mass(self, lo: int, hi: int) -> float:
        """Exact ``P(lo <= K <= hi)`` under the priors."""
        lo, hi = int(lo), int(hi)
        if not 0 <= lo <= hi <= self.num_mechanisms:
            raise ValueError(f"need 0 <= lo <= hi <= {self.num_mechanisms}, "
                             f"got [{lo}, {hi}]")
        return float(min(1.0, self.fault_count_pmf(hi)[lo:].sum()))

    def sample(self, shots: int, seed=None, return_errors: bool = False,
               fault_band=None, return_fault_counts: bool = False):
        """Draw ``shots`` independent shots from the priors.

        Returns ``(syndromes, observables)`` as uint8 arrays of shape
        ``(shots, m)`` and ``(shots, k)``, then ``errors`` (shots, n) when
        ``return_errors`` is set, then the fault counts ``K`` (shots,) when
        ``return_fault_counts`` is set.  Deterministic given ``seed``.

        ``fault_band=(lo, hi)`` samples from the priors *conditioned on*
        ``lo <= K <= hi`` (exactly; see :meth:`band_mass`).  Certified failures
        of such shots give a population lower bound through
        :func:`population_bracket`.
        """
        if fault_band is not None:
            syn, obs, errs, K = self._sample_band(shots, seed, fault_band,
                                                  return_errors)
        else:
            syn, obs, errs, K = self._sample_plain(shots, seed, return_errors)
        out = (syn, obs)
        if return_errors:
            out += (errs,)
        if return_fault_counts:
            out += (K,)
        return out

    def _sample_plain(self, shots, seed, return_errors):
        rng = np.random.default_rng(seed)
        n = self.num_mechanisms
        syn = np.empty((shots, self.num_detectors), dtype=np.uint8)
        obs = np.empty((shots, self.num_observables), dtype=np.uint8)
        K = np.empty(shots, dtype=np.int64)
        errs = np.empty((shots, n), dtype=np.uint8) if return_errors else None
        chunk = max(1, min(shots, 4_000_000 // max(n, 1)))
        for a in range(0, shots, chunk):
            b = min(shots, a + chunk)
            e = (rng.random((b - a, n)) < self.priors).astype(np.uint8)
            syn[a:b] = self.syndrome(e)
            obs[a:b] = self.observables(e)
            K[a:b] = e.sum(axis=1)
            if return_errors:
                errs[a:b] = e
        return syn, obs, errs, K

    def _sample_band(self, shots, seed, band, return_errors):
        from ._sampling import sample_fault_band
        lo, hi = (int(x) for x in band)
        # one 63-bit seed for the numba RNG, derived reproducibly from ``seed``
        s63 = int(np.random.default_rng(seed).integers(0, 2**63 - 1))
        ptr, idx, K = sample_fault_band(self.priors, lo, hi, shots, s63)
        E = _sp.csr_matrix((np.ones(idx.size, dtype=np.int64), idx, ptr),
                           shape=(shots, self.num_mechanisms))
        Hc, Lc = self._csr()
        syn = (np.asarray((E @ Hc.T).todense()) % 2).astype(np.uint8)
        obs = (np.asarray((E @ Lc.T).todense()) % 2).astype(np.uint8)
        errs = E.toarray().astype(np.uint8) if return_errors else None
        return syn, obs, errs, K
