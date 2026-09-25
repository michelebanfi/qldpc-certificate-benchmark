"""Decoders: the user-facing interface and the built-in decoders.

Any decoder takes a syndrome (uint8 array of length m).  It returns either

* a **correction**: a 0/1 vector over the n error mechanisms (columns of H).
  The predicted observables are ``L @ correction mod 2``.  A correction that
  does not reproduce the syndrome counts as a logical failure.
* the predicted **observables**: a 0/1 vector of length k (the usual output of
  Stim-style decoders such as PyMatching).

Wrap a plain function with :class:`FunctionDecoder`, or subclass
:class:`Decoder` and implement :meth:`Decoder.decode`.

Every decoder has a :class:`Role`, which fixes how its answer is used:

``PROBE``
    Measured against the certified bracket, and **never** used to build it.
    This is the decoder under test, so its benchmark is independent of the
    certificate.  Default for user decoders.
``SEED``
    Measured, and its (re-verified) corrections may tighten the bracket.  This
    speeds up certification, but that decoder's own comparison is then not
    independent.  Default for the built-in BP+OSD (``ldpc``) and Tesseract helpers.
``CROSSCHECK``
    Measured and checked for contradictions with the certificate.  Never seeds.
    Used for the exact Simplex MLE.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ._engine.sector import pack_class

__all__ = [
    "Role", "Decoder", "FunctionDecoder", "BPOSD", "Tesseract", "Simplex",
    "as_decoder", "has_tesseract",
]


def has_tesseract() -> bool:
    """True when the optional ``tesseract-decoder`` + ``stim`` stack imports."""
    from ._engine.tesseract import _HAS_TESSERACT
    return _HAS_TESSERACT


class Role(Enum):
    PROBE = "probe"
    SEED = "seed"
    CROSSCHECK = "crosscheck"


class DecodeResult:
    """One decoder's output on one shot (internal).

    ell:            packed predicted logical class, or None for a certain failure
                    (e.g. a correction violating the syndrome).
    correction:     verified 0/1 int64 correction usable as a seed, or None.
    ran:            whether the decoder ran on this shot.
    low_confidence: the decoder flagged its answer as unreliable.
    invalid:        the decoder returned a correction violating the syndrome.
    """
    __slots__ = ("ell", "correction", "ran", "low_confidence", "invalid")

    def __init__(self, ell=None, correction=None, ran=True, low_confidence=False,
                 invalid=False):
        self.ell = ell
        self.correction = correction
        self.ran = ran
        self.low_confidence = low_confidence
        self.invalid = invalid


def _result_from_correction(sector, s, corr):
    corr = np.asarray(corr).astype(np.int64).ravel() & 1
    if corr.size != sector.Hdec.shape[1]:
        raise ValueError(f"correction has length {corr.size}, expected "
                         f"{sector.Hdec.shape[1]} (number of columns of H)")
    if not np.array_equal(sector.Hdec.astype(np.int64) @ corr % 2,
                          np.asarray(s, dtype=np.int64)):
        return DecodeResult(ell=None, correction=None, invalid=True)
    return DecodeResult(ell=pack_class(sector.H_log @ corr % 2, sector.powers),
                        correction=corr)


def _result_from_observables(sector, pred):
    pred = np.asarray(pred).astype(np.int64).ravel() & 1
    k = len(sector.powers)
    if pred.size < k:     # a DEM may advertise fewer trailing observables
        pred = np.concatenate([pred, np.zeros(k - pred.size, np.int64)])
    if pred.size != k:
        raise ValueError(f"observable prediction has length {pred.size}, expected {k}")
    return DecodeResult(ell=pack_class(pred, sector.powers), correction=None)


class Decoder:
    """Base class for a decoder.

    Subclass it and implement :meth:`decode`.  Set ``output`` to
    ``"correction"`` or ``"observables"`` to declare what ``decode`` returns.
    ``"auto"`` infers it from the returned length.  Override :meth:`prepare`
    for setup that needs the problem, e.g. building a matching graph.
    ``prepare`` runs once in every worker process.
    """
    name = "decoder"
    role = Role.PROBE
    output = "auto"

    def __init__(self, name=None, role=None, output=None):
        if name is not None:
            self.name = name
        if role is not None:
            self.role = Role(role) if not isinstance(role, Role) else role
        if output is not None:
            self.output = output
        if self.output not in ("auto", "correction", "observables"):
            raise ValueError(f"output must be 'auto', 'correction' or 'observables', "
                             f"got {self.output!r}")

    def prepare(self, problem):
        """Per-worker setup hook.  Default: no-op."""

    def decode(self, syndrome):
        raise NotImplementedError

    # -- internal ----------------------------------------------------------
    def _setup(self, problem, sector):
        self.prepare(problem)

    def _run(self, sector, s, shot_idx):
        out = self.decode(np.asarray(s, dtype=np.uint8))
        out = np.asarray(out).ravel()
        n, k = sector.Hdec.shape[1], len(sector.powers)
        kind = self.output
        if kind == "auto":
            if out.size == n and n != k:
                kind = "correction"
            elif out.size == k and n != k:
                kind = "observables"
            else:
                raise ValueError(
                    f"decoder {self.name!r} returned length {out.size}; cannot infer "
                    f"whether it is a correction (n={n}) or observables (k={k}). "
                    f"Set output='correction' or output='observables'.")
        if kind == "correction":
            return _result_from_correction(sector, s, out)
        return _result_from_observables(sector, out)

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r}, role={self.role.name})"


class FunctionDecoder(Decoder):
    """Wrap ``fn(syndrome) -> correction | observables`` as a :class:`Decoder`.

    ``factory`` is an alternative to ``fn``: ``factory(problem)`` is called once
    per worker process and must return the decode function.  Use it for
    decoders that cannot be pickled or shared across processes.
    """

    def __init__(self, fn=None, name="decoder", role=Role.PROBE, output="auto",
                 factory=None):
        super().__init__(name=name, role=role, output=output)
        if (fn is None) == (factory is None):
            raise ValueError("pass exactly one of fn or factory")
        self._fn = fn
        self._factory = factory

    def prepare(self, problem):
        if self._factory is not None:
            self._fn = self._factory(problem)

    def decode(self, syndrome):
        return self._fn(syndrome)


# ----------------------------------------------------------------------------
# Built-in decoders
# ----------------------------------------------------------------------------

def _ldpc_channel(priors):
    """Priors for ldpc: columns that no decision depends on may carry p >= 0.5
    (they are ignored by the certificate), so clip into ldpc's valid range."""
    return np.clip(np.asarray(priors, dtype=np.float64), 1e-12, 0.49).tolist()


class BPOSD(Decoder):
    """BP+OSD from Joschka Roffe's ``ldpc`` package (``ldpc.BpOsdDecoder``).

    Keyword arguments are passed to ``ldpc.BpOsdDecoder``.  The parity-check
    matrix and the per-column priors (``error_channel``) come from the problem.
    Defaults: product-sum BP, 50 iterations, OSD-CS of order 7.  ``osd_order``
    defaults to 0 for ``osd_method="osd0"`` and to 7 otherwise.
    """
    name = "bp_osd"
    output = "correction"

    def __init__(self, max_iter=50, bp_method="product_sum", schedule="parallel",
                 osd_method="osd_cs", osd_order=None, name="bp_osd", role=Role.SEED,
                 **ldpc_kwargs):
        super().__init__(name=name, role=role)
        if osd_order is None:
            osd_order = 0 if str(osd_method).lower() in ("osd0", "osd_0") else 7
        self.kwargs = dict(max_iter=max_iter, bp_method=bp_method, schedule=schedule,
                           osd_method=osd_method, osd_order=osd_order, **ldpc_kwargs)
        self._dec = None

    def _setup(self, problem, sector):
        from ldpc import BpOsdDecoder
        from scipy.sparse import csr_matrix
        self._dec = BpOsdDecoder(csr_matrix(sector.Hdec.astype(np.uint8)),
                                 error_channel=_ldpc_channel(sector.channel_probs),
                                 **self.kwargs)

    def _run(self, sector, s, shot_idx):
        return _result_from_correction(
            sector, s, self._dec.decode(np.asarray(s, dtype=np.uint8)))

    def __getstate__(self):          # the C++ decoder is rebuilt in each worker
        d = self.__dict__.copy()
        d["_dec"] = None
        return d


class Tesseract(Decoder):
    """Google's Tesseract beam-search MLE decoder (``pip install tesseract-decoder``).

    ``beam_climbing`` is the accuracy knob (arXiv:2503.10988).  With climbing, a
    beam of about 20 matches the exact MLE on the codes we tested.  Without it,
    widening the beam floods the priority queue and yields low-confidence wrong
    answers.
    """
    name = "tesseract"
    output = "observables"

    def __init__(self, beam=20, beam_climbing=True, det_orders=0, name="tesseract",
                 role=Role.SEED):
        super().__init__(name=name, role=role)
        self.beam = beam
        self.beam_climbing = beam_climbing
        self.det_orders = det_orders

    def _setup(self, problem, sector):
        from ._engine.tesseract import attach_tesseract_decoders, _HAS_TESSERACT
        if not _HAS_TESSERACT:
            raise ImportError("Tesseract needs: pip install 'qldpc-certificate-benchmark[tesseract]'")
        if sector.tess_beam is None:
            attach_tesseract_decoders(sector, beam=self.beam, simplex_too=False,
                                      beam_climbing=self.beam_climbing,
                                      det_orders=self.det_orders)

    def _run(self, sector, s, shot_idx):
        sb = np.asarray(s).astype(bool)
        res = _result_from_observables(sector, sector.tess_beam.decode(sb))
        res.low_confidence = bool(sector.tess_beam.low_confidence_flag)
        # Seed only from a confident answer that genuinely reproduces the syndrome.
        if not res.low_confidence:
            try:
                errs = sector.tess_beam.decode_to_errors(sb)
            except Exception:
                errs = None
            if errs:
                x = np.zeros(sector.H_aug.shape[1], dtype=np.int64)
                x[sector.dem_kept_cols[np.asarray(errs, dtype=np.int64)]] = 1
                if np.array_equal(sector.Hdec.astype(np.int64) @ x % 2,
                                  np.asarray(s, dtype=np.int64)):
                    res.correction = x
        return res


class Simplex(Decoder):
    """Exact MLE via Tesseract's bundled Simplex ILP.  Slow; runs on the first
    ``shot_cap`` shots only.  Never seeds (``correction`` is always None), so it
    stays an independent cross-check of the certificate."""
    name = "simplex"
    output = "observables"

    def __init__(self, shot_cap=100, name="simplex", role=Role.CROSSCHECK):
        super().__init__(name=name, role=role)
        self.shot_cap = shot_cap

    def _setup(self, problem, sector):
        from ._engine.tesseract import attach_simplex_decoder, _HAS_TESSERACT
        if not _HAS_TESSERACT:
            raise ImportError("Simplex needs: pip install 'qldpc-certificate-benchmark[tesseract]'")
        if sector.tess_simplex is None:
            attach_simplex_decoder(sector)

    def _run(self, sector, s, shot_idx):
        if shot_idx is None or shot_idx >= self.shot_cap:
            return DecodeResult(ran=False)
        res = _result_from_observables(
            sector, sector.tess_simplex.decode(np.asarray(s).astype(bool)))
        res.correction = None
        return res


_BUILTINS = {"bp_osd": BPOSD, "bposd": BPOSD, "tesseract": Tesseract,
             "simplex": Simplex}


def as_decoder(obj, name=None):
    """Coerce a Decoder, a built-in name, or a callable into a Decoder."""
    if isinstance(obj, Decoder):
        if name is not None:
            obj.name = name
        return obj
    if isinstance(obj, str):
        key = obj.lower()
        if key not in _BUILTINS:
            raise ValueError(f"unknown built-in decoder {obj!r}; known: "
                             f"{sorted(set(_BUILTINS))}")
        d = _BUILTINS[key]()
        if name is not None:
            d.name = name
        return d
    if callable(obj):
        return FunctionDecoder(obj, name=name or getattr(obj, "__name__", "decoder"))
    raise TypeError(f"cannot interpret {obj!r} as a decoder")
