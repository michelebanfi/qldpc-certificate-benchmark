"""Google Tesseract overlay: export a Sector to a Stim DEM and decode it.

Tesseract (an external MLE decoder built around Stim detector error models) is
optional (``pip install tesseract-decoder``, Python >= 3.10).  ``_HAS_TESSERACT`` guards its use so the rest of the
tool still runs without it.  A Sector's per-column noise model IS a detector
error model, so ``dem_from_sector`` assembles the DEM directly from the columns
(no Stim circuit); ``attach_tesseract_decoders`` compiles the beam decoder (and,
optionally, the bundled exact Simplex ILP) onto the sector.  It MUST run in
each worker process, because the decoders are C++/pybind objects.

This module deliberately does NOT import ``sector`` -- it duck-types the Sector
passed in -- so the optional ``stim`` import stays isolated here.
"""

import numpy as np

# Tesseract (Google QuantumAI) is an external MLE decoder built around Stim DEMs.
# Our per-column noise model IS a detector error model, so we can decode the same
# shots with it and overlay its LER on the certified-ML bracket.  Guarded so the
# rest of the tool still runs if the package (Python >= 3.10) is absent.
try:
    import stim
    from tesseract_decoder import tesseract as _tess, simplex as _simplex
    _HAS_TESSERACT = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_TESSERACT = False


def dem_from_sector(sector):
    """Export a Sector's per-column noise model to a stim DetectorErrorModel.

    Column j is one error mechanism: the detectors it flips are nonzero(Hdec[:,j]),
    the observables it flips are nonzero(H_log[:,j]), its probability is
    channel_probs[j].  Trivial-bucket columns (zero detector AND zero logical
    support -- the ones make_sector zeroes the weight of) are skipped: they flip
    nothing and their folded probability can exceed 0.5.  No Stim *circuit* is ever
    built -- the DEM is assembled directly from the columns.

    Returns (dem, kept_cols) where kept_cols[i] is the original sector-column index
    of the i-th DEM error, so decode_to_errors() indices map back to columns.
    """
    Hdec, H_log, probs = sector.Hdec, sector.H_log, sector.channel_probs
    dem = stim.DetectorErrorModel()
    kept = []
    for j in range(Hdec.shape[1]):
        dets = np.flatnonzero(Hdec[:, j])
        obs = np.flatnonzero(H_log[:, j])
        if dets.size == 0 and obs.size == 0:
            continue
        dem.append("error", float(probs[j]),
                   [stim.target_relative_detector_id(int(d)) for d in dets]
                   + [stim.target_logical_observable_id(int(o)) for o in obs])
        kept.append(j)
    # Advertise exactly m detectors and k observables so a full-length syndrome
    # vector aligns (stim otherwise sizes the DEM to the max index that appears,
    # and some detector rows are never set by any single-fault mechanism).
    m, k = Hdec.shape[0], H_log.shape[0]
    dem.append("detector", [], [stim.target_relative_detector_id(m - 1)])
    dem.append("logical_observable", [], [stim.target_logical_observable_id(k - 1)])
    return dem, np.asarray(kept, dtype=np.int64)


def _ensure_dem(sector):
    if sector.tess_dem is None:
        sector.tess_dem, sector.dem_kept_cols = dem_from_sector(sector)
    return sector.tess_dem


def attach_simplex_decoder(sector):
    """Build Tesseract's exact Simplex ILP decoder for this sector (per worker)."""
    sdec = _simplex.SimplexConfig(_ensure_dem(sector)).compile_decoder()
    sdec.init_ilp()  # required before SimplexDecoder.decode()
    sector.tess_simplex = sdec


def attach_tesseract_decoders(sector, beam=20, simplex_too=False,
                              beam_climbing=True, det_orders=0):
    """Build and attach the Tesseract beam decoder (and optionally the exact
    Simplex ILP) for one sector.  The decoders are C++/pybind objects, so this
    MUST run in each worker process (after fork), never in the parent.

    ``beam_climbing`` is the paper's key accuracy knob (arXiv:2503.10988).  When
    the fixed ``det_beam`` prunes away the real solution, the search widens
    instead of returning a low-confidence miss.  ``det_orders`` adds that many
    extra random detector orderings; 0 means a single natural order.  The
    orderings are seeded, so every worker builds the identical decoder.
    """
    dem = _ensure_dem(sector)
    kept = sector.dem_kept_cols
    cfg_kwargs = dict(det_beam=beam, beam_climbing=beam_climbing, merge_errors=False)
    if det_orders and det_orders > 0:
        m = sector.Hdec.shape[0]
        rng = np.random.default_rng(0)   # fixed seed: identical orderings per worker
        cfg_kwargs["det_orders"] = [rng.permutation(m).astype(np.int64).tolist()
                                    for _ in range(det_orders)]
    sector.tess_beam = _tess.TesseractConfig(dem, **cfg_kwargs).compile_decoder()
    if simplex_too:
        attach_simplex_decoder(sector)
    # Indexing check: the DEM must advertise exactly (m detectors, k observables),
    # and decoding the syndrome of a single known column must round-trip to that
    # same syndrome.
    assert sector.tess_beam.num_detectors == sector.Hdec.shape[0], \
        f"{sector.name}: DEM num_detectors != H rows"
    assert sector.tess_beam.num_observables == sector.H_log.shape[0], \
        f"{sector.name}: DEM num_observables != L rows"
    if kept.size:
        j0 = int(kept[0])
        s0 = (sector.Hdec[:, j0] % 2).astype(bool)
        errs0 = sector.tess_beam.decode_to_errors(s0)
        cols0 = sector.dem_kept_cols[errs0] if errs0 else np.empty(0, np.int64)
        induced0 = (sector.Hdec[:, cols0].sum(axis=1) % 2).astype(np.int8)
        assert np.array_equal(induced0, s0.astype(np.int8)), \
            f"{sector.name}: DEM detector indexing round-trip failed"
