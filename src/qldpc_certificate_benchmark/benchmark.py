"""Benchmark drivers: sample -> decode -> certify, in parallel.

:func:`benchmark` samples shots from a :class:`DecodingProblem`, runs your
decoders plus the bracket seeders, and certifies every shot.
:func:`certify_shots` does the same for shots and predictions you computed
yourself, for example with Stim's samplers and a batch decoder.

Both return a :class:`BenchmarkResult`.  Its rows are in **shot order**.  With
``target_failures`` the run stops early, and the result holds a prefix of the
shots.
"""

from __future__ import annotations

import copy
import os
import sys
import time
import warnings
from multiprocessing import cpu_count, get_context

import numpy as np

from .config import CertConfig, DEFAULT_CFG
from .certifier import Certifier
from .decoders import (Decoder, DecodeResult, Role, BPOSD, Tesseract, Simplex,
                       as_decoder, has_tesseract,
                       _result_from_correction, _result_from_observables)
from .results import BenchmarkResult

__all__ = ["benchmark", "certify_shots"]


class _Precomputed(Decoder):
    """Replays a precomputed (shots, k) observable or (shots, n) correction array."""

    def __init__(self, name, arr, kind, role):
        super().__init__(name=name, role=role)
        self._arr = arr
        self._kind = kind

    def _run(self, sector, s, shot_idx):
        if shot_idx is None:
            return DecodeResult(ran=False)
        row = self._arr[shot_idx]
        if self._kind == "correction":
            return _result_from_correction(sector, s, row)
        return _result_from_observables(sector, row)


def _resolve_seeders(seeders):
    if seeders is None or seeders is False:
        return []
    if isinstance(seeders, str) and seeders == "auto":
        out = [BPOSD(role=Role.SEED)]
        if has_tesseract():
            out.append(Tesseract(role=Role.SEED))
        return out
    out = []
    for d in seeders:
        d = as_decoder(d)
        d.role = Role.SEED
        out.append(d)
    return out


def _build_decoder_list(decoders, seeders, crosscheck):
    if decoders is None:
        decoders = []
    if isinstance(decoders, dict):
        decs = [as_decoder(v, name=k) for k, v in decoders.items()]
    elif isinstance(decoders, (Decoder, str)) or callable(decoders):
        decs = [as_decoder(decoders)]
    else:
        decs = [as_decoder(d) for d in decoders]
    decs += _resolve_seeders(seeders)
    if crosscheck:
        cap = crosscheck if isinstance(crosscheck, int) and crosscheck is not True else 100
        decs.append(Simplex(shot_cap=cap))
    names = [d.name for d in decs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate decoder names {sorted(dup)}: give each decoder a "
                         f"unique name (e.g. pass a dict {{name: decoder}})")
    return decs


# ------------------------------------------------------------ process start
def _native_thread_count():
    """Number of OS threads in this process, or None if it cannot be determined.

    Forking a multi-threaded process is unsafe.  In particular HiGHS (scipy's
    LP/MILP solver) keeps a persistent thread pool after its first call, and a
    forked child that calls HiGHS again can deadlock.  We therefore only fork
    from a single-threaded parent."""
    try:
        if sys.platform.startswith("linux"):
            return len(os.listdir("/proc/self/task"))
        if sys.platform == "darwin":
            import ctypes
            import ctypes.util
            lib = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib")
            buf = ctypes.create_string_buffer(96)          # struct proc_taskinfo
            n = lib.proc_pidinfo(os.getpid(), 4, ctypes.c_uint64(0), buf, 96)
            if n == 96:   # 6 x uint64, then int32 fields; pti_threadnum is the 10th
                return int(np.frombuffer(buf.raw, dtype=np.int32, offset=48)[9])
    except Exception:
        pass
    return None


def _choose_context(requested, payload):
    if requested is not None:
        return requested
    if sys.platform == "win32":
        return "spawn"
    if _native_thread_count() == 1:
        return "fork"
    try:
        import pickle
        pickle.dumps(payload)
    except Exception:
        warnings.warn(
            "this process is already multi-threaded (e.g. scipy's HiGHS solver ran "
            "here before), so forking worker processes may deadlock, but the decoders "
            "cannot be pickled for a clean 'forkserver' start.  Falling back to "
            "'fork'.  To avoid the risk, define decoders at module level (or use "
            "FunctionDecoder(factory=...) with a module-level factory), or run "
            "benchmark() before any in-process certification.", RuntimeWarning,
            stacklevel=4)
        return "fork"
    return "forkserver"


# --------------------------------------------------------------------- workers
_W = {}


def _pin_threads():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


def _init_worker(problem, decoders, cfg, syn, obs, warm=True):
    if not warm:                        # in-process: let errors propagate directly
        return _init_worker_body(problem, decoders, cfg, syn, obs, warm)
    # A Pool silently respawns a worker whose initializer raises, forever.
    # Record the error instead and re-raise it from the first task, so the
    # parent receives the real traceback.
    try:
        _init_worker_body(problem, decoders, cfg, syn, obs, warm)
    except BaseException:
        import traceback
        _W["init_error"] = traceback.format_exc()


def _init_worker_body(problem, decoders, cfg, syn, obs, warm):
    if warm:
        _pin_threads()
    cert = Certifier(problem, cfg)
    decs = [copy.copy(d) for d in decoders]
    for d in decs:
        d._setup(problem, cert.sector)
    _W.update(cert=cert, decs=decs, syn=syn, obs=obs)
    if warm and len(syn):
        # compile every kernel before the first timed shot (never in the parent:
        # the parent must stay single-threaded so that forking stays safe)
        i = int(np.argmax(syn.any(axis=1)))
        cert._decode_and_certify(np.ascontiguousarray(syn[i], dtype=np.int8),
                                 np.asarray(obs[i], dtype=np.int64) & 1, decs, None)


def _do_shot(i):
    w = _W
    if "init_error" in w:
        raise RuntimeError("worker initialisation failed:\n" + w["init_error"])
    s = np.ascontiguousarray(w["syn"][i], dtype=np.int8)
    lam = np.asarray(w["obs"][i], dtype=np.int64) & 1
    t0 = time.perf_counter()
    row = w["cert"]._decode_and_certify(s, lam, w["decs"], i)
    row["time"] = time.perf_counter() - t0
    return i, row


def _run(problem, decs, cfg, syn, obs, num_workers, target_failures, stop_on,
         progress, mp_context, meta):
    shots = len(syn)
    if num_workers is None:
        num_workers = cpu_count()
    num_workers = max(1, min(int(num_workers), shots or 1))
    if target_failures is not None and stop_on is None:
        probes = [d.name for d in decs if d.role is Role.PROBE]
        stop_on = probes[0] if probes else "ml"
    rows = [None] * shots
    done = 0
    fails = 0
    t0 = time.time()

    def _tick(i, row):
        nonlocal done, fails
        rows[i] = row
        done += 1
        if target_failures is not None:
            if stop_on == "ml":
                fails += int(row["verdict"] == 1)
            else:
                fails += int(row[f"ran:{stop_on}"] and not row[f"ok:{stop_on}"])
        if progress and (done % max(1, shots // 50) == 0 or done == shots):
            el = time.time() - t0
            sys.stderr.write(f"\r  certified {done}/{shots} shots  "
                             f"({el:.0f}s, {done / max(el, 1e-9):.1f} shots/s)")
            sys.stderr.flush()
        return target_failures is not None and fails >= target_failures

    if num_workers == 1:
        _init_worker(problem, decs, cfg, syn, obs, warm=False)
        for i in range(shots):
            if _tick(*_do_shot(i)):
                break
    else:
        ctx = get_context(_choose_context(mp_context, (problem, decs, cfg)))
        # imap (ordered) keeps results in shot order, so an early stop on
        # target_failures leaves a contiguous prefix of the shots.
        with ctx.Pool(num_workers, initializer=_init_worker,
                      initargs=(problem, decs, cfg, syn, obs)) as pool:
            for i, row in pool.imap(_do_shot, range(shots), chunksize=1):
                if _tick(i, row):
                    pool.terminate()
                    break
    if progress:
        sys.stderr.write("\n")
    n_done = next((j for j, r in enumerate(rows) if r is None), shots)
    return BenchmarkResult.from_rows(
        rows[:n_done], decoders=[(d.name, d.role.value) for d in decs],
        problem=problem, config=cfg, elapsed=time.time() - t0, meta=meta)


# ------------------------------------------------------------------ public API
def benchmark(problem, decoders=None, shots: int = 1000, *, seed=0,
              seeders="auto", crosscheck=False, config: CertConfig = DEFAULT_CFG,
              num_workers=None, target_failures=None, stop_on=None,
              progress: bool = True, mp_context=None) -> BenchmarkResult:
    """Sample shots from ``problem``, decode them, and certify ML on every shot.

    Parameters
    ----------
    problem : DecodingProblem
    decoders : decoder, list of decoders, or dict ``{name: decoder}``
        Each entry is a :class:`Decoder`, a callable ``syndrome -> correction |
        observables``, or a built-in name (``"bp_osd"``, ``"tesseract"``).  User decoders default to role ``PROBE``, so they never
        influence the certificate.
    shots : int
        Number of shots to sample.
    seed : int
        RNG seed for sampling.  The run is reproducible for any ``num_workers``.
    seeders : "auto", list, or None
        Helper decoders that seed (tighten) the bracket.  ``"auto"`` means
        BP+OSD, plus Tesseract when it is installed.  Seeders are also
        measured, but flagged as not independent.  ``None`` disables seeding.
    crosscheck : bool or int
        Add the exact Simplex MLE (needs ``tesseract-decoder``) on the first 100
        shots, or on the first ``crosscheck`` shots when an int is given.  It
        should never contradict a proven verdict.
    config : CertConfig
        Engine budgets and toggles.
    num_workers : int, optional
        Worker processes.  Defaults to all cores.  ``1`` runs in-process.
    target_failures : int, optional
        Stop once ``stop_on`` has failed this many times (default: the first
        PROBE decoder, or ``"ml"`` for certified ML failures).
    mp_context : str, optional
        Multiprocessing start method.  By default ``"fork"`` is used when this
        process is single-threaded, and ``"forkserver"`` otherwise (forking a
        multi-threaded process can deadlock the LP solver).
    """
    syn, obs = problem.sample(shots, seed=seed)
    decs = _build_decoder_list(decoders, seeders, crosscheck)
    meta = {"mode": "benchmark", "seed": seed, "shots_requested": shots}
    return _run(problem, decs, config, syn, obs, num_workers, target_failures,
                stop_on, progress, mp_context, meta)


def certify_shots(problem, syndromes, observables, predictions=None, *,
                  decoders=None, seed_corrections=None, seeders="auto",
                  crosscheck=False, config: CertConfig = DEFAULT_CFG,
                  num_workers=None, progress: bool = True,
                  mp_context=None) -> BenchmarkResult:
    """Certify ML on shots you already have, and score precomputed predictions.

    Parameters
    ----------
    syndromes : (shots, m) 0/1 array (detection events)
    observables : (shots, k) 0/1 array (the TRUE observable flips)
    predictions : dict ``{name: array}``, optional
        Per-decoder predictions: ``(shots, k)`` observables, or ``(shots, n)``
        corrections over the columns of ``H``.  These are scored as ``PROBE``s.
    decoders : optional
        Live decoders to also run on these shots (same forms as :func:`benchmark`).
    seed_corrections : dict ``{name: (shots, n) array}``, optional
        Corrections that may seed the bracket (role ``SEED``).
    """
    syn = np.ascontiguousarray(np.asarray(syndromes, dtype=np.uint8))
    obs = np.ascontiguousarray(np.asarray(observables, dtype=np.uint8))
    if syn.ndim != 2 or syn.shape[1] != problem.num_detectors:
        raise ValueError(f"syndromes must have shape (shots, {problem.num_detectors})")
    if obs.ndim == 1:
        obs = obs[:, None]
    if obs.shape != (syn.shape[0], problem.num_observables):
        raise ValueError(f"observables must have shape ({syn.shape[0]}, "
                         f"{problem.num_observables})")
    n, k = problem.num_mechanisms, problem.num_observables
    pre = []

    def _kind(name, arr):
        arr = np.asarray(arr, dtype=np.uint8)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] != syn.shape[0]:
            raise ValueError(f"{name}: expected {syn.shape[0]} rows, got {arr.shape[0]}")
        if arr.shape[1] == k and k != n:
            return arr, "observables"
        if arr.shape[1] == n:
            return arr, "correction"
        raise ValueError(f"{name}: width {arr.shape[1]} is neither k={k} "
                         f"(observables) nor n={n} (corrections)")

    for name, arr in (predictions or {}).items():
        a, kind = _kind(name, arr)
        pre.append(_Precomputed(name, a, kind, Role.PROBE))
    for name, arr in (seed_corrections or {}).items():
        a, kind = _kind(name, arr)
        if kind != "correction":
            raise ValueError(f"seed_corrections[{name!r}] must be (shots, n) corrections")
        pre.append(_Precomputed(name, a, kind, Role.SEED))
    decs = pre + _build_decoder_list(decoders, seeders, crosscheck)
    names = [d.name for d in decs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate decoder names: {names}")
    meta = {"mode": "certify_shots", "shots_requested": int(syn.shape[0])}
    return _run(problem, decs, config, syn, obs, num_workers, None, None,
                progress, mp_context, meta)
