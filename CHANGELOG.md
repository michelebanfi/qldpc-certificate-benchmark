# Changelog

## Unreleased

**Joint X/Z decoding**
- `CSSPartition`: which detectors and observables of a joint model belong to
  the X-error and to the Z-error sector (`DecodingProblem(..., css=...)`).
- `DecodingProblem.css_split()`: the two marginal (separate-decoding) models of
  a joint one, exact: identical X parts merge with the probability that an odd
  number of them fires, so each sector model has the joint model's law of its
  syndrome and observables.
- Joint model builders: `from_css_code_pauli(Hx, Hz, px, py, pz)` and
  `from_css_code_depolarizing(Hx, Hz, p)` (code capacity; independent X, Y, Z
  mechanisms solved so the Pauli channel is reproduced exactly, so the
  marginals are exactly `from_css_code(..., px+py)` / `(..., pz+py)`), and
  `from_css_sectors(x_sector, z_sector, mechanisms)` for any CSS circuit given
  the two sector models and its faults as `(x_col, z_col, p)`.
- `compare_joint_separate(problem, shots, ...)` -> `JointComparison`: certifies
  joint ML and separate ML (each sector's marginal model on its rows of the
  same shot; a shot fails iff either sector fails) on identical shots, with
  both certified brackets, one-sided paired tests in both directions, and
  decoders run jointly and/or separately (`decoder_vs_ml`).
- `combine_sector_verdicts`: the separate-decoding verdict of a shot.

**Speed**
- The A* kernels index arrays with unsigned subscripts, so Numba no longer
  emits a negative-index wraparound check on every access: ~1.45x faster
  searches, bit-identical results (slightly faster than a line-by-line C port).
- The covering-LP dual of every shot is handed to HiGHS directly instead of
  through `scipy.optimize.linprog`, whose wrapper was 50-70% of the solve:
  1.7-3x faster, bit-identical solutions.  It uses scipy's vendored highspy
  (scipy >= 1.15), self-checked against `linprog` once per process, and falls
  back to `linprog` otherwise.
- Pool workers started with `forkserver` or `spawn` (always the case on
  Linux, where BLAS threads make the parent multi-threaded) now start in
  parallel instead of one after another: 8 workers on Linux 16.4 s -> 6.3 s
  for an 800-shot run.  Each shot travels with its task, so workers no longer
  hold a copy of every syndrome.

**Seeders**
- **Behavior change:** seeders are off by default (`seeders=None` in
  `benchmark`, `certify_shots` and `compare_joint_separate`); declare them
  with `seeders="auto"` (BP+OSD, plus Tesseract when installed) or a list.
- A seeder runs only on the shots the cheap first pass leaves open, and it is
  never scored: seeders no longer appear among the result's decoders, and
  `result.meta["seeders"]` records them.  ML verdicts, bounds and routes are
  unchanged for the same seeders; on circuit-level runs that used the old
  `"auto"` default, the whole run is 3-4x faster.
- To benchmark BP+OSD or Tesseract, pass it in `decoders=`: `BPOSD` and
  `Tesseract` now default to role `probe`, and a decoder with role `seed` in
  `decoders=` is rejected.  `certify_shots(seed_corrections=...)` corrections
  are no longer scored either (pass them in `predictions` too to score them).

## 0.2.0

**Soundness**
- Every certificate MIP now runs HiGHS at an exact gap (`mip_rel_gap=0`) and
  takes its lower bound from `mip_dual_bound`. At the default 1e-4 relative
  gap, HiGHS reports status 0 when the incumbent may still be ~0.01 nats above
  the minimum, which is within the near-tie regime. An audit of 1268 archived
  ILP-based verdicts found none wrong, but 23 had rested on that claim.

**Speed**
- New cheap first pass (`CertConfig.use_fast_pass`, on by default): a unified
  single-coset A* with no true-class cap, bounded by the covering-LP dual. It
  is exact and needs no anchor or seeds. On [[144,12,12]] circuit noise it
  takes 2–4 ms per sector, against 80–226 ms for the decomposition, with
  identical verdicts on 6000 sectors. Shots it leaves open go to the full
  certificate as before.
- Last resorts for shots that stay undecided:
  `fast_retry_node_budget` (a large-budget cap-free search) and
  `use_global_route` (a global ILP witness plus an infeasibility proof against
  every competing class; no MIP optimality claim is trusted).
- `Certificate.route` and `BenchmarkResult.route_counts()` report which stage
  decided each shot.

**Statistics**
- Rare-event sampling: `benchmark(..., fault_band=(lo, hi))` samples exactly
  from the priors conditioned on the fault count `K`.
  `DecodingProblem.band_mass`, `.fault_count_pmf` and
  `sample(fault_band=..., return_fault_counts=True)` are the building blocks.
- `population_bracket(results)` pools plain and band runs into a joint
  Clopper-Pearson bracket on the population ML LER:
  `Σ P(S)·CP_lower(F_S, n_S)` below and `CP_upper` over the plain shots above.
- `clopper_pearson_lower` / `clopper_pearson_upper`: one-sided exact limits.
- **Behavior change:** `ml_bracket(alpha)` now splits `alpha` across its two
  sides, so `[lb_ci, ub_ci]` is a *joint* `1-alpha` bracket. In 0.1.0 each side
  was one-sided at `1-alpha`, a joint coverage of only `1-2·alpha`, although
  the report called it "95%".
- Results record per-shot fault counts and routes (`fault_counts`, `route` in
  the `.npz`) and a problem fingerprint.
