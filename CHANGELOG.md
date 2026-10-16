# Changelog

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
- The A* kernels index arrays with unsigned subscripts, so Numba no longer
  emits a negative-index wraparound check on every access: ~1.45x faster
  searches, bit-identical results (slightly faster than a line-by-line C port).

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
