# qldpc-certificate-benchmark

**Certified maximum-likelihood benchmarking for quantum error-correction decoders.**

Decoder papers usually compare against other heuristic decoders, because the
true maximum-likelihood (ML) decoder is intractable to run. This library
avoids that problem. For every shot it **proves** whether ML decoding would
succeed or fail. Averaged over shots, this gives a *certified bracket* on the
ML logical error rate, and any decoder can then be measured against the
optimum instead of against another heuristic.

It works on any independent-error decoding problem:

* a **Stim detector error model** or circuit (circuit-level noise),
* a **parity-check matrix** with its logical operators (code-capacity or
  phenomenological noise),
* any `(H, L, priors)` triple you build yourself.

Here is an example: a rotated surface code, d=5, 5 rounds, circuit-level
p=0.005, 2000 Stim shots, PyMatching as the decoder under test
([`examples/stim_surface_code.py`](examples/stim_surface_code.py)):

```
Certified ML benchmark -- rotated surface d=5, p=0.005 (m=120, n=1679, k=1)
  shots: 2000   wall: 9.3s   verdicts: 1992 success / 8 fail / 0 tie / 0 undecided
  ML LER certified bracket: [4.000e-03, 4.000e-03]   (95% CI-widened: [1.992e-03, 7.206e-03])

  decoder       role              LER  95% CI                  recov. intrin. unkn.  p(worse than ML)
  pymatching    probe       6.500e-03  [3.47e-03, 1.11e-02]        8       5     0             0.113
  bp_osd        seed        4.500e-03  [2.06e-03, 8.53e-03]        2       7     0               0.5
  tesseract     seed        4.000e-03  [1.73e-03, 7.87e-03]        0       8     0                 1
```

On these shots, ML provably fails on exactly 8 (the bracket is closed).
PyMatching fails on 13: 5 of them no decoder could have corrected, and 8 are
shots where ML provably succeeds.

---

## Installation

```bash
pip install qldpc-certificate-benchmark            # core: numpy, scipy, numba, ldpc
pip install "qldpc-certificate-benchmark[stim]"    # + Stim DEM / circuit import
pip install "qldpc-certificate-benchmark[all]"     # + Tesseract beam MLE & exact Simplex MLE
```

Python ≥ 3.10. The first call JIT-compiles the Numba kernels, which takes
about 20 s. The compiled kernels are cached for later runs.

## Quick start

### 1. Benchmark your decoder on a code (code-capacity noise)

```python
import qldpc_certificate_benchmark as qcb

Hx, Hz, Lx, Lz = qcb.codes.bb_72_12_6()                   # [[72,12,6]] bivariate-bicycle code
problem = qcb.DecodingProblem.from_css_code(Hx, Hz, p=0.04)  # X errors; logicals computed for you

def my_decoder(syndrome):          # any function: syndrome -> correction (length n)
    ...                            #            or syndrome -> observables (length k)
    return correction

result = qcb.benchmark(problem, {"mine": my_decoder}, shots=2000)
print(result)
```

### 2. Circuit-level noise from Stim

```python
import stim, pymatching
import qldpc_certificate_benchmark as qcb

circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5,
                                 after_clifford_depolarization=0.005,
                                 before_measure_flip_probability=0.005,
                                 after_reset_flip_probability=0.005)
problem = qcb.DecodingProblem.from_stim_circuit(circuit)
# or: qcb.DecodingProblem.from_dem(circuit.detector_error_model())
# or: qcb.DecodingProblem.from_dem("path/to/model.dem")

def make_matcher(problem):          # called once in each worker process
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem).decode

result = qcb.benchmark(problem,
                       {"pymatching": qcb.FunctionDecoder(factory=make_matcher,
                                                          output="observables")},
                       shots=5000)
```

### 3. Certify predictions you already computed

If you already have a sampling and batch-decoding pipeline, hand over the
shots and predictions. Nothing is re-sampled.

```python
dets, obs = circuit.compile_detector_sampler(seed=0).sample(10_000, separate_observables=True)
pred = matcher.decode_batch(dets)                     # (shots, k) observables

result = qcb.certify_shots(problem, dets, obs, predictions={"pymatching": pred})
print(result.ml_bracket(), result.ler("pymatching"), result.gap_to_ml("pymatching"))
```

`predictions` may hold `(shots, k)` observable predictions or `(shots, n)`
corrections over the columns of `H`.

### 4. Certify a single shot

```python
cert = qcb.Certifier(problem)
c = cert.certify(syndrome, true_observables)
c.verdict        # "success" | "fail" | "tie" | "undecided"
c.ml_correct     # True / False / None
```

### Your own matrices

```python
problem = qcb.DecodingProblem(H, L, priors)   # H: (m,n), L: (k,n), priors: scalar or (n,)
```

`H` and `L` may be dense arrays or `scipy.sparse` matrices. Column `j` is one
independent error mechanism that fires with probability `priors[j]`, flips
the detectors in `H[:, j]`, and flips the logical observables in `L[:, j]`.

### Where the logical operators come from

The certificate compares *logical classes*: the class of an error `e` is
`L e mod 2`. So it needs `L`, and where `L` comes from depends on the input:

* **Stim DEM or circuit.** `L` is the DEM's `L0, L1, ...` observables, as
  defined by the circuit's `OBSERVABLE_INCLUDE`. Nothing is computed; the
  certificate answers exactly the question your Stim benchmark asks.
* **CSS code, no logicals at hand.** `DecodingProblem.from_css_code(Hx, Hz, p,
  error_type="X")` computes them with GF(2) elimination. A basis of `ker Hx`
  modulo `rowspace Hz` gives the Z logicals, which classify X errors. Both
  matrices are needed: from `Hz` alone you cannot tell a stabilizer
  (in `rowspace Hx`) from a logical operator. `qcb.codes.css_logicals(Hx, Hz)`
  returns the logicals directly.
* **Your own `L`.** Used as given. Any valid basis gives identical verdicts,
  because another basis, or logicals shifted by stabilizers, only relabels the
  classes. If you pass only a subset of the logicals, the certificate
  answers "does ML predict *these* observables correctly", which is the same
  convention as Stim.

---

## What exactly is certified

For a shot with syndrome `s` and true observables `λ`, write the cost of an
error pattern `e` as `Σ_j e_j·log((1-p_j)/p_j)` (its negative log-likelihood,
up to a constant). The certifier decides, **with proof**, how the cheapest
solution of `H e = s` in the true class `L e = λ` compares with the cheapest
solution in every wrong class:

| verdict     | meaning                                                                  |
|-------------|--------------------------------------------------------------------------|
| `success`   | every wrong-class solution is strictly more expensive, so ML is correct  |
| `fail`      | some wrong-class solution is strictly cheaper, so ML is wrong            |
| `tie`       | the two optima are equal (within 1e-9), so ML depends on tie-breaking    |
| `undecided` | the search budget ran out before a proof was found                       |

Over `N` shots, `#fail/N ≤ LER_ML ≤ (#fail + #tie + #undecided)/N` holds
**exactly** on the sampled shots. `result.ml_bracket()` also gives a bracket
on the *population* ML LER, `[lb_ci, ub_ci]`, which holds jointly with 95%
confidence: exact one-sided Clopper-Pearson limits at 2.5% each. For rates too
small to see with plain sampling, see [Rare events](#rare-events-fault-count-bands).

This is *non-degenerate* ML: the most likely single error pattern, which is
the target of minimum-weight / MLE decoders such as Tesseract and exact ILP
decoders. It is not the most likely logical coset. A decoder can
occasionally beat non-degenerate ML; those shots are reported as `lucky`.
Code-capacity models with *uniform* priors have many exactly-degenerate
minima and therefore many `tie`s. Circuit-level DEMs, whose priors are not
uniform, rarely produce them.

**How it is proven.** Each shot first tries a cheap pass: one unified A*
search over the whole syndrome coset with no true-class cap. The first goal it
pops is the exact global minimum, and it stops once every unseen solution is
provably above it. This settles most shots in a few milliseconds. The shots it
leaves open go to the full certificate: an exact A* search over each coset with
admissible heuristics, strengthened by a covering-LP dual bound, adaptive
parity-cut LPs (Feldman), and exact integer programs (HiGHS). A certified
cluster decomposition splits shots with independent error regions. Every
witness produced by an LP/ILP solver or a seeding decoder is re-verified in
NumPy before it can enter a proof, so a solver bug or a wrong decoder can
make a shot `undecided` but never flip a verdict. Every MIP runs at an exact
gap (`mip_rel_gap=0`), and its lower bound is read from HiGHS's dual bound. The
engine is validated
against exhaustive enumeration by `qcb.selftest()`. The kernels are compiled
without `fastmath`, so the strict comparisons keep IEEE semantics.

## Reading the results

```python
result.ml_bracket()        # {'lb', 'ub', 'lb_ci', 'ub_ci', 'confidence', ...}
                           # lb_ci/ub_ci: joint 95% population bracket
result.ler("mine")         # LER, failures, shots, Clopper-Pearson CI
result.decomposition("mine")
#   intrinsic   -- ML also provably fails: no decoder could fix these
#   recoverable -- ML provably succeeds: the decoder is provably suboptimal here
#   unknown     -- ML verdict is tie/undecided
#   lucky       -- decoder right where non-degenerate ML provably fails
result.gap_to_ml("mine")   # one-sided exact McNemar vs ML; uncertified shots are
                           # resolved AGAINST the claim, so p is certified
result.compare("mine", "bp_osd")  # paired exact McNemar between two decoders
result.summary()           # everything as a dict
result.route_counts()      # which certifier stage decided each shot
result.save("run1")        # run1.json (summary) + run1.npz (per-shot arrays)
```

`result.verdict`, `result.ok[name]`, and `result.ran[name]` hold the per-shot
arrays **in shot order**.

## Decoders and roles (why the benchmark stays independent)

Every decoder has a role:

| role         | measured | may tighten the certificate | default for                           |
|--------------|:--------:|:---------------------------:|---------------------------------------|
| `PROBE`      | ✓        | ✗                           | your decoders                         |
| `SEED`       | ✓        | ✓ (re-verified)             | `BPOSD`, `Tesseract` (via `seeders`)  |
| `CROSSCHECK` | ✓        | ✗                           | `Simplex` (exact MLE)                 |

The bracket is built only from `SEED` decoders plus an internal
decoder-agnostic anchor. A `PROBE`'s answer structurally cannot influence its
own certificate, so its comparison with ML is independent. Seeders only make
certification faster, by lowering the `undecided` fraction. By default
(`seeders="auto"`) BP+OSD seeds, plus Tesseract when it is installed. Seeding
with Tesseract is strongly recommended on large circuit-level DEMs. Pass
`seeders=None` to disable seeding.

Built-in decoders:

* `qcb.BPOSD(max_iter=50, osd_method="osd_cs", osd_order=7, **kw)`: BP+OSD from
  Joschka Roffe's [`ldpc`](https://github.com/quantumgizmos/ldpc) package
  (`ldpc.BpOsdDecoder`). Extra keyword arguments go straight to `ldpc`. Pass
  `role="probe"` to benchmark it rather than seed with it.
* `qcb.Tesseract(beam=20, beam_climbing=True)`: Google's beam-search MLE
  (`[all]` extra). Keep `beam_climbing=True`: without it, a wider beam gives
  worse answers.
* `qcb.Simplex(shot_cap=100)`: exact MLE ILP. Pass `crosscheck=True` to
  `benchmark` to verify that it never contradicts a proven verdict.

Custom decoders: pass a function, or subclass `qcb.Decoder`:

```python
class MyDecoder(qcb.Decoder):
    name, output = "mine", "correction"      # or "observables"
    def prepare(self, problem):              # once per worker process
        self.H = problem.H
    def decode(self, syndrome):
        ...
```

A correction that does not reproduce the syndrome counts as a logical failure
(reported as `invalid`).

## Rare events: fault-count bands

At low `p` the ML failure rate can be far below `1/shots`, and a plain run
certifies no failures at all. ML failures need many faults, so they concentrate
at fault counts `K` (the number of fired mechanisms) well above the mean. A
**fault-count band** samples only shots with `lo ≤ K ≤ hi`, drawn exactly from
the prior law conditioned on that event:

```python
problem.band_mass(12, 60)          # exact P(12 <= K <= 60) (Poisson-binomial)
plain = qcb.benchmark(problem, decoders, shots=100_000, seed=1)
band  = qcb.benchmark(problem, decoders, shots=20_000, seed=2, fault_band=(12, 60))
qcb.population_bracket([plain, band])
# {'lower': ..., 'upper': ..., 'joint_confidence': 0.95, 'band_rows': [...], ...}
```

Given `K` in the band, every shot fails ML with the same probability, so the
number of certified failures `F` among `n` band shots is exactly binomial:

* **lower bound:** `Σ_bands P(S)·CP_lower(F_S, n_S; α_lower/#bands)`. Each band
  pools its own runs with the plain shots whose `K` falls inside it.
* **upper bound:** `CP_upper(N − #success, N; α_upper)` over the plain shots. A
  tie or undecided shot counts as a failure here.

Coverage is at least `1 − α_lower − α_upper` (95% by default). A missed proof
only makes a bound more conservative. The Clopper-Pearson limits assume
fixed-size runs, so leave `target_failures` unset for runs you pool.
`population_bracket` rejects results from different problems, repeated seeds,
and overlapping bands. `qcb.clopper_pearson_lower/upper` expose the one-sided
limits directly.

## Joint X/Z decoding

For a CSS code, decoders usually treat the X-error and Z-error sectors as two
independent problems, each with the *marginal* noise of its sector. A `Y`, or a
correlated two-qubit fault after a CNOT, is then priced as two independent
faults, although it is one event. A **joint** model keeps such a fault as one
mechanism that flips detectors of both sectors. The certificate runs on it
unchanged (it is just a problem with `2k` observables), so you can certify
joint ML and separate ML **on the same shots** and compare them:

```python
Hx, Hz, Lx, Lz = qcb.codes.bb_72_12_6()

# depolarizing noise: X, Y, Z with probability p/3 each (exactly)
joint = qcb.DecodingProblem.from_css_code_depolarizing(Hx, Hz, p=0.06)
x_model, z_model = joint.css_split()   # what separate decoding uses:
                                       # from_css_code(Hx, Hz, 2p/3, "X") / "Z"

mine = qcb.BPOSD(osd_method="osd0", name="bp+osd0", role="probe")
cmp = qcb.compare_joint_separate(joint, shots=2000,
                                 joint_decoders=[mine],     # on the joint model
                                 separate_decoders=[mine])  # once per sector
print(cmp)                  # both certified ML brackets, the paired test,
cmp.ml_brackets()           # and every decoder used jointly and separately
cmp.paired_test()           # {'joint_better': {...}, 'separate_better': {...}}
cmp.decoder_vs_ml("bp+osd0", mode="joint", ml="separate")  # beats separate ML?
```

A shot is a separate-ML failure iff either sector fails. The paired tests are
one-sided and resolve uncertified shots against the claim being tested. The
bracket seeders (`seeders="auto"`: BP+OSD, and Tesseract when installed) run on
all three problems and are reported in both modes too. A separate decoder is
set up once per sector, so pass built-in names, `Decoder` objects or
`FunctionDecoder(factory=...)`: a bare function cannot tell the sectors apart.

Building joint models:

| constructor | noise |
|---|---|
| `from_css_code_pauli(Hx, Hz, px, py, pz)` | code-capacity Pauli channel; independent X, Y, Z mechanisms with probabilities solved so that the channel is reproduced exactly (needs `px(1-p) >= py*pz` and cyclically) |
| `from_css_code_depolarizing(Hx, Hz, p)` | the same with `px = py = pz = p/3` |
| `from_css_sectors(x_sector, z_sector, mechanisms)` | any CSS circuit: the two sector models plus every elementary fault as `(x_col, z_col, p)`, the columns of its X and Z parts (`-1` for none) |
| `DecodingProblem(H, L, priors, css=CSSPartition(...))` | a joint model you already have, e.g. a Stim circuit DEM |

`CSSPartition` records which detectors and observables belong to each sector;
the constructors above set it. `css_split()` returns the two marginal models,
whose columns are the distinct X (Z) parts of the joint mechanisms with the
exact probability that an odd number of them fires. Each has exactly the joint
model's law of its sector's syndrome and observables.

With uniform priors (code-capacity depolarizing noise) many solutions tie
exactly, which widens both brackets; circuit-level or non-uniform priors rarely
tie.

## Performance knobs

```python
cfg = qcb.CertConfig(true_node_budget=200_000, wrong_node_budget=100_000,
                     exact_ilp_time_limit=30.0)
qcb.benchmark(problem, decoders, shots=10_000, config=cfg, num_workers=16,
              target_failures=100)    # stop after 100 failures of the first PROBE
```

No knob can make a certificate wrong; knobs only trade wall-time against the
`undecided` fraction. Two last resorts help with the shots that stay
`undecided`:

```python
cfg = qcb.CertConfig(
    fast_retry_node_budget=1_000_000, pool_cap=20_000_000,  # big cap-free search
    use_global_route=True, global_time_limit=120.0)         # exact global ILPs
```

The global route takes an exact syndrome-only ILP witness, then proves by
infeasibility that no competing class costs less than the witness plus a
margin (margins 0.1 down to 1e-6). It is slow, about 1–2 minutes on a hard
distance-12 shot, but it trusts no MIP optimality claim.
`use_fast_pass=False` switches the cheap first pass off (for A/B tests only).

 The certificate is cheapest at low physical error rates.
For large circuit-level DEMs at high `p`, expect a growing `undecided` share:
raise the budgets and seed with Tesseract.

Results are identical for any `num_workers`. Workers are started with `fork`
when the calling process is single-threaded. Otherwise they are started with
`forkserver`, for example after an in-process run, because scipy's HiGHS
solver keeps a thread pool alive and forking a process that holds it can
deadlock the workers. `forkserver` needs picklable decoders (module-level
functions or classes). In scripts, protect the entry point with
`if __name__ == "__main__":`.

## Command line

```bash
qldpc-certificate-benchmark selftest      # brute-force validation of the engine
qldpc-certificate-benchmark demo --p 0.04 --shots 500   # BP+OSD-0 on [[72,12,6]]
```

## Citing

If you use this library, please cite our paper:

> *arXiv reference coming soon.*

## License

MIT
