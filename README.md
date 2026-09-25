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
**exactly** on the sampled shots. `result.ml_bracket()` also widens the
bracket with one-sided Clopper-Pearson intervals to cover sampling error.

This is *non-degenerate* ML: the most likely single error pattern, which is
the target of minimum-weight / MLE decoders such as Tesseract and exact ILP
decoders. It is not the most likely logical coset. A decoder can
occasionally beat non-degenerate ML; those shots are reported as `lucky`.
Code-capacity models with *uniform* priors have many exactly-degenerate
minima and therefore many `tie`s. Circuit-level DEMs, whose priors are not
uniform, rarely produce them.

**How it is proven.** Each shot runs an exact A* search over the coset with
admissible heuristics, strengthened by a covering-LP dual bound, adaptive
parity-cut LPs (Feldman), and exact integer programs (HiGHS). A certified
cluster decomposition splits shots with independent error regions. Every
witness produced by an LP/ILP solver or a seeding decoder is re-verified in
NumPy before it can enter a proof, so a solver bug or a wrong decoder can
make a shot `undecided` but never flip a verdict. The engine is validated
against exhaustive enumeration by `qcb.selftest()`. The kernels are compiled
without `fastmath`, so the strict comparisons keep IEEE semantics.

## Reading the results

```python
result.ml_bracket()        # {'lb', 'ub', 'lb_ci', 'ub_ci', 'uncertified_frac'}
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

## Performance knobs

```python
cfg = qcb.CertConfig(true_node_budget=200_000, wrong_node_budget=100_000,
                     exact_ilp_time_limit=30.0)
qcb.benchmark(problem, decoders, shots=10_000, config=cfg, num_workers=16,
              target_failures=100)    # stop after 100 failures of the first PROBE
```

No knob can make a certificate wrong; knobs only trade wall-time against the
`undecided` fraction. The certificate is cheapest at low physical error rates.
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
