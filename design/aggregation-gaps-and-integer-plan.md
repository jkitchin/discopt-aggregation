# discopt-aggregation — NLP gaps + integer aggregation plan (0.4.0)

**Status:** NOT STARTED. Successor to `aggregation-review-fixes-plan.md`
(complete at 0.3.0, 398 tests). Two goals: (1) close the remaining *NLP* gaps —
the O(tree) incidence bottleneck, the undemonstrated reliability win, and the
dual-recovery scope holes; (2) add **integer variable aggregation** (MIP-style
doubleton/affine elimination with integrality-implication rules), the largest
well-specified feature not in Naik et al.'s paper.

**Executor notes (read first — includes pitfalls learned in the 0.3.0 loop).**
- Work items in phase order; within a phase, in listed order. One item per loop
  iteration; commit after each item's validation passes. Never `--no-verify`.
- **Regression-test-first** for every fix-shaped item: add the inlined repro as
  a failing test, confirm it fails, fix, confirm the whole suite passes.
- Gates after every item:
  `ruff check src tests && ruff format src tests && mypy src`
  then `PYTHONUNBUFFERED=1 python -m pytest tests/ -q -p no:cacheprovider --timeout=120`.
  Hash-seed gate before finishing a phase: seeds `0 5 7` full-suite green.
- **Pitfalls that cost time in the last loop — avoid them:**
  - `discopt.Model.solve()` is a **global** spatial B&B on nonconvex NLPs. For
    any timing/reliability work use the local solver:
    `discopt.aggregation.benchmark._local_solve(model)` →
    `(status_str, iterations, wall, objective)`.
  - Background `pytest` buffers its output; you will see nothing until exit.
    Run foreground with a generous tool timeout, `PYTHONUNBUFFERED=1`, and
    pytest `--timeout=120` so a hang becomes a visible failure, not silence.
    Kill strays with `pkill -9 -f pytest` before re-running.
  - Until SC1 lands, **do not build test models with deep nonlinear elimination
    chains** (`y1=f(y0), y2=g(y1), …`) except to test that pathology — the
    incidence walk is still O(tree) and will hang at *aggregate* time.
  - `ruff format` after appending tests to a file (appends rarely match style).
  - Test counts/pins: `tests/test_packaging.py` pins `len(__all__)` and the
    version string — update both when you add exports or bump the version.
- File:line references may drift; locate by quoted content.

**Ground rules (unchanged from prior plans).**
- discopt / pounce / feral source untouched; upstream needs get filed, not vendored.
- New features opt-in/default-off (`integer_aggregation=False`); soundness
  fixes default-on with a CHANGELOG note.
- Every item ships with tests + a worked example for user-visible behavior;
  hash-seed-robust tests; honest negatives get written up, not hidden.
- The Claude skill (`src/discopt/aggregation/skill/`) is user-facing
  documentation: update it in the same item that changes behavior it describes.

Item IDs: `SC*` scaling, `RL*` reliability, `DU*` dual completeness,
`IA*` integer aggregation, `X*` release. Effort: S=1, M=2–3, L=4+ iterations.

---

## 0. Phase overview and dependencies

| Phase | Theme | Items | Exit gate |
|---|---|---|---|
| 1 | Scaling — the load-bearing NLP gap | SC1 SC2 | gr/lm aggregate a 5k-var DAE in seconds; build scaling measured |
| 2 | Dual completeness | DU1 DU2 | maximize + active-eliminated-bound duals recovered (or provably out of scope) |
| 3 | Reliability demonstration | RL1 RL2 | a bundled instance where aggregation demonstrably converges more instances — or a written honest negative with the implicit-path comparison |
| 4 | Integer aggregation | IA1 IA2 IA3 IA4 IA5 IA6 | MINLP with integer chains aggregates, round-trips, recovers integral values; default behavior unchanged |
| 5 | Release | X1 | 0.4.0 gate |

Dependencies (finish-to-start):
```
SC1 ──> IA1 (IA needs SC1's affine-coefficient extractor)
SC1 ──> RL2-at-scale and SC2's measurement (SC1 first, always)
RL1 ──> RL2
IA1 ──> IA2 ──> IA3 ──> IA4 ──> IA5 ──> IA6   (strict chain)
DU1, DU2: independent of everything (can interleave)
```
**SC1 is first, full stop** — it unblocks scale for every later phase and IA1
reuses its machinery.

---

## Phase 1 — Scaling

### SC1. DAG-memoised affine-coefficient extractor; retire `_isolated_affine_coeff` (M) — ✅ DONE

**Result.** Added `exprwalk.affine_coefficients(expr, nonlinear_names=frozenset())
-> (const, {name: coeff_or_None}) | None`: one `O(#distinct nodes)` bottom-up
pass memoised by `id(node)` (the `classify_linearity` pattern), representing each
node as `(const, {name: coeff}, nonlinear_set)` with the `+`/`-`/`*`/`/`/power/
FunctionCall/CustomCall/MatMul rules from the fix design and discopt's leaf-only
`_const_value` convention for the multiply/divide scale factor (so `y*(2*3)`
reads nonlinear exactly as discopt does). Routed `incidence.linear_vars` (one
pass per body, then O(1) lookups, keeping the `pivot_tol` filter) and
`strategies._constraint_is_linear` (linear iff no nonlinear var) through it, and
removed both `_isolated_affine_coeff` and `_collect_var_names` discopt imports
from the plugin (only comments now name them). **Validation:** oracle equivalence
527/0 over every `PROBLEMS` body and their `ld1/ecd2/ld2/d2` reduced models plus
10 hand bodies (coeff to 1e-12, None-agreement); the shared-DAG regression
(`linear_vars` on a depth-20 `e=e+e` body — a 2^20-node tree over an O(20)-node
DAG) drops from **10.3 s pre-fix to instant**; constant-term convention
`3x+2y-7 → (-7, {x:3, y:2})`; a `cstr_dynamic(n=6)` `lm` round-trip matches the
direct solve to 1e-4; full suite 404 passed. **Scope note:** the extractor is
`O(dag)`, so it only accelerates bodies with *real* DAG sharing (reaction_diffusion:
9k-node DAG / infinite tree). `cstr_dynamic(n=50)` `lm` still does not finish — its
`lm` substitution produces an expression whose *distinct*-node count itself grows
exponentially (tree/dag ≈ 2, no sharing), i.e. a substitution-copy blow-up that
is SC2's native-indexed-incidence / single-pass-substitution territory, not the
affine-walk this item fixes. Files touched: `exprwalk.py`, `incidence.py`,
`strategies.py`, `tests/test_aggregation_incidence.py`.

**Problem.** `incidence.linear_vars` calls discopt's `_isolated_affine_coeff`
once per candidate per constraint, and that walker is **O(tree)**. Measured: one
call on a substitution-expanded reduced constraint took **28.7 s** (a 2.5M-node
tree over a 5k-node DAG). Consequence: recursive `gr`/`lm` on any real DAE hangs
at *aggregate time* — before any solve, before any guard can fire. The metrics
module already escaped this via `exprwalk.classify_linearity` (0.3.0); incidence,
`strategies._constraint_is_linear`, and `guards` did not.

**Repro (turn into `tests/test_aggregation_incidence.py::test_linear_vars_scales_on_shared_dags`):**
```python
# Build a shared-DAG body whose tree expansion is huge, then time linear_vars.
import time
import discopt.modeling as dm
from discopt.aggregation.api import aggregate
from discopt.aggregation.incidence import build_incidence
from discopt.aggregation.testproblems import reaction_diffusion

m = reaction_diffusion(n=8)
rm = aggregate(m, method="lm").reduced_model      # constraints with tree>>dag
t0 = time.perf_counter()
build_incidence(rm)                                # TODAY: ~minutes. MUST BECOME: < 2 s.
assert time.perf_counter() - t0 < 2.0
```

**Fix design.**
- New function in `exprwalk.py`:
  `affine_coefficients(expr, nonlinear_names=frozenset()) -> tuple[float, dict[str, float | None]] | None`
  returning `(constant_term, {name: coeff_or_None})` — coeff is the summed
  affine coefficient when the variable participates *only* affinely
  (`2y + 3y → 5.0`, matching `_isolated_affine_coeff` semantics), `None` when it
  participates nonlinearly; the outer `None` only for an unanalyzable node.
  One bottom-up pass, memoised by `id(node)` (the `classify_linearity` pattern),
  representing each node as `(const, {name: coeff}, nonlinear_vars)`:
  - `+`/`-`: merge affine parts additively; union nonlinear sets.
  - `*`: if one side is a pure constant, scale the other; else every variable on
    both sides goes nonlinear.
  - `/`: constant divisor scales; anything else → nonlinear.
  - power / FunctionCall / CustomCall / MatMul: all reachable variables nonlinear.
  - `nonlinear_names` forces listed variables nonlinear (array-variable parity
    with discopt, exactly as `classify_linearity` does).
- Route through it:
  - `incidence.linear_vars` (keep the `pivot_tol` filter on the returned coeffs).
  - `strategies._constraint_is_linear` (a constraint is linear iff the extractor
    reports no nonlinear vars).
  - `guards._raw_define` consumers stay as-is (they operate on already-built
    expressions), but grep `_isolated_affine_coeff` across `src/` and eliminate
    every remaining use; the discopt import should disappear from the plugin.
- Keep `classify_linearity` (it is the boolean projection); implement it *via*
  `affine_coefficients` if that stays exact, else leave both.

**Validation.**
- **Oracle equivalence:** the 0.3.0 validation harness pattern — for every
  constraint body of every `PROBLEMS` model and its `ld1/ecd2/ld2/d2` reduced
  models, plus ~10 hand models (mixed `x + 2y - 3`, `x*y - z`, `x**2 + y`,
  `x/z`, `sin(x)+y`, `2y+3y`, `y + y**3` [y must be None], constants):
  `affine_coefficients` matches `_isolated_affine_coeff` per (body, var) —
  coefficient value to 1e-12 when both affine, None-agreement otherwise.
  Target ≥ 400 checks, 0 mismatches (the classify_linearity validation hit 442/0).
- The timing repro above (< 2 s where it was minutes).
- End-to-end: `aggregate(cstr_dynamic(n=50), method="lm")` completes < 10 s and
  round-trips via `_local_solve` (objective matches original to 1e-4).
- Constant-term correctness (needed by IA1): `affine_coefficients` on
  `3*x + 2*y - 7` returns `(-7.0, {'x': 3.0, 'y': 2.0})` (sign convention: the
  body as written).
- Full suite green; the guards/strategies tests unchanged.

**Files.** `exprwalk.py`, `incidence.py`, `strategies.py`, tests.

### SC2. Build-scaling measurement post-SC1; go/no-go on native indexed incidence (S spike, L if go) — ✅ DONE (PARK)

**Result (NO-GO / park).** Re-ran `profiling.build_scaling(indep_scaling_model,
…)` post-SC1 up to 16k flat vars for `d2` and `lm` (script under scratchpad; no
source change — the park path). Under the shipped default (`scalarize._DEFAULT_LIMIT
= 10_000` *per array*), the largest **fully-scalarized** point is n=8000 (two arrays
of 8000 → 16k flat vars); n≥5001-per-array arrays exceed the cap and are silently
not scalarized (`n_eliminated=0`), so 16k flat is exactly SC2's measurable top under
shipped behavior. Measured `aggregate` build time (seconds):

| flat vars | n | scalarize_s | incidence_s | aggregate_s (d2) | ×/doubling | aggregate_s (lm) | ×/doubling |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 000 | 1 000 | 0.055 | 0.025 | 0.142 | — | 0.163 | — |
| 4 000 | 2 000 | 0.182 | 0.047 | 0.393 | 2.76 | 0.345 | 2.12 |
| 8 000 | 4 000 | 0.890 | 0.084 | 1.410 | 3.59 | 1.295 | 3.75 |
| 16 000 | 8 000 | 3.642 | 0.201 | 5.394 | 3.83 | 5.159 | 3.98 |

At the 16k-flat-var target the build is **5.4 s** (d2) / 5.2 s (lm) — far under the
60 s go-bar. The curve *is* superlinear (~2.5–4×/doubling, ≈ quadratic), which
literally trips the ">2.5×/doubling" go-trigger, **but** root-cause profiling
localizes the superlinearity to **discopt-internal** `discopt.modeling.core._check_name`:
at n=8000 it is **2.976 s of the 3.995 s** scalarize (75%), 16 000 calls (one per
scalar variable `scalarize_model` creates), with per-call cost doubling per doubling
→ an **O(n²) name-uniqueness check**. The plugin's own machinery is **linear**: the
substitution walker `scalarize._rewrite` is 0.098 s cumulative at n=8000, and SC1's
`build_incidence` is 0.02→0.20 s (linear). This **refutes the R3 hypothesis** that
the quadratic came from "each pass re-substitutes into the growing model" — post-SC1
the substitution/incidence are linear; the residual quadratic is upstream, hit only
because scalarization materializes n scalar variables.

Native indexed incidence (F1 stage b) was scoped to fix a *substitution* blow-up that
SC1 already eliminated; it would sidestep `_check_name` only as a side effect of not
creating scalar variables. Since (a) absolute build time is ≪ 60 s through paper
scale (16k→5.4 s; even naive quadratic extrapolation puts 32k flat ≈ 20 s, still
under bar), (b) the plugin is already linear where F1 would act, and (c) the ground
rules forbid vendoring around a discopt-internal cost, the L-sized rewrite is **not
justified**. **Park** it. **Upstream item filed (recommendation):** discopt's
`Model.continuous`/`_check_name` uses an O(n) scan per add → O(n²) bulk creation;
a dict/set name registry makes it O(1). File against discopt; do not vendor.
Reconsider F1 only if a future model needs >30k *scalarized* vars in one build or if
discopt fixes `_check_name` and a plugin-side quadratic then surfaces. Files touched:
this doc only (park path).

**Problem.** 0.3.0's R3 profiling measured scalarize+aggregate scaling
~quadratically (0.045 → 1.272 s over 1k→8k flat vars) because each pass
re-substitutes into the growing model. SC1 may change the constants enough that
this stops mattering at the 30k-var scale of the paper's models — measure before
building the L-sized fix.

**Design.** Re-run `profiling.build_scaling` post-SC1 up to 16k flat vars with
`d2` and `lm`. **Go criterion:** if aggregate build time at 16k vars > 60 s or
clearly superlinear (>2.5× per doubling), implement native indexed incidence
(F1 stage b): `(name, index)` incidence nodes, substitution at `IndexExpression`
granularity, single-pass substitution per stage. Otherwise **park** with the
measured table appended to this plan (the R3 note pattern).

**Validation (if go).** Scaling curve ~linear where it was quadratic; round-trip
correctness at n=100 (recovered values match closed form); the full suite green.
**Files.** spike: none (script under scratchpad; table into this doc). If go:
`incidence.py`, `transform.py`, `scalarize.py`, tests.

---

## Phase 2 — Dual completeness

### DU1. Maximize-objective dual recovery (S–M)

**Problem.** `recover_duals` / `recover_duals_from_result` return
`available=False, reason="maximise"`. The fix is a sign normalization, but the
convention must be *verified*, not assumed.

**Design.**
1. **Verify discopt's convention empirically first** (one scratch script):
   solve a tiny maximize model directly (`m.maximize(-(x**2) - y**2)`, one
   equality) and check whether discopt's reported `constraint_duals` satisfy
   `∇f + Σ μ ∇body = 0` with `f` the *maximize* objective, or with `-f`.
   Record the finding in the duals module docstring.
2. Implement per the finding: internally treat maximize as minimize(-f) — negate
   the objective-gradient term in `_solve_defining_duals`'s RHS and in
   `kkt_stationarity_residual` — and report multipliers in discopt's own
   convention so recovered duals remain directly comparable to a direct solve's.
3. The KKT self-check must use the same normalized convention (it gates
   correctness; if the sign is wrong the self-check will force `available=False`,
   which is the designed failure mode — a wrong sign cannot ship as `available=True`).

**Validation.** A maximize model with an eliminable variable and an active
inequality: recovered duals match the *original model's* direct-solve duals to
1e-6 and pass `kkt_stationarity_residual < 1e-6`; the existing
`test_maximize_is_out_of_scope` is replaced by a positive test; minimize-path
tests untouched.
**Files.** `duals.py`, tests (`test_aggregation_duals*.py`), `examples.py`
(extend `example_dual_recovery`), skill `references/recovery-and-implicit.md`
+ `references/troubleshooting.md` (remove the "maximise" reason).

### DU2. Active-eliminated-bound γ-terms (M)

**Problem.** An eliminated variable sitting on a finite bound at the optimum
returns `available=False, reason="active_eliminated_bound:<name>"`. But the
information needed is already in the reduced solve: the box→inequality
constraints emitted for eliminated bounds (`constraint_origin` entries
`("bound_lo"|"bound_hi", var)`) carry their own multipliers, and those *are* the
bound γ-terms in base-model space.

**Repro to flip (from the 0.3.0 review, currently asserts unavailable —
`test_active_eliminated_bound_reported_unavailable`):**
```python
m = dm.Model("t")
x = m.continuous("x", lb=-10, ub=10)
y = m.continuous("y", lb=-10, ub=2.0)      # y = x^2 hits ub=2 at the optimum
m.subject_to(y - x**2 == 0.0)
m.minimize((x - 5.0) ** 2)
# true multiplier of the defining equality ≈ -2.536 (hand KKT / direct solve)
```

**Design.**
- In both recovery paths, collect bound-origin duals: reduced constraint `j`
  with origin `("bound_lo"|"bound_hi", var)` → `gamma[var] += ±mu_j` (sign per
  sense and discopt's normalized `<=` form — derive on paper, then verify with
  the self-check).
- Extend the stationarity system: the row for eliminated variable `v` gains the
  known term `gamma.get(v, 0.0)`; rows for kept variables need no change (the
  bound inequality's body is the substituted defining expression, so its dual
  already reaches kept variables through the surviving-constraint sum — confirm
  this by including bound-origin constraints in `surviving` mapped to a
  *synthetic* id, or handle explicitly; decide by which keeps
  `kkt_stationarity_residual` exact).
- Remove the `_active_eliminated_bound` bail **only when the corresponding bound
  inequality survived in the reduced model** (i.e. γ is observable). If it was
  pruned as redundant, it cannot be active — keep a defensive check. If duals
  for it are missing from the backend, keep `available=False`.
- `DualRecovery` gains `bound_duals: dict[str, float]` (γ per eliminated
  variable, default empty).
- The KKT self-check stays the gate: extend `kkt_stationarity_residual` to
  accept the γ map so an algebra error still becomes honest-unavailable.

**Validation.** The repro returns `available=True` with the defining-equality
multiplier matching the direct solve (±1e-4) and residual < 1e-6; an *interior*
case still recovers identically to 0.3.0; a pruned-bound case stays available
(γ=0); randomized property — 10 models with mixed active/inactive eliminated
bounds, every `available=True` has residual < 1e-6.
**Files.** `duals.py`, `transform.py` (only if origin plumbing needs a field),
tests, `examples.py`, skill `references/recovery-and-implicit.md`.

---

## Phase 3 — Reliability demonstration

### RL1. Iteration-cap plumbing for local solves (S)

**Problem.** The paper's reliability protocol counts convergence **within an
iteration limit** (3000 there). `_local_solve` and `reliability_sweep` expose no
cap, so a "fails in full space" instance cannot even be expressed.

**Design.** Inspect `discopt.solvers.nlp_pounce.solve_nlp_from_model(options=...)`
for the max-iterations key (likely `max_iter`; verify by passing it and watching
the iteration count clamp). Thread `options: dict | None = None` through
`_local_solve`, `benchmark_model`, `_solve_instance`, and
`reliability_sweep(..., solver_options=None)`. A capped run that stops at the
limit counts as **not converged**.

**Validation.** A deliberately hard instance with `{"max_iter": 3}` reports not
converged where the uncapped run converges; default behavior byte-identical
(`options=None`); sweep tests green.
**Files.** `benchmark.py`, `sweep.py`, tests.

### RL2. A tuned-to-fail reliability instance (M–L, mostly search compute)

**Problem.** The paper's headline — aggregation converges more instances — has
no bundled demonstration: every shipped problem converges ~100% for every
method. The 0.3.0 R1 attempt (recycle loop) was an honest negative for
*explicit* aggregation.

**Design (a bounded search procedure, not a hope).**
1. Write a scratch search script sweeping, for each of
   `gas_pipeline` (nonconvex Weymouth), `cstr_dynamic` (rational kinetics), and
   two new *candidate* variants (below), a **wider** parameter grid than the
   registered sweep ranges × iteration caps `{15, 25, 50}` × starting-point
   scale, comparing `original` vs `d2` vs `eliminate_implicit` (where a cyclic
   block exists) convergence per cell via RL1's cap.
2. Candidate variants to add to the search (each a small builder):
   - `gas_pipeline` pushed toward infeasibility (demand near max deliverable,
     supply pressure low) — IPM-style failures cluster near the feasible
     boundary;
   - a **badly-scaled** `cstr_dynamic` (rate constant `k ∈ [50, 500]`, states
     scaled ~1e3 apart) — elimination removes the worst-scaled rows, which is
     the paper's conditioning mechanism;
   - the closed recycle loop solved via `eliminate_implicit` (the R1 note said
     the genuine reliability win is the implicit path — test that claim).
3. **Acceptance:** a (problem, grid, cap) where, deterministically
   (rerun ×3, same statuses — determinism is guaranteed post-C6), `original`
   converges ≤ 85% of cells and some aggregated/implicit form converges at
   least 10 points more. Register it as `HARD_INSTANCE = TestProblem(...)` with
   the cap recorded, a `slow`-marked test asserting the gap with margin, and a
   paragraph in `design/aggregation-stage2-results.md` replacing the "needs a
   tuned-to-fail instance" caveat.
4. **Honest-negative fallback:** if the bounded search (≤ ~2 h compute) finds no
   explicit-aggregation gap but finds an implicit-path gap, ship that with the
   R1 explanation. If nothing at all: append the search table to this plan and
   the stage2 doc — that is itself the deliverable (do not overfit a fragile
   instance; flaky reliability tests are worse than none).

**Validation.** The registered instance's `slow` test passes ×3 runs; default
suite unaffected (`slow` deselected); stage2 doc updated with the grid figure
data (per-cell CSV via `sweep_to_csv`).
**Files.** `testproblems.py`, `sweep.py` (nothing beyond RL1), tests,
`design/aggregation-stage2-results.md`, skill `references/study-tools.md`.

---

## Phase 4 — Integer aggregation (all opt-in behind `integer_aggregation=True`)

Everything in this phase is default-off; with the flag off, behavior must be
byte-identical to 0.3.0 (assert this — see IA5). Soundness rule being
implemented (standard MIP presolve, Achterberg et al. 2020 §4;
Zhang & Yuan 2024): an integer variable `y` may be eliminated via
`coeff·y + rest = 0` only when integrality of `y` is **implied**:

> the constraint is fully affine; the pivot `coeff = ±1` (within 1e-9); every
> other variable in it is integer/binary; every other coefficient and the
> constant term are integral (within 1e-9).

(The general divisibility condition — pivot divides all coefficients — is
deliberately restricted to ±1; note it as a scoped-out refinement.)

### IA1. Integer candidates + integrality-implication filter (M)

**Design.**
- `candidate_variables(model, integer_aggregation=False)`: when the flag is on,
  also include scalar `INTEGER` and `BINARY` variables. Track a
  `integer_names: frozenset[str]` on `IncidenceModel` (all integer/binary
  variable names, candidates or not — needed for the "every other variable is
  integer" check and already-half-present via N4's total counting).
- In `incidence.linear_vars` (now SC1-based): for an **integer** candidate `y`,
  admit the edge only if `affine_coefficients` reports the body fully affine
  (no nonlinear vars, outer non-None), `|coeff_y| == 1 ± 1e-9`, every other
  participating variable ∈ `integer_names`, every other coefficient and the
  constant integral (`abs(c - round(c)) <= 1e-9`). Continuous candidates keep
  the existing (pivot_tol) rule unchanged.
- Thread `integer_aggregation` through `build_incidence` → `aggregate` → `solve`
  (the T1 signature-symmetry test will enforce `solve` parity automatically).
- Strategy filters need no change beyond what the edge filter enforces —
  verify: `ld1` on `y == 3` (integer) works via the same path; `ecd2`/`ld2`
  admit `y = ±x + c` integer pairs; `d2`'s "nonlinear f(x)" case can never admit
  an integer `y` (nonlinear body fails the affine requirement) — add a test
  asserting exactly that.

**Validation.**
- `j = i + 3` (both integer) eliminated with the flag, not without (the
  assessment repro).
- Rejections, each as its own test: `y = x + i` with continuous `x` (continuity
  in the definition); `y = 2*i` (pivot fine but — careful — here the *pivot* on
  `y` is 1 and the other coeff 2 is integral, so this IS admissible; the
  rejection case is eliminating `i` FROM `y = 2*i`, pivot 2 ≠ ±1);
  `y = i + 0.5` (fractional constant); `y = 0.5*i + z` (fractional coeff);
  binary `b = i1 + i2` admissible-by-rule (integrality implied) — include and
  confirm the recovered value lands in {0,1} or the model is infeasible via the
  bound conversion (IA3 covers the box).
- Flag off ⇒ `aggregate` results identical to 0.3.0 on the whole `PROBLEMS`
  corpus (compare eliminated lists).
**Files.** `incidence.py`, `api.py`, tests (`test_aggregation_integer.py`, new).

### IA2. Fixed integer variables and integral-infeasibility (S)

**Design.** `y == c` with integer `y`: eliminable iff `c` integral (1e-9) and
within `y`'s (integer) bounds. Non-integral `c` → append to
`infeasible_eliminations` (the C1 machinery; `solve` already returns
`infeasible`)... **but only when the flag admitted `y` as a candidate** — with
the flag off, `y` is not a candidate and nothing changes (an integer fixed to
0.5 is the *solver's* infeasibility to find, as today; add a test freezing that).

**Validation.** `i == 3` eliminated (flag on) with `res.x['i'] == 3`; `i == 2.5`
→ `solve` returns `infeasible` and `aggregate(...).infeasible_eliminations`
names `i`; flag off: both models behave exactly as 0.3.0.
**Files.** `transform.py` (the C1 constant check gains the integrality case),
tests.

### IA3. Integer-aware bound conversion and pruning (S–M)

**Design.** In `apply_aggregation`'s box→inequality conversion, when the
eliminated variable is integer: tighten first — `lb' = ceil(lb - 1e-9)`,
`ub' = floor(ub + 1e-9)` — and emit/prune against `[lb', ub']`. Pruning gets
strictly *more* effective (a defining expression with interval `(-0.4, 3.4)`
over integer-valued inputs... keep the enclosure real-valued and conservative;
only the compared box tightens — do NOT round the expression interval, that
would be unsound without proving the expression integer-valued). Same tightening
in `eliminate_implicit`'s bound emission (C4 path) for symmetry, though integer
block variables are out of scope for implicit elimination — assert they are
skipped rather than mishandled.

**Validation.** `y = i1 + i2`, `y ∈ [-0.5, 4.5]` integer → constraints use
`[0, 4]`; a bound made redundant only by the tightening is pruned; feasible-set
soundness property (sampled feasible integer points remain feasible in the
reduced model, the G2 harness pattern).
**Files.** `transform.py`, tests.

### IA4. Recovery integrality + duals scope (S)

**Design.**
- `RecoveryMap.recover` (or `AggregationResult.recover_full`): for an eliminated
  *integer* variable, assert `|v - round(v)| <= 1e-6` then return the rounded
  int-valued float. A larger residual raises — it would mean an IA1 rule was
  violated, which is a bug, not a data condition. (Plumb "which eliminated names
  are integer" through the recovery map — a `integer_vars: frozenset[str]`
  field, default empty.)
- Dual recovery: if any eliminated variable is integer →
  `available=False, reason="eliminated_integer_variable"` (KKT multipliers are
  not defined over the branching variables; do not attempt).

**Validation.** Round-trip on an integer chain recovers exactly integral values;
a hand-corrupted recovery map (monkeypatched non-integral definition) raises;
`solve(..., integer_aggregation=True, recover_duals=True)` on a mixed model
returns the new reason.
**Files.** `recovery.py`, `api.py`, `duals.py`, tests.

### IA5. MINLP validation problem + no-regression corpus (M)

**Design.** Add `inventory_chain` to `testproblems.py`: horizon `n`, integer
order quantities `q[t] ∈ [0, 10]`, integer stock balance
`s[t] = s[t-1] + q[t] - d_t` (integer demand data `d_t`), integer initial stock,
a nonlinear (continuous) holding-cost term so the model is genuinely MINLP, and
two sweep parameters (demand scale, holding cost). Under
`integer_aggregation=True`, the stock chain `s[1..n]` is eliminable (all-integer,
±1 coefficients; the *chain* is affine so there is no expression blow-up — but
run with `order="min_fill"` in tests anyway, and note SC1 makes the walk cheap).

**Validation.**
- Round-trip: `solve(m, method="lm", integer_aggregation=True)` matches the
  direct solve's objective (1e-6) and every recovered `s[t]` is integral and
  satisfies the balance exactly.
- Benefit, reported not asserted (solver node counts aren't exposed): wall-time
  and variable-count table original vs aggregated printed by the worked example;
  assert only `n_eliminated >= n-1` and objective equality.
- **No-regression corpus:** with `integer_aggregation=False` (default), the
  eliminated-variable lists on every `PROBLEMS` model are byte-identical to a
  pinned snapshot (write the snapshot in this item); `unit_selection` binaries
  remain untouched even with the flag ON (they appear in inequalities, not
  defining equalities — a good adversarial check).
**Files.** `testproblems.py`, tests, registry/`__all__` (+ packaging pin bump).

### IA6. Example, docs, skill (S)

**Design.** `example_integer_aggregation` in `examples.py` (chain model, before/
after sizes, recovered integral stocks) + asserting test; README "Methods"
section gains an integer paragraph; skill updates: `SKILL.md` options table row
(`integer_aggregation`), `references/methods.md` "What each method will NOT
eliminate" (now conditional), `references/options.md` recipe, and
`references/troubleshooting.md` ("integer not eliminated → check the four
implication conditions"). CHANGELOG under 0.4.0.
**Files.** `examples.py`, `README.md`, `src/discopt/aggregation/skill/*`, tests,
`CHANGELOG.md`.

---

## Phase 5 — Release

### X1. 0.4.0 gate (S)

1. Full suite green (default and `-m slow`) on `PYTHONHASHSEED ∈ {0, 5, 7}`;
   ruff + format + mypy clean; `python -m discopt.aggregation.examples` exits 0.
2. Spot-check two regression tests fail on pre-fix code (`git stash` the fix).
3. Version 0.4.0 in `pyproject.toml` + `__init__.__version__` + the packaging
   test's pinned string; `plugin.json` version bumped to match; `__all__` pin
   updated for new exports (`inventory_chain`, anything from DU/RL).
4. CHANGELOG `[0.4.0]`: SC1 as **Fixed** (performance/correctness-at-scale),
   DU1/DU2 as **Fixed** (scope holes closed), RL2 as **Added** (or the honest
   negative recorded in the stage2 doc), IA as **Added** (opt-in), with the
   integrality-implication rule stated verbatim.
5. Skill: re-run both `skill/examples/*.py` scripts (they are executable docs);
   confirm the pip-install path still works
   (`discopt-aggregation-skill install` in a temp dir).
6. Re-run the acceptance repros inlined in this plan end to end.

---

## Suggested loop invocation

```
/loop read design/aggregation-gaps-and-integer-plan.md and execute the next
unfinished item (phase order, one item per iteration): regression-test-first,
run the lint/type/test gates, commit with the item ID in the message, and mark
the item DONE in the plan with a one-paragraph result note. Stop when X1 passes.
```
Mark items directly in this file (`### SC1 … — ✅ DONE (note)`) so the loop is
resumable and the plan doubles as the log, as in the previous two plans.
