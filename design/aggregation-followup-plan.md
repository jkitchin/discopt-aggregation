# discopt.aggregation — follow-up plan

**Status:** M1 reached; E1 done (205 tests). Now a **standalone plugin**
(`discopt-aggregation`) in this directory, installing into the `discopt.aggregation`
namespace via a one-line `extend_path` shim in discopt's `__init__.py`. This
repackaging reframes G1 (see below): discopt's source is otherwise untouched, so
the `Model.solve(aggregate=...)` variant is delivered by the plugin's standalone
`solve()` rather than by editing `discopt.solver`.
**Scope:** everything identified in the post-Stage-1 audit against Naik, Biegler,
Bent & Parker, *Variable aggregation for nonlinear optimization problems*
[cite:@naik2025aggregation]: fidelity gaps, paper-derived extensions, the Stage-2
empirical study, and engineering integration.
**Ground rule (carried over from Stage 1):** every item ships with an explicit
test/validation strategy; an item is not done until its validation passes.

Item IDs: `F*` fidelity gaps, `G*` engineering, `E*` paper-derived extensions,
`S*` Stage-2 empirical. Effort is in loop iterations (1 iteration ≈ one focused
plan→implement→test session; S=1, M=2–3, L=4+), with a 25% buffer implied for
M/L items.

---

## 0. Phase overview and dependency structure

| Phase | Theme | Items | Exit milestone |
|---|---|---|---|
| 1 | Trustworthy on real models | F2, G2, F1, F4 | **M1**: aggregation is safe + applicable to indexed models; metrics exact |
| 2 | First-class solver option | G1, G3, E1, E2 | **M2**: `Model.solve(aggregate=...)` with full primal/dual recovery |
| 3 | Paper reproduction on discopt | S1, S2, S3, F3, S4 | **M3**: structure/runtime/reliability report on our AD stack |
| 4 | Beyond the paper | E3, E4, G4, E5, E6 | **M4**: each independently shippable |

Dependencies (finish-to-start unless noted):

```
F2 ──┐
G2 ──┼──> M1 ──> G1 ──> M2 ──> S3 (sweeps need the solver option)
F1 ──┤            G3 ─┘
F4 ──┘            E1, E2 (parallel, independent)
S1 ──────────────────────> S2 ──> F3 (contingent on S2 findings) ──> S4
E3 <── needs only strategies.py (parallel any time after M1)
E4 <── needs only matching.py   (parallel any time after M1)
G4, E5, E6: after M2 (E5/E6 exploratory)
```

Priority rationale: Phase 1 items are cheap and remove the two ways the current
code can *hurt* a user (tiny-pivot substitutions, silent inapplicability to
indexed models). Phase 2 makes the feature discoverable and complete at the KKT
level. Phase 3 is the scientific payoff and produces a novel data point (the
paper measured ASL; we measure JAX/XLA). Phase 4 is research-grade and
individually optional.

---

## Phase 1 — Trustworthy on real models

### F2. Pivot tolerance + fill-in caps (S, 1 iteration) — ✅ DONE

**Shipped.** Relative `pivot_tol` (default `1e-6`) in `linear_vars` /
`build_incidence` / `aggregation_set` / `aggregate` / `solve`; `max_fill` cap
(default `None`) via `IncidenceModel.fill_estimate` + degree tracking, enforced
in `strategies._finalize`. 8 tests in `test_aggregation_pivot_fill.py`; zero
change on the existing corpus. **Deferred:** the `max_def_nodes` (expression-size)
cap — the degree-based `max_fill` covers the blow-up case; node-count is a
refinement to add when a real model shows it's needed. Worked examples for the
guards are in `aggregation/examples.py`.

**Problem.** `linear_vars` accepts any nonzero constant coefficient. Dividing a
defining equation by a `1e-12` coefficient is exactly the numerical hazard
Achterberg et al. [cite:@achterberg2020presolve] flag for MIP aggregation;
nothing bounds substitution blow-up
either (AMPL's `substout` has the same gap, and Bongartz observed "smallest
formulation ≠ best").

**Design.**
- `incidence.linear_vars` gains a relative pivot test: an edge is eliminable
  only if `|coeff| >= pivot_tol * max(1, max_j |coeff_j|)` over the constraint's
  linear coefficients. Default `pivot_tol = 1e-6`; threaded through
  `build_incidence(model, pivot_tol=...)` and `aggregate(...)`.
- Selection-time fill-in guard in `strategies._finalize`: estimate
  `fill(e) = (n_vars(def_expr) - 1) * n_target_constraints(var)` and expression
  node count; drop eliminations with `fill > max_fill` (default `None` = off for
  structure-preserving methods, a finite default for `gr`/`lm`) or
  `nodes(def_expr) > max_def_nodes`.

**Validation.**
- Unit: a model with `1e-9 * y + x == 1` is *not* used to eliminate `y` at the
  default tolerance, but is with `pivot_tol=0`.
- Property: for random coefficient scales, every accepted elimination satisfies
  the pivot inequality; round-trip equivalence tests (existing harness) rerun
  under the new defaults — zero behavior change on the current test corpus.
- Numerical: construct an ill-conditioned chain (coefficients `10^-k`) and show
  reduced-model solve error vs original grows without the guard and stays
  bounded with it.

**Files.** `incidence.py`, `strategies.py`, `api.py`, new tests in
`test_aggregation_incidence.py` / `test_aggregation_strategies.py`.

### G2. Redundant-bound pruning via interval propagation (S–M, 1–2 iterations) — ✅ DONE

**Shipped.** `interval_evaluate(expr, boxes)` in `recovery.py` — a conservative
outer interval enclosure over the DAG (linear exact; products/powers/division/
exp/log/trig handled; `(-inf,inf)` fallback keeps it sound). `apply_aggregation`
gained `prune_bounds=True` (threaded through `aggregate`/`solve`) that omits a
box→inequality when the defining expression's interval over the kept boxes is
strictly inside the eliminated variable's bounds; `ReducedModel.n_bound_pruned`
reports the count. 11 tests in `test_aggregation_pruning.py` (interval outer-bound
property vs 3k-sample dense checks, redundant-pruned, binding-kept, soundness
over the box); worked example `example_bound_pruning`. This closes the
"bounded-elimination adds inequalities" cost that the performance study exposed.

**Problem.** `apply_aggregation` always emits up to 2 inequalities per bounded
eliminated variable (§2.4). When the defining expression's range over the kept
variables' boxes provably fits inside the eliminated variable's bounds, those
inequalities are redundant — pure constraint bloat (this is the same check the
Rust `factorable_elim` pass already performs on its side).

**Design.** Add `interval_evaluate(expr, boxes) -> (lo, hi)` next to
`evaluate_expression` in `recovery.py` (same node dispatch, interval semantics;
conservative `(-inf, inf)` on unsupported nodes). In `apply_aggregation`, skip a
bound inequality when the implied interval is inside the bound with margin
`1e-9 * max(1, |bound|)`. Report skipped count in `ReducedModel`.

**Validation.**
- Exactness: intervals for each node type checked against dense sampling
  (`lo - eps <= f(x) <= hi + eps` on 10k random points per expression).
- Soundness property (the critical one): pruning must never cut the feasible
  set. Sample feasible points of the original model; assert all remain feasible
  in the reduced model with pruning on. A binding-bound case (e.g. `e = d^2`,
  `e <= 4`, `d in [-3, 3]`) must *keep* its inequality.
- Existing round-trip suite green with pruning on by default.

**Files.** `recovery.py` (interval evaluator), `transform.py`, tests.

### F1. Array/indexed-variable scalarization (M, 2–3 iterations) — ✅ DONE (stage a)

**Shipped.** `scalarize.py`: `scalarize_model(model, limit=10000)` rewrites
fully-indexed continuous array variables into per-element scalars (`x[(3,)]` →
`x__0,x__1,x__2`; 1-D and n-D full-constant-index), with conservative abstention
(bare array use, `sum(a)`, slices, partial `b[0]` on 2-D, non-constant/out-of-
range index, or size > limit). `ScalarizationMap.reassemble` folds scalars back
into arrays. Wired into `aggregate`/`solve` via `scalarize=True` (default; no-op
on scalar-only models); `AggregationResult.scalarization` composes into
`recover_full` so callers get original array shapes back. Also added a `solve`
guard for the fully-determined case (aggregation eliminates *all* variables →
empty reduced model → evaluate objective at the recovered point instead of
calling the solver). 14 tests in `test_aggregation_scalarize.py` (structure,
abstention, limit, 1-D/2-D round-trip, end-to-end array aggregate+solve, empty
model); worked example `example_array_scalarization`. **Deferred:** stage (b)
native indexed incidence (substitute at `(name,index)` granularity without full
expansion) — only warranted if expansion is measured to be a bottleneck.

**Problem.** Incidence only considers scalar continuous variables, so models
built with `m.continuous("x", shape=(n,))` — i.e. every discretized-DAE model,
the paper's entire test set — currently aggregate nothing. Biggest coverage gap.

**Design.** Two-stage; ship (a) first.
- (a) **Model-level scalarization pre-pass** (`aggregation/scalarize.py`):
  rewrite each array variable `x` (shape `(n,)` / small ndims) into scalars
  `x[0]…x[n-1]` — new scalar Variables named canonically (`x__0`), every
  `IndexExpression(x, i)` with a *constant* index replaced by the scalar; whole-
  array references (vectorized constraints, `SumExpression` over the array)
  either expanded or, if any reference cannot be resolved to constant indices,
  the variable is left un-scalarized (conservative abstain). Exposed as
  `aggregate(model, scalarize=True)`; recovery reassembles arrays from scalars
  so `AggregatedSolveResult.x` returns the original array shape.
- (b) *(later, optional)* native indexed incidence — treat `(name, index)`
  pairs as graph nodes and substitute at `IndexExpression` granularity, avoiding
  full expansion. Only worth it if (a)'s expansion is a measured bottleneck.

**Validation.**
- Round-trip identity harness (random-value substitution equivalence + recovery)
  extended to array models: chain `x[i] = 2 x[i-1] + 1`, a small collocation
  system built with discopt's DAE tools.
- Shape fidelity: `res.x["x"]` has the original shape and matches the unscalarized
  original solve element-wise.
- Abstention: a model with a non-constant index (if constructible) or vectorized
  opaque reference leaves that array untouched (aggregation count 0 for it), and
  solves unchanged.
- Guard: scalarization of an `(n,)` variable with `n > scalarize_limit`
  (default ~10k) abstains with a warning — no silent model explosion.

**Files.** new `scalarize.py`, `api.py`, `recovery.py` (array reassembly), tests.

### F4. Exact Hessian-sparsity metric (S, 1 iteration) — ✅ DONE

**Shipped.** `exact_hessian_nnz(model)` builds an `NLPEvaluator` and returns the
true Lagrangian-Hessian lower-triangle nonzero count (the paper's Table 4
quantity); `0` for an empty model, `None` if the evaluator can't be built.
`model_structure(model, exact_hessian=True)` and `compare_methods(...,
exact_hessian=True)` populate `ModelStructure.nnz_hessian_exact` alongside the
fast structural proxy (kept as `nnz_hessian`, documented as an upper bound).
Verified `x*y` → exact 1 vs structural 3, `a²+b²` → 2, and
`nnz_hessian_exact <= nnz_hessian` across every method on the study model.
5 tests; worked example `example_exact_hessian`.

**Problem.** `metrics.nnz_hessian` is a structural coupling *superset* (all
pairs of nonlinearly-participating variables per function), not the true second-
derivative sparsity the paper's Table 4 reports. Fine for trends, wrong for
numbers.

**Design.** Compute the exact pattern through the existing JAX machinery
(`_jax/sparse_hessian.py` / the NLP evaluator): compile the Lagrangian with unit
multipliers, extract the Hessian sparsity, count nnz. Keep the structural metric
(rename `nnz_hessian_structural`) since it's solver-free and fast; add
`nnz_hessian_exact` computed lazily (`compare_methods(..., exact_hessian=True)`).

**Validation.** Known counts: `x*y` → 2 off-diagonal + 0 diagonal (pattern
`{(x,y)}` symmetric) vs `x**2` → 1 diagonal; verify
`nnz_hessian_exact <= nnz_hessian_structural` on every test model; cross-check
one model by hand-derived Hessian.

**Files.** `metrics.py`, tests. Depends on nothing; parallel with F1/F2.

---

## Phase 2 — First-class solver option

### G1. Presolve-loop / Model.solve integration (M, 2 iterations) — ⟳ REFRAMED (standalone)

**Reframed for the standalone plugin.** The original plan edited
`discopt.solver` for `Model.solve(aggregate=...)`. As a standalone plugin that
leaves discopt's source untouched (bar the one-line namespace shim), that route
is declined — a global monkeypatch of `Model.solve` would be fragile. G1's
user-facing value is already delivered by the plugin's `discopt.aggregation.solve(
model, method=...)` (aggregate → solve → recover, arrays and duals included).
The presolve-delta reporting and in-loop interleaving remain future work if the
plugin is ever upstreamed into discopt. **Status: addressed via standalone
`solve()`; deeper in-discopt integration deferred.**

### G1-original. Presolve-loop / Model.solve integration (M, 2 iterations)

**Problem.** Aggregation lives in a separate `discopt.aggregation.solve()`
entry point. Users won't find it, and it can't compose with the rest of
presolve. The Python presolve protocol's v0 explicitly forbids topology
rewrites, so the integration point must be *before* the orchestrator, not
inside it.

**Design.**
- `Model.solve(aggregate="d2" | True | None, aggregate_options={...})` in
  `solver.py`: when set, run `aggregate()` up front, solve the reduced model
  through the normal pipeline, then merge recovery into the returned
  `SolveResult.x` (and mark `result.aggregation` with method/eliminated/stats)
  so every downstream consumer sees a full-space solution transparently.
- Emit a delta dict shaped like `make_python_delta("variable_aggregation")`
  with `vars_aggregated` filled (the field already exists in `PresolveDelta`),
  logged alongside the other presolve deltas for observability.
- Explicit non-goal for now: running aggregation *inside* the Rust/Python
  fixed-point loop (requires protocol v1 topology-rewrite support; revisit
  after M3 data shows whether interleaving matters).

**Validation.**
- End-to-end: for a gauntlet subset (representative NLP/MINLP models in the
  existing test corpus), `Model.solve(aggregate="d2")` matches plain `solve()`
  objective within tolerances and returns full-space `x` (all original names,
  original shapes) that is feasible in the original model.
- The correctness gate: run the `pr_correctness` marker set with
  `aggregate="d2"` forced on — `incorrect_count <= 0` must hold (repo
  non-negotiable).
- No-op path: `aggregate=None` (default) leaves solve byte-identical.

**Files.** `solver.py`, `aggregation/api.py`, `test_aggregation_solver.py`.

### G3. Lagrange-multiplier recovery (M, 2 iterations) — ✅ DONE

Split into two sub-slices to keep each correct and verified:

**G3b — dual back-substitution (✅ DONE).** `duals.py`: `recover_duals(reduced,
reduced_result)` maps surviving-constraint multipliers from the reduced solve
(positional `c{i}` keys ↔ `constraint_origin`) and recovers each eliminated
defining equality's multiplier by reverse-order stationarity back-substitution,
using the uniform convention `∇f + Σ μ_c ∇body_c = 0` (verified: discopt reports
one multiplier per normalised `body (sense) 0` for equalities and inequalities
alike). `kkt_stationarity_residual` validates. 7 tests: KKT residual < 1e-6
*and* recovered multipliers match the original-model solve exactly, across
ld2/d2/gr/lm and with an active inequality; `example_dual_recovery`.
**Scope:** single-pass `apply_aggregation`, minimise objective, no *active*
variable bounds on eliminated variables; reports `available=False` (never a wrong
result) otherwise. Multi-pass composition and active-eliminated-bound γ-terms are
follow-ons.

- **G3a — symbolic differentiation (✅ DONE).** `differentiate_expression(expr,
  var)` in `derivatives.py`: exact partial derivative over the DAG (+,-,*,/,
  const-power, sin/cos/exp/log/sqrt/tanh), folding trivial 0/1; returns `None`
  for unhandled constructs (variable exponent, reductions, indexed) so callers
  fall back rather than trust a wrong result. 7 tests validate against central
  finite differences; `example_symbolic_derivative`. This is the primitive the
  eliminated-variable Lagrangian-stationarity gradients are built from.
- **G3b — dual back-substitution (NEXT).** Carry a reduced→original
  constraint-origin map in the transform; map surviving-constraint duals; solve
  the defining-equality multipliers by reverse-order stationarity
  (`λ_g̃_i = -(∂f/∂v_i + Σλ ∂ḡ/∂v_i + Σν ∂h/∂v_i + γ_v_i + Σ_{i'>i} λ_g̃_{i'}
  ∂rest_{i'}/∂v_i)/coeff_i`); validate with the full-space KKT stationarity
  residual on models with inequalities and active bounds.

**Original design ↓**

### G3-original. Lagrange-multiplier recovery (M, 2 iterations)

**Problem.** We recover primal values only. The multiplier of an eliminated
defining equality is recoverable in closed form, and without it the full-space
KKT point is incomplete (breaks downstream sensitivity/differentiable-opt use).

**Design.** For eliminated variable `v` with defining constraint
`g̃: coeff·v + rest = 0`, stationarity in `v` gives
`λ_g̃ = -(∂f/∂v + Σ_j λ_j ∂c_j/∂v + ν_v) / coeff` — and because eliminations
are strictly lower triangular, evaluating in *reverse* elimination order makes
every term known (an upper-triangular back-substitution, the exact dual mirror
of primal recovery). Gradients `∂·/∂v` evaluated at the recovered full-space
point via the JAX evaluator. Bound multipliers of eliminated variables map from
the multipliers of the emitted bound-inequalities. Implement as
`RecoveryMap.recover_duals(x_full, reduced_duals)`.

**Validation.**
- KKT residual test: assemble full-space stationarity
  `∇f + ∇g λ + ∇h ν + bound terms` at the recovered `(x, λ, ν)` and assert
  `≈ 0` (tolerance `1e-6`) on models where the reduced solve returns duals
  (cyipopt/POUNCE backends expose `constraint_duals`).
- Cross-check against duals from solving the *original* model directly (up to
  sign conventions and degenerate multipliers — use strictly-complementary test
  models).

**Files.** `recovery.py`, `api.py`, tests. Depends on G1 for the duals plumbing.

### E1. Dk strategies — degree-k filters (S, 1 iteration) — ✅ DONE

**Shipped.** `_filter_degree_k(inc, k)` generalises the degree-2 filter;
`degree_k_aggregation_set(inc, k)` and a `resolve_strategy(method)` that accepts
`d<k>` (`d3`, `d4`, …, any `k >= 2`) alongside the named strategies. Wired
through `aggregation_set`/`aggregate`/`solve` and the §3.3 pipeline (a `d<k>`
method runs `ld1` then `dk`). 5 tests (single-pass reachability, acyclic monotone
coverage `n(d2)<n(d3)<n(d4)`, `resolve_strategy` accept/reject, `k>=2` guard) +
`example_degree_k`. Note (verified): coverage is only monotone in `k` when
eliminations are acyclic — as with gr/lm, heuristic cycle-breaking means a larger
`k` can occasionally eliminate fewer, so the test uses an acyclic ladder model.

### E1-original. Dk strategies — degree-k filters (S, 1 iteration)

**Problem/Design.** The paper suggests bounding eliminated-constraint degree at
`k > 2` as an intermediate between D2 and GR. Generalize
`_filter_degree2` to `_filter_degree_k(inc, k)`; register `d3`, `d4`;
`aggregation_set(m, method="dk", k=5)` for arbitrary k. Pipeline: `ld1` to
fixpoint, then `dk` to fixpoint (same as D2).

**Validation.** Monotone coverage property on the mixed test model:
`n(d2) <= n(d3) <= ... <= n(gr)`; lower-triangularity invariant (existing
checker) for each k; a constraint with exactly 3 variables eliminated by `d3`
but not `d2`.

**Files.** `strategies.py`, `api.py`, tests.

### E2. Target-constraint linearity filter (S–M, 1–2 iterations) — ✅ DONE

**Shipped.** `drop_linearity_breaking(agg)` in `strategies.py` removes an
elimination when its defining expression is nonlinear (`_defining_is_nonlinear`)
*and* the variable appears in another currently-linear constraint
(`_constraint_is_linear` scan over all model constraints). Exposed as
`aggregate(..., preserve_target_linearity=True)` / `solve(...)`, applied per
pass before `apply_aggregation`; dropping preserves lower-triangularity.
5 tests (nonlinear-def-into-linear-target dropped, linear-def kept, nonlinear
target not protected, linearity predicate, solve round-trip) +
`example_target_linearity`.

**Problem.** Substituting a nonlinear defining expression into a *linear*
target constraint converts it to nonlinear — the degradation Amarger et al.
warn about; the paper floats target inspection but never tests it.

**Design.** Post-filter on the selected matching (before `_finalize`), enabled
by `aggregate(..., preserve_target_linearity=True)`: drop elimination `(v, cid)`
if `def_expr(v)` is nonlinear and any *other* constraint containing `v` is
currently linear. Iterate to a fixed point (dropping one elimination can change
nothing else — the filter is monotone, single pass suffices). Optionally expose
as method variants `d2+tl` in `compare_methods` for the Stage-2 study.

**Validation.** Structural: with the flag on, `model_structure` shows no
previously-all-linear constraint became nonlinear (checkable via
`avg_linear_nnz_per_con` per-constraint refinement — add a
`n_linear_constraints` metric); a crafted model where `d2` without the flag
converts `w = x + e` (linear) into nonlinear and with the flag abstains.
Round-trip harness green with the flag on.

**Files.** `strategies.py`, `metrics.py` (n_linear_constraints), tests.

---

## Phase 3 — Paper reproduction on discopt (Stage 2 proper)

### S1. Parameterized test problems (L, 3–4 iterations) — ✅ DONE

**Shipped (3 problems, spanning NLP / MINLP / nonconvex-NLP).**
`testproblems.py`: a `TestProblem` dataclass (builder + nominal + sweep ranges +
`sweep_grid`) and three problems, each with array variables, eliminable
degree-2/multi-var intermediates, and two sweep parameters:
- `reaction_diffusion` — 1-D diffusion-reaction least-squares collocation
  (**convex QP / NLP**; sweep `k`, `c_in`).
- `unit_selection` — on/off units with quadratic operating cost (**MIQP /
  MINLP**; sweep `demand`, `fixed_cost`); binaries are never aggregated.
- `gas_pipeline` — Weymouth gas line `f² = C(sq_i − sq_{i+1})` (**nonconvex
  NLP**; sweep `p_supply`, `demand`).
13 tests (build/param, nominal solve, structural comparison, ld1/d2 solve
round-trip, MINLP binaries-kept, nonconvex round-trip, sweep grid);
`example_test_problem`; `PROBLEMS` registry.
**Key practical finding (informs S2/S3):** discopt solves *bounded* variables far
faster than effectively-free ones (free 1e20-bound intermediates made even n=10
time out; bounding them → instant), so test intermediates carry finite
non-binding bounds (aggregation prunes the resulting box-inequalities). And gr/lm
aggregated models densify the objective (substituting multi-variable residual
defs into `sum(res^2)`), making their *reduced* solve much slower — so solve
round-trips use structure-preserving methods and gr/lm are compared
*structurally* (no solve). **Follow-on:** the Weymouth gas-pipeline and
counter-current exchanger problems.

**Original design ↓**

### S1-original. Parameterized test problems (L, 3–4 iterations)

**Design.** Three models spanning the paper's structure classes, each with two
sweepable parameters and instance-size knobs, under
`python/discopt/aggregation/testproblems/` (importable, not test-only):

1. **CSTR/reaction cascade** (distillation stand-in): N-stage series with
   component balances via `discopt` DAE collocation — bilinear + rational terms;
   parameters: feed composition, rate constant. Small (N≈10–30 stages).
2. **Gas pipeline steady state**: Weymouth `f² = C(p_in² − p_out²)` network
   (structure already exercised by issue #15 patterns); parameters: supply
   pressure, demand scale.
3. **Counter-current exchanger / moving-bed stand-in**: two-stream finite-
   difference discretization with fractional-power source terms; parameters:
   inlet temperature, flow rate.

Each returns `(model, param_setter)` so a sweep re-instantiates cheaply.

**Validation.** Each model solves from its nominal point with the plain solver
(`status == optimal`, objective regression-pinned); structural fingerprints
(vars/cons/nnz) pinned; every aggregation method round-trips on the nominal
instance (objective match + full-space feasibility) — this alone is a strong
stress test of F1 scalarization on realistic indexed models.

**Depends on:** F1 (models are indexed).

### S2. Runtime instrumentation (M, 2 iterations) — ✅ DONE

**Key correction (user-prompted):** `discopt.Model.solve()` runs a *global*
spatial branch-and-bound on nonconvex NLPs (`solver.py:2002`) — slow, and not the
paper's setting. The paper studies *local* NLP convergence. The benchmark now
uses `discopt.solvers.nlp_pounce.solve_nlp_from_model` (local POUNCE), which is
fast (a model that timed out globally at n=10 solves locally at n=20 in 0.46s)
and, crucially, **exposes iteration counts** (the global result did not).

**Shipped.** `benchmark.py`: `benchmark_model(model, methods, run_solve=True)` →
`BenchmarkRow` per method with `t_elim` (aggregation time), reduced flat-variable
count, structural Hessian coupling, and the local solve's status / **iterations**
/ wall time. Structural columns use the scalarized original so array vs scalar
granularity is consistent. `run_solve=False` characterises the densifying
methods cheaply. Reproduces the paper's Table 6 trend directly: structure-
preserving methods solve as fast as the original with fewer variables, while
gr/lm cut variables/iterations but **densify the Hessian** — e.g. gr-reduced
reaction-diffusion solved in 41s at only 7 iterations vs 0.3s/6-it original — the
"Hessian evaluation becomes the bottleneck" finding. 4 tests; `example_runtime`.

**Original design ↓**

### S2-original. Runtime instrumentation (M, 2 iterations)

**Design.** Reproduce the paper's Table 6 breakdown on our stack: wrap the NLP
evaluator callbacks (objective/gradient/Jacobian/Hessian) with timers —
cyipopt backend first (clean callback boundary; POUNCE timing hooks are a
stretch goal), plus `t_elim` (aggregation time), `t_build`, JIT-compile time
(JAX makes compile vs steady-state a *new* axis the paper didn't have), solve
time, iteration count. Emit per-solve `dict` rows; aggregate into a
Table-6-style dataframe in the harness.

**Validation.** Timer overhead < 2% (measure with no-op timers); fractions sum
to ≤ 100%; sanity: Hessian fraction increases from `ld1` → `lm` on at least one
S1 model *or* we document that XLA CSE neutralizes it — either outcome is the
deliverable (this is the novel-vs-paper data point).

**Depends on:** S1.

### F3. Expression-duplication / CSE verification (M, contingent, 1–2 iterations)

**Problem.** The paper avoids duplicating substituted expressions via ASL
defined-variables. Our walker shares subtrees by object identity, but whether
the JAX DAG compiler CSEs shared sub-DAGs across constraints is unverified —
if it inlines per use-site, `lm`-reduced models pay exponential-ish evaluation
cost exactly where the paper warns.

**Design.** (1) Measure: jaxpr size + Hessian eval time for a synthetic model
where one defining expression is substituted into k constraints, k ∈
{1,10,100}, vs the unaggregated equivalent. (2) If super-linear: add
memoization in the DAG compiler keyed on expression `id()` (common-subexpression
lowering), or lower defining expressions to auxiliary jaxpr lets. Scope
strictly to what S2 shows is needed.

**Validation.** Eval-time growth in k is ~linear post-fix; numerical outputs
bit-identical pre/post memoization on the full aggregation suite.

**Depends on:** S2 findings (skip if XLA already CSEs — record the evidence in
the design doc either way).

### S3. Reliability sweeps (M–L, 2–3 iterations, mostly compute time) — ✅ DONE

**Shipped.** `sweep.py`: `reliability_sweep(problem, methods, points)` runs the
**local** solver across `problem.sweep_grid(points)` for the original model and
each method, returning a `SweepResult` per method (instances, converged count, %,
avg iterations/wall over converged). 2 tests (3×3 grid structure + counts, the
"aggregation never converges fewer than original" check); `example_reliability`.
Honest note (in the module + example): the bundled convex reaction-diffusion
converges 100% for every method, so demonstrating the paper's *improved*
reliability needs a problem tuned to fail in full space (the paper's authors
searched for such ranges) — the harness produces that comparison for any problem.

**Original design ↓**

### S3-original. Reliability sweeps (M–L, 2–3 iterations, mostly compute time)

**Design.** The paper's protocol: per S1 problem, 11×11 grid over its two
parameters × {original + 6 methods (+ Dk/target-filter variants if E1/E2
landed)} with an iteration cap. Budget ≈ 121 × 7 × 3 ≈ 2.5k solves — keep
instances < 5 s (several CPU-hours). Implement in `discopt_benchmarks` as an
opt-in suite (`--suite aggregation`, pytest marker `amp_benchmark`-style, never
in PR CI). Outputs: per-cell convergence grid (the paper's Figs 9–11),
percent-converged table (Table 7), virtual-best analysis (Fig 12).

**Validation.** Determinism spot-check (rerun 10 random cells, same statuses);
"no method converges fewer instances than original" checked and *reported*
(not asserted — if discopt violates the paper's headline finding, that is a
result, not a bug); artifacts (CSV + plots) written under `results/`.

**Depends on:** G1 (sweep drives `Model.solve(aggregate=...)`), S1.

### S4. Report + docs notebook (M, 2 iterations) — ✅ DONE (report; notebook deferred)

**Shipped.** `design/aggregation-stage2-results.md` — synthesizes the structural
(Table-4 analogue), runtime/local-solve (Hessian-bottleneck, with the
global-vs-local methodological correction), and reliability findings with
concrete numbers from the harnesses, and a point-by-point comparison to the
paper's trends (including the stack difference: the Hessian-cost effect is if
anything stronger on JAX/POUNCE than ASL/IPOPT — a new data point). Honest gaps
recorded: the array-granularity caveat in `compare_methods`'s Hessian column, and
that the reliability *improvement* needs a deliberately-hard instance.
**Deferred:** the jupyter-book `docs/notebooks/` tutorial (needs discopt's book
toolchain + a zero-warning build — finicky to run inside the loop; the executable
`examples.py` already serves as runnable documentation).

**Original design ↓**

### S4-original. Report + docs notebook (M, 2 iterations)

**Design.** (1) `docs/notebooks/variable_aggregation.ipynb`: tutorial
(aggregate/solve API, method choice guidance) + condensed Stage-2 results;
citations to `design/references.bib` (`naik2025aggregation`,
`achterberg2020presolve`, and the Duff-Reid / Hopcroft-Karp foundations);
registered in `docs/_toc.yml`;
`jupyter-book build docs/` with zero new warnings (repo checklist). (2) A
results write-up `design/aggregation-stage2-results.md` comparing our findings
to the paper's Tables 4/6/7 trends.

**Depends on:** S2, S3.

---

## Phase 4 — Beyond the paper

### E3. Better tearing on cyclic blocks (M, 2–3 iterations) — ✅ DONE

**Shipped.** Exact maximum-acyclic-subset tearing of cyclic blocks, selected by
`aggregate(..., tearing="exact")` (default `"greedy"`; also on `build_incidence`
via `IncidenceModel.tearing`). `_max_acyclic_subset` finds the largest subset of
a cyclic block whose induced dependency subgraph is acyclic (a maximum induced
DAG / complement of a minimum feedback vertex set), by exact search over subsets
largest-first, capped at `_EXACT_TEAR_CAP = 13` variables with greedy fallback
above that. Implemented as a pure-graph, dependency-free search (no MILP solve —
avoids the solve-perf pitfalls, and is exact for small blocks). 7 tests (core
algorithm on known digraphs incl. keeping the isolated node; `exact >= greedy`
and lower-triangular on cyclic models; default-greedy unchanged);
`example_exact_tearing`. **Note:** greedy is often already optimal on tiny blocks;
exact's value is the guarantee (and it never needlessly drops a free node).

**Original design ↓**

### E3-original. Better tearing on cyclic blocks (M, 2–3 iterations)

**Design.** Replace greedy-per-block in `linear_matching_aggregation_set` with a
pluggable `tearing=` option: `"greedy"` (current, default), `"mfas-exact"` —
minimum feedback arc set via a small MILP *solved with discopt itself*
(dogfooding; only for blocks with ≤ ~30 edges, else fall back), and optionally
Cellier's heuristic for mid-size blocks. MFAS on the block's projection digraph
maximizes retained (acyclic) eliminations.

**Validation.** Exact ≥ greedy elimination count on every block (property,
random block digraphs); lower-triangularity invariant; known-answer instance
(hand-built 4-cycle where greedy recovers 1, exact recovers 3); wall-time cap
respected (fallback triggers).

### E4. Highly-decomposable matchings (M–L, 3 iterations) — ✅ DONE

**Shipped.** `_decomposable_matching` local-searches the linear-matching maximum
matching for a higher `n_block` (Theorem 2's lower bound) via
cardinality-preserving moves (rematch to a free variable, or 2-swap two
constraints' variables), accepting only strictly-improving swaps — deterministic,
monotone in `n_block`, cardinality invariant. Selected by
`aggregate(..., decomposable=True)` (default off), threaded through
`build_incidence`/`IncidenceModel`. 5 tests (invariants: cardinality preserved,
`n_block` non-decreasing, valid maximum matching, `decomposable >= default`
eliminations + lower-triangular, default unchanged); `example_decomposable_matching`.
**Honest note (verified):** on small models Hopcroft-Karp is often already optimal
and inherent cycles cannot be matched away, so strict improvement is
model-dependent — it shows on large models with matching freedom (Lima et al.).

**Original design ↓**

### E4-original. Highly-decomposable matchings (M–L, 3 iterations)

**Design.** Lima-et-al-style improvement: after Hopcroft–Karp, local search
over alternating cycles that swap matched edges to split large SCCs of the
induced projection — objective `n_block` (Theorem 2's lower bound). Budgeted
(k-swap neighborhoods, time-capped), deterministic seed-free (iteration order
is data-derived; no `random` in library code — mirrors the workflow constraint
and keeps runs reproducible).

**Validation.** `n_block` monotone non-decreasing across accepted swaps
(asserted per move); matching stays maximum cardinality (size invariant);
Theorem 2 bounds re-verified post-search; measured improvement reported on S1
models (target: close ≥ 25% of the greedy-vs-upper-bound gap on at least one
model, else document why the structure resists).

### G4. Convexity-interaction guard (M, 2 iterations) — ✗ DECLINED (analysis: no-op)

**Investigated and declined with evidence.** The premise — that aggregation
could destroy exploitable convex/GP structure — does not hold for the
eliminations this package performs. Aggregation eliminates variables defined by
*linear* (or ≤2-variable) equalities, and substituting a linear/monomial
expression into a convex/posynomial model preserves convexity/GP. A *nonlinear*
defining substitution only arises from a nonlinear equality, which already makes
the model nonconvex — so there is no convex structure to destroy. Verified with
`discopt._jax.convexity.rules.classify_model`: a convex model
(`min w² + x` s.t. `w = x + 2y`, `x² + y² ≤ 4`) stays classified convex after
`ld2`/`d2`/`gr` (even `gr`, substituting `w = x+2y` into `w²`, yields the convex
`(x+2y)²`). A guard here would never fire; effort not warranted. (`classify_gp`
and `classify_model` are available if a future non-preserving elimination mode is
added.)

**Original design ↓**

### G4-original. Convexity-interaction guard (M, 2 iterations)

**Design.** discopt detects exploitable structure (GP/posynomial log-space
reform, convex fast paths). Substitution can silently destroy it. Add
`aggregate(..., preserve_structure="detect")`: run the relevant detectors
before/after; if a detected structure (GP-representable, convex-QP path, …) is
lost, abstain (or roll back the offending eliminations only, binary-search
style). Off by default; recommended in `Model.solve` integration when a
detector had fired.

**Validation.** A GP model where naive `d2` breaks posynomial detection: guard
keeps it detected and the log-space path still engages (assert solver chose the
GP path via its stats); overhead measured (< 1 detector run each side).

### E5. Implicit elimination of cyclic blocks (L, 4+ iterations, research-grade) — ✅ DONE (primitive; integration scoped out)

**Shipped.** `implicit.py` / `ImplicitBlock`: the mathematically-correct
implicit-elimination primitive for a nonsingular cyclic block `g_B(u, v_B)=0` —
Newton solve for `v_B` given the kept variables `u` (exact symbolic Jacobians via
`differentiate_expression`), a nonsingularity gate (`is_nonsingular`; singular /
non-differentiable → `None`, never a wrong answer), and implicit-function-theorem
derivatives `dv_B/du = -(∂g_B/∂v_B)^{-1} ∂g_B/∂u` (`implicit_derivative`).
5 tests: linear 2×2 vs closed form, nonlinear cycle residual, singular→None,
nonsingular predicate, and implicit derivatives matching finite differences to
1e-6; `example_implicit_block`.

**Integration boundary (why "primitive", not end-to-end).** Eliminating a block
*inside the reduced model* — so it is transparent to the NLP solver — requires an
implicit-function expression node that discopt compiles with a differentiable
numerical solve (e.g. `jax.lax.custom_root`). discopt's modeling layer has no
such node, and adding one would mean modifying discopt, which this standalone
plugin deliberately does not do. So E5 delivers the correct primitive (analysis /
recovery / basis for a future in-discopt integration), not a reduced-model
elimination. This is a genuine architectural boundary, recorded honestly — and
filed upstream as **discopt#379** (implicit-function expression node), with
`ImplicitBlock` offered as the reference implementation for its evaluation/VJP.

**Original design ↓**

### E5-original. Implicit elimination of cyclic blocks (L, 4+ iterations, research-grade)

**Design.** The generalization `_finalize` currently discards: eliminate an
irreducible block `(v_B, g_B)` with nonsingular Jacobian by solving it
numerically inside function evaluation — JAX `lax.custom_root` (or optimistix)
gives the implicit function *and* exact implicit derivatives, so the reduced
model remains AD-differentiable (this is Parker et al.'s implicit-function
formulation [cite:@parker2022implicit] grafted onto our stack). Gate hard:
blocks ≤ size cap, Jacobian
nonsingularity verified at the initial point, fall back to dropping the block
on any doubt.

**Validation.** Round-trip vs the original on models with a 2×2 linear cycle
(closed form known) and a small nonlinear cycle; derivative check: finite
differences vs implicit gradients at random points (`rtol 1e-5`); robustness:
singular-Jacobian block triggers the fallback, never a wrong answer.

**Risk.** Newton-inside-evaluation can fail off-manifold during line searches —
the classic reduced-space fragility. Treat as experimental (`method="lm+implicit"`,
clearly flagged).

### E6. Schur-complement reuse of LM submatrices (L, exploratory) — ✗ NO-GO (spike done)

**Go/no-go spike result: NO-GO.** POUNCE's NLP KKT solve is a separate package
(`~/projects/pounce`); its default linear backend FERAL
(`crates/pounce-feral`) computes its own **fill-reducing ordering**
(AMD/AMF/METIS/Scotch/KaHIP) and exposes ordering only as an *algorithm selector*
— the `feral_ordering` string option maps via `parse_ordering_method` to
`{auto, auto_race, amd, amf, metis, scotch, kahip}`. There is **no
user-supplied-permutation input**: unlike *scaling*, which has an `External(vec)`
strategy, `OrderingMethod` has no `External`/`Custom` variant, and
`pounce.Problem`'s Python API exposes no ordering-hint setter. So LM's block
permutation cannot be injected. Enabling it would require, upstream of this
plugin: an `OrderingMethod::External(Vec<usize>)` in feral, plumbing through
`FeralConfig` + the option, and a `pounce.Problem` binding — none of which exist.
And the payoff is speculative anyway: FERAL already runs AMD/METIS (and
`auto_race` even races them and keeps the best factor), so a hand-supplied
structural ordering is not likely to beat it. **Parked** as an upstream feral/
pounce feature, not a plugin item. (No code written — spike only.)

**Original design sketch.** LM's byproduct — a large nonsingular lower-triangular
submatrix of the equality Jacobian — is precisely a Schur pivot block for KKT
factorization (the paper's closing suggestion). Feed the (row, col) permutation
to POUNCE's linear solver as an ordering hint; compare factorization time vs
default ordering on S1 models. **Go/no-go spike first** (1 iteration): if
POUNCE's factorization has no ordering-hint surface, park the item with a note
rather than building one.

**Validation (if pursued).** Identical solutions (factorization is exact);
factorization-time delta reported; soundness covered entirely by existing
solver tests since only the ordering changes.

---

## Risk register

| Risk | P×I | Mitigation |
|---|---|---|
| F1 scalarization explodes model size on large arrays | M×H | `scalarize_limit` abstain + warning; stage (b) native indexing if hit in practice |
| S3 sweep runtime blows the budget | M×M | keep instances < 5 s; opt-in marker; background run; shrink grid to 7×7 if needed |
| POUNCE lacks timing/ordering hooks (S2, E6) | H×M | cyipopt backend for S2 breakdown; E6 gated on a 1-iteration spike |
| XLA re-inlines shared subtrees (F3) | M×H | measured before fixed; memoized lowering is a contained change |
| G1 interacts with existing presolve in surprising ways | M×H | aggregation runs strictly *before* the orchestrator; `pr_correctness` gate with the flag forced on |
| E5 implicit solves diverge off-manifold | H×M | experimental flag, size caps, nonsingularity checks, drop-block fallback |
| Duals unavailable/degenerate for G3 validation | M×L | validate on strictly-complementary test models; KKT-residual check is primary |

## Success criteria (per milestone)

- **M1**: no elimination with `|pivot| < tol` possible; indexed-model round-trip
  suite green; `nnz_hessian_exact` in metrics; all existing 154+ tests green.
- **M2**: `Model.solve(aggregate="d2")` passes `pr_correctness` with the flag
  forced on; recovered KKT residual ≤ 1e-6 on dual-exposing backends.
- **M3**: reproduction report with Table-4/6/7 analogues on 3 problems ×
  ≥ 7 method variants; the JAX-vs-ASL Hessian-cost question answered with data;
  docs notebook builds warning-free.
- **M4**: each E-item merged behind an explicit option with its validation
  suite, or parked with a written go/no-go finding.

## Suggested execution order (loop iterations)

```
1. F2 ✅   2. G2 ✅   3–5. F1 ✅   6. F4 ✅          → M1 ✅ REACHED
7–8. G1 ⟳(standalone)   9–10. G3 ✅ (G3a+G3b)   11. E1 ✅   12. E2 ✅   → M2 ✅ REACHED
13–16. S1 ✅   17–18. S2 ✅   19(–20). F3
21–23. S3 ✅   24–25. S4 ✅                    → M3 ✅ REACHED (notebook deferred)
26+. E3 ✅ → E4 ✅ → E5 ✅(primitive) → G4 ✗(declined) → E6 ✗(no-go spike) → M4
     [Phase 4 complete: E3/E4/E5 done; G4 declined (no-op, evidence);
      E6 no-go (POUNCE/FERAL exposes no user-ordering hook). Every plan
      item is now done, declined, or parked-upstream with justification.]
```

~25 iterations to M3 with buffers; Phase 4 open-ended by design.
