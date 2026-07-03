# discopt-aggregation — review-fixes plan (post-audit)

**Status:** NOT STARTED. This plan operationalizes the July 2026 four-track
adversarial review (core pipeline, numerics/extensions, tests/completeness,
literature). Every finding below was **reproduced** during the review; the
reproduction is inlined in each item so it can be turned into a regression test
before the fix is written.

**Executor notes (read first).**
- Work items in phase order; within a phase, in listed order. Do not start an
  item until the previous item's validation passes.
- **Regression-test-first**: for every C*/N* item, first add the inlined repro
  as a failing test (or a test asserting the *current wrong* behavior inverted),
  confirm it fails, then fix, then confirm the whole suite passes.
- The project is not currently a git repository. Run `git init && git add -A &&
  git commit` before touching anything, and commit after each completed item so
  a bad step can be reverted. Never use `--no-verify`.
- File:line references are from the review snapshot and may drift a few lines —
  locate by the quoted content, not the number.
- Line/format/type gate after every item:
  `ruff check src tests && ruff format --check src tests && mypy src`
  and the test gate `python -m pytest tests/ -q`.
- **Hash-seed gate** (mandatory after Phase 2, recommended from the start):
  `for s in 0 1 5 7 9; do PYTHONHASHSEED=$s python -m pytest tests/ -q || echo "SEED $s FAILED"; done`
  (seed 5 currently fails `test_inequality_model_offsets_eq_dual_block` — that
  is item C6).

**Ground rules (carried over, one amendment).**
- discopt / pounce / feral source stays untouched; anything needing core
  support is filed upstream, not vendored.
- New *features* are opt-in/default-off. **Amendment: soundness fixes
  (Phase 1–2) are default-on** — returning `optimal` for an infeasible model is
  a bug, not a behavior a flag should preserve. Note each default-on change in
  the CHANGELOG.
- Every item ships with tests; an item is not done until its validation passes.
- Tests must be hash-seed-robust (assert sets/sorted lists, or accept any valid
  outcome; see C6 for the subprocess pattern to test determinism itself).
- Keep a worked example (`examples.py`) for user-visible behavior changes.

Item IDs: `C*` confirmed correctness bugs, `N*` numerics/fidelity issues,
`T*` test/API hardening, `P*` paper-protocol completeness, `D*` docs/bib,
`R*` research extensions (optional, gated). Effort in loop iterations
(S=1, M=2–3, L=4+).

---

## 0. Phase overview

| Phase | Theme | Items | Exit gate |
|---|---|---|---|
| 1 | Soundness — wrong answers possible today | C1 C2 C3 C4 C5 | no known input yields `optimal` for an infeasible model, dangling refs, or wrong Schur indices |
| 2 | Determinism + dual recovery | C6 C7 | full suite green on PYTHONHASHSEED 0–9; duals never wrong-with-`available=True` |
| 3 | Guards & paper fidelity | N1 N2 N3 N4 N5 N6 N7 N8 | guards mean what their docs say; Theorem-2 stats truthful |
| 4 | API & test hardening | T1 T2 T3 T4 T5 T6 T7 T8 | solve()/aggregate() symmetric; maximize path covered; examples assert |
| 5 | Study completeness vs the paper | P1 P2 P3 P4 | Figs 9–12 producible; protocol-scale sweep runnable; timers answered |
| 6 | Docs & bibliography | D1 D2 D3 | no stale count/claim; bib corrections applied |
| 7 | Research extensions (optional) | R1 R2 R3 R4 | each behind a flag or parked with a written spike result |

Dependencies: C3 before T6 (opaque-node tests reuse C3's walker helper).
C6 before C7 (dual tests must be seed-stable first). P1 before P2.
Everything else is independent within its phase.

---

## Phase 1 — Soundness

### C1. Constant-definition bound feasibility silently discarded (S)

**Problem.** `transform.py:243-247` (in `apply_aggregation`'s box→inequality
conversion): when an eliminated variable's defining expression references no
kept variable (fixed-variable elimination), the bound conversion is skipped
with a comment saying the box "is a feasibility check on a constant" — but the
check is never performed. An infeasible original model becomes feasible.

**Repro (turn into `tests/test_aggregation_transform.py::test_infeasible_fixed_bound_detected`):**
```python
import discopt.modeling as dm
from discopt.aggregation import aggregate, solve
m = dm.Model("t")
y = m.continuous("y", lb=0.0, ub=1.0)
x = m.continuous("x", lb=-10.0, ub=10.0)
m.subject_to(y - 5.0 == 0.0)      # ld1 eliminates y = 5, violating y ∈ [0,1]
m.subject_to(x - y >= 0.0)
m.minimize(x**2)
res = solve(m, method="ld1")
# TODAY: res.status == "optimal", objective 25.0. MUST BECOME: infeasible.
```

**Fix design.**
- In `apply_aggregation`, when the defining expression is constant: evaluate it
  (`recovery.evaluate_expression` with an empty env). If the value lies outside
  `[lb - tol, ub + tol]` with `tol = 1e-9 * max(1, |bound|)` (reuse the pruning
  margin convention from G2), record it on the reduced model:
  `ReducedModel.infeasible_eliminations: list[tuple[str, float, float, float]]`
  (name, value, lb, ub), default empty.
- `aggregate()` propagates the field to `AggregationResult` and emits a
  `UserWarning` naming the variable.
- `solve()` checks the field *before* solving and returns a result with
  `status="infeasible"` (match whatever status string discopt uses for
  infeasibility — inspect `discopt`'s SolveResult statuses; do not invent a new
  one) and no solver call.
- Do **not** raise from `aggregate()` — analyzing an infeasible model is
  legitimate; only `solve()` turns it into a terminal status.

**Validation.** The repro test (status infeasible, warning raised); a feasible
fixed variable (`y ∈ [0,10]`, `y == 5`) still eliminates and solves unchanged;
boundary case `y ∈ [0,5]`, `y == 5` remains feasible (tolerance respected);
existing suite green.

**Files.** `transform.py`, `api.py`, tests, `CHANGELOG.md`.

### C2. Fully-determined solve path skips residual constraints (S)

**Problem.** `api.py:288-298`: when aggregation eliminates every variable,
`solve()` evaluates the objective at the recovered point and returns
`status="optimal"` without checking the now variable-free residual constraints
or the eliminated variables' bounds.

**Repro (`tests/test_aggregation_api.py::test_fully_determined_infeasible_detected`):**
```python
m = dm.Model("t")
a = m.continuous("a", lb=-10, ub=10)
m.subject_to(a - 1.0 == 0.0)
m.subject_to(a - 2.0 == 0.0)      # infeasible with the first
m.minimize(a)
res = solve(m, method="ld1")
# TODAY: status "optimal", objective 1.0. MUST BECOME: infeasible.
```

**Fix design.** In the fully-determined branch: evaluate every reduced-model
constraint body at the recovered full point via `evaluate_expression`; an
equality is violated if `|body| > 1e-8 * max(1, scale)`, an inequality if it
exceeds its sense by the same margin. Also verify the recovered values against
the original variables' bounds (C1 covers the constant-definition subset;
this covers chains, e.g. `a == 1`, `b == a + 100`, `b ∈ [0,1]`). Any violation
→ `status="infeasible"`. Factor the point-feasibility check into a small
helper (`_check_point_feasible(model, point, tol)`) — C4's validation and T4
reuse it.

**Validation.** Repro test; the existing fully-determined happy-path test
(`test_aggregation_scalarize.py`, empty-model guard) still green; a
chained-bound violation case as above.

**Files.** `api.py`, tests.

### C3. Opaque nodes (`CustomCall`, `MatMulExpression`) break analysis in the anti-conservative direction (M)

**Problem.** Three related holes, one root cause: walkers treat unknown node
kinds as *empty leaves*, which for occurrence counting is the unsafe default.
1. `incidence.py:64-67` — `all_vars`/`_collect_var_names` returns `{}` for
   opaque bodies; `incidence.py:198-201` (`var_degree`) skips them. A variable
   used *only inside* a `CustomCall` is invisible, so it can be selected for
   elimination.
2. `transform.py:91-92` — `substitute_expression` passes `CustomCall` /
   `MatMulExpression` through unchanged (the `FunctionCall` branch does NOT
   catch `CustomCall`; it is a distinct class). Result: the reduced model
   retains references to a deleted variable.
3. `scalarize.py:101-127` (`_find_abstentions`) and `:138-166` (`_rewrite`)
   fall through to "leaf" for these nodes: an array used whole via `C @ a` is
   scalarized anyway and the rewritten constraint still references the old
   model's array (verified corruption).

**Repro 1 (dangling reference; `tests/test_aggregation_incidence.py::test_customcall_blocks_elimination`):**
```python
m = dm.Model("t")
x = m.continuous("x"); y = m.continuous("y"); z = m.continuous("z")
m.subject_to(y - (x**2 + 1.0) == 0.0)
m.subject_to(dm.custom_call(lambda a, b: a * b, y, z) - 4.0 == 0.0)  # locate discopt's CustomCall constructor
m.minimize(x**2 + z**2)
agg = aggregate(m, method="lm")
# TODAY: y eliminated; reduced model constraint still references y (walk its leaves).
# MUST BECOME: y is not a candidate (it appears in an unanalyzable constraint).
```
**Repro 2 (scalarize corruption; `tests/test_aggregation_scalarize.py::test_matmul_forces_abstention`):**
a shape-(2,) array `a` fully indexed in one constraint and used as `C @ a` in
another → `scalarize_model` must abstain for `a` (today it scalarizes and
leaves a phantom `a` in the rewritten matmul).

**Fix design.**
- New module `exprwalk.py` with a single child-iteration authority:
  `child_exprs(node) -> list[Expression] | None` — explicit cases for every
  node kind the walkers already know, **plus** `CustomCall` (walk `.args`) and
  `MatMulExpression` (walk its operands; find the attribute names by reading
  discopt's expression classes), and a generic fallback that scans the node's
  attributes for `Expression` instances / sequences thereof. Returns `None`
  only if the kind is unknown AND the generic scan finds no children.
- `incidence._collect_var_names`: use `child_exprs`. On `None` → mark the
  constraint **opaque**. Opaque constraints: (a) never pass any defining-
  constraint filter; (b) every variable name discoverable in them is removed
  from `candidate_variables`; (c) if a truly-unknown node yields no children at
  all, refuse loudly — raise `NotImplementedError` naming the node type (the
  same convention as the complementarities check at `transform.py:228-231`),
  because occurrence of *any* variable can then not be certified.
- `substitute_expression`: handle `CustomCall`/`MatMulExpression` by rebuilding
  the node with substituted children; on an unknown kind with children, rebuild
  generically if the class constructor allows, else raise `NotImplementedError`
  (never pass through silently).
- `scalarize._find_abstentions` / `_rewrite`: route recursion through
  `child_exprs`; unknown/opaque node containing an array reference → abstain
  for that array (conservative direction).
- Sweep the other private walkers onto the same helper so the fix is total:
  `hessian_cost._children`, `guards._var_names`, `implicit._contains_index` /
  `_block_var_names`, `recovery.evaluate_expression` / `interval_evaluate`
  (these two may keep their `NotImplementedError`/`(-inf, inf)` fallbacks —
  they are already in the safe direction).

**Validation.** Both repros; a `CustomCall` whose args contain an eliminable
variable used *elsewhere* eliminably — substitution rebuilds the call and the
solve round-trips; a leaf-walk assertion utility (`assert_no_dangling(model)`)
run over the reduced models of the whole existing round-trip suite; suite green.

**Files.** new `exprwalk.py`, `incidence.py`, `transform.py`, `scalarize.py`,
`hessian_cost.py`, `guards.py`, `implicit.py`, tests.

### C4. `eliminate_implicit` drops eliminated block variables' bounds (S–M)

**Problem.** `implicit.py:368-431`: unlike `apply_aggregation`
(`transform.py:238-266`), the implicit rewrite emits no box→inequality
constraints for the eliminated block variables — their bounds vanish.

**Repro (`tests/test_aggregation_implicit_e2e.py::test_block_bounds_enforced`):**
a 2-variable cyclic block with `v0, v1 ∈ [0, 1]` whose solution at the
unconstrained optimum is `v0=4, v1=2` (reconstruct the review model: kept
variable `u`, block equations coupling `v0, v1` to `u`, objective pulling `u`
to 3). TODAY: reduced model has 0 constraints, solves "optimal", recovery
violates the boxes.

**Fix design.** In `eliminate_implicit`, for each eliminated block variable
with a finite lb/ub: add the inequality `lb <= component_i <= ub` on the
corresponding implicit-node component expression, mirroring
`transform.py`'s conversion including the interval-based pruning
(`prune_bounds` semantics; reuse the same code path — extract the conversion
into a shared helper rather than copying). If the component expression cannot
appear in a constraint (verify against discopt's `Model.implicit` API), then
*abstain*: refuse to eliminate blocks containing bounded variables and record
why on `ImplicitElimination`.

**Validation.** Repro test: either the reduced solve respects the box (bound
constraints emitted) or the block is left in place (abstention) — assert
whichever the API allows, and assert the recovered point is feasible in the
original model via C2's `_check_point_feasible`. Unbounded-block behavior
unchanged (existing 5 e2e tests green).

**Files.** `implicit.py`, `transform.py` (helper extraction), tests.

### C5. `kkt_schur_indices` wrong when any array variable survives scalarization (S)

**Problem.** `schur.py:57-58,70`: `n_x = len(model._variables)` counts a
shape-(3,) array as one, and variable offsets use *position*, not cumulative
scalar offset. Silent in practice only because pounce falls back transparently
on a bad partition.

**Repro (`tests/test_aggregation_schur.py::test_array_variable_offsets`):**
model with variables in declaration order `[a (shape (3,), used only as
sum(a) so scalarize abstains), x, y]`, one constraint making `y` eliminable,
one inequality. TODAY returns `[2, 4]`; the true pounce layout
(5 primal scalars; eq-dual base 5 + n_ineq 1 = 6) is `[4, 6]`.

**Fix design.** Compute per-variable scalar sizes (`int(np.prod(shape))`,
scalars = 1) in declaration order; `n_x = Σ sizes`; an eliminated variable's
x-index = cumulative offset of its name (+ flat element index if it is an
array element — after scalarize, eliminated candidates are scalars, but the
*surviving* arrays shift the offsets). Verify the assumed flattening order
matches the NLP evaluator's (`discopt`'s flat-variable layout — find the
authoritative ordering in the evaluator and cite it in a comment). Also make
the scalarized-model caveat explicit: `kkt_schur_indices` should return
`(indices, model_analyzed)` or take/return the scalarized model so a caller
cannot pair the indices with the wrong model (see review finding (b)-schur).

**Validation.** The hand-computed fixture above; an equality-only and a mixed
model without arrays reproduce the existing expected indices; e2e smoke tests
still green (they assert objective equality; T8 strengthens them).

**Files.** `schur.py`, tests.

---

## Phase 2 — Determinism and dual recovery

### C6. Hash-seed nondeterminism in `maximum_matching` (S)

**Problem.** `matching.py:125,142` iterate adjacency **sets** of variable-name
strings (built from `incidence.constraint_linear_vars`, a set —
`incidence.py:158-160`, consumed at `strategies.py:341`). Which valid pivot
gets matched depends on `PYTHONHASHSEED`. Everything downstream of `lm`
(aggregation choice, Schur block, ordering, implicit block selection, which
duals path runs) is irreproducible across processes. `_greedy` at
`strategies.py:123` already sorts with a "# deterministic" comment — the
matching path never got the same treatment. This, not test pollution, is why
`tests/test_aggregation_schur.py::test_inequality_model_offsets_eq_dual_block`
is flaky: seeds 0–4 pass it *in isolation*, seeds 5–9 fail it *in isolation*
(`[0, 4] != [1, 4]`; seed decides whether `y` (coeff 1.0) or `x` (coeff −2.0)
of `y − (2x+1) == 0` is eliminated — both are valid).

**Fix design.**
- In `maximum_matching` (and any other consumer of adjacency sets in
  `matching.py`, e.g. edge construction feeding `block_triangularize`):
  iterate `sorted(...)`. Sort keys: variable name; constraint id. One
  canonical ordering, applied at iteration sites (cheaper than changing the
  set-valued API).
- Audit `strategies.py`/`incidence.py` for any other set-iteration that feeds
  a *choice* (grep for `for .* in .*vars` over set-typed values); sort those
  too. Reuse the existing `# deterministic` comment convention.
- Fix the over-specified test: after determinism lands, the eliminated pivot
  is stable — pin the now-deterministic expectation, and add a comment that
  the other pivot is equally valid mathematically.

**Validation.**
- New `tests/test_aggregation_determinism.py`: run
  `subprocess.run([sys.executable, "-c", SNIPPET], env={..., "PYTHONHASHSEED": s})`
  for seeds `{0, 5, 7}` where SNIPPET aggregates a 3-model corpus with
  `lm`/`gr`/`d2` and prints the sorted elimination list + `kkt_schur_indices`
  + `block_triangular_ordering` — assert the three outputs are byte-identical
  across seeds.
- Full suite green for `PYTHONHASHSEED in {0..9}` (the phase gate).

**Files.** `matching.py`, possibly `incidence.py`/`strategies.py`, tests.

### C7. Dual recovery: multi-pass algebra wrong; active eliminated bounds undetected (M–L)

**Problem A (wrong duals, `available=True`).** `duals.py:207-229`
(`recover_duals_from_result`) back-substitutes reverse-chronologically across
passes, claiming (docstring `duals.py:167-170`) the combined defining system is
strictly lower triangular. It is not: a pass-0 defining constraint can
reference a variable eliminated in pass 1, and rows can be *mutually* coupled,
so no back-substitution order exists — a linear solve is required. At some
hash seeds the same model instead hits the `coeff == 0` bail (`duals.py:214`)
because a later pass's base-space body no longer isolates the variable it
defines.

**Repro (`tests/test_aggregation_duals_recursive.py::test_cross_pass_coupling`):**
```python
m = dm.Model("t")
a = m.continuous("a"); b = m.continuous("b"); c = m.continuous("c"); d = m.continuous("d")
m.subject_to(b - (2*a + 1) == 0.0)
m.subject_to(c - (b + a) == 0.0)
m.subject_to(d - (c - 2) == 0.0)
m.subject_to(a + d >= 1.0)                      # active at the optimum
m.minimize((a - 0.1)**2 + b**2 + c**2 + d**2)
# recursive d2 → 2 passes. True duals (hand KKT, residual 5e-12):
# μ = (−3.3, 0.7, 5.7, 6.7).
# TODAY: available=True with wrong μ (e.g. −4.95, 4.0…); KKT residual 4.95–19.8.
```

**Problem B (active eliminated bound).** Design doc (parity D1) and docstrings
promise `available=False` when an eliminated variable's bound is active; there
is **no detection anywhere** (`duals.py:107-113, 196-202` skip bound-origin
constraints with a comment). Repro: `y − x² == 0` forcing `y`'s elimination,
`y ≤ 2` active, `min (x−5)²` → returns `available=True`,
`constraint_duals={0: -0.0}`; true multiplier ≈ −2.536.

**Problem C (no self-check).** Docstrings claim "validated by
`kkt_stationarity_residual`", but recovery never calls it.

**Fix design (in order).**
1. **Replace cross-pass back-substitution with a linear solve.** In base-model
   space, unknowns are the defining multipliers `μ_j` (all passes). Assemble
   the square system from stationarity w.r.t. each eliminated variable `v_i`:
   `Σ_j (∂ body_j/∂ v_i) μ_j = −(∂f/∂v_i + Σ_surviving μ_c ∂ body_c/∂ v_i)`,
   all partials via `derivatives.differentiate_expression` evaluated at the
   recovered full point (`evaluate_expression`); any `None` derivative →
   `available=False`. Solve with `numpy.linalg.solve`; singular / ill-scaled
   (`cond > 1e12` via `numpy.linalg.cond` on these small systems) →
   `available=False`. The single-pass case is the triangular special case —
   keep it as a fast path or delete it and always solve (systems are
   `n_elim × n_elim`; on the model sizes here, always-solve is simpler and
   safer; measure only if a benchmark says otherwise).
2. **Active-bound detection.** For each eliminated variable with a finite
   bound: active if `min(|v−lb|, |ub−v|) ≤ 1e-8 * max(1, |v|)`. If active →
   `available=False` with a machine-readable reason field
   (`DualRecovery.reason = "active_eliminated_bound:<name>"`). (Incorporating
   the surviving bound-inequality γ-terms instead of bailing is a follow-on —
   record it as scoped-out, do not attempt in this item.)
3. **Self-check.** After any successful recovery, compute
   `kkt_stationarity_residual`; if `> 1e-6` → downgrade to `available=False`,
   `reason="residual_check_failed"`. This converts every future algebra gap
   from wrong-answer into honest-unavailable.

**Validation.** Repro A now returns the true μ (match hand values to 1e-6 and
residual < 1e-8) **across seeds 0/5/7** (C6 first); repro B returns
`available=False` with the reason; all existing dual tests green (they are the
no-cross-coupling cases — they must keep `available=True` and exact oracle
match); a randomized property test: 20 random chain models × {single-pass,
recursive} — every `available=True` result has residual < 1e-6 (the self-check
makes this a tautology *by construction*, so also assert ≥ half the cases stay
`available=True`, guarding against the fix degenerating to always-unavailable).

**Files.** `duals.py`, tests, `examples.py` (`example_recursive_duals` gains
the coupled-chain case), design-doc scope note.

---

## Phase 3 — Guards & paper fidelity

### N1. `conditioning_estimate` ignores defining-expression coefficients (S–M)

**Problem.** `guards.py:62-85` tracks only `Π 1/|pivot|` with `max` over
referenced chains — defining-expression coefficient magnitudes and
multi-reference accumulation are ignored, so the most common amplification
hazard passes the guard. Repro: `u = x` (pivot 1), `v = 1e6·u` (pivot 1) →
estimate 1.0; true recovery amplification 1e6.

**Fix design.** Amplification of an elimination
`A(v) = (1/|pivot_v|) · Σ_ref |lin_coeff_ref| · A(ref)` over the referenced
*eliminated* variables (A = 1 for kept variables), with nonlinear references
contributing a conservative factor: use `interval_evaluate`-style local
bound of |∂def/∂ref| over the kept boxes when finite, else treat as
`available=False`-style conservatism → estimate `inf` (which `max_condition`
then drops). Keep the old quantity accessible if any test pinned it, but the
documented `conditioning_estimate` becomes the sound bound. Update the
docstring to state exactly what is bounded (worst-case first-order error
amplification of recovery).

**Validation.** The 1e6-coefficient repro is caught (`max_condition=1e3` drops
it); the existing hand-checked `0.01` pivot case still gives 100; a chain
mixing both (pivot 0.1, coeff 10) gives 100; monotonicity: adding a link never
decreases the estimate.

**Files.** `guards.py`, tests, `example_elimination_guards`.

### N2. Exact tearing accepts a non-optimal MILP incumbent as "exact" (S)

**Problem.** `tearing.py:70-73` (`_solve_min_cover`) never checks
`result.status`. A timeout/limit with a feasible incumbent yields a valid but
possibly non-maximum kept set — sound, but it breaks the "exact" contract
silently.

**Fix design.** Check the returned status against discopt's optimal status; on
anything else, fall back to greedy for that block (the existing never-degrade
path) and record `tearing_fallbacks: int` on the aggregation stats. Keep the
existing exception→greedy path.

**Validation.** Monkeypatch the MILP solve to return a suboptimal-status
result → greedy fallback taken and counted; normal path unchanged (existing 7
MILP tests green, still matching brute force).

**Files.** `tearing.py`, `strategies.py` (stats), tests.

### N3. LM per-block greedy consults full incidence; Theorem-2 stats corrupted (M)

**Problem.** `strategies.py:356` runs `_greedy(inc, [cid …])` with each
constraint's **full** `constraint_linear_vars`, not the block's induced
subgraph — a block can "steal" a variable matched in another block; the
duplicate silently collapses via the `right_to_left` dict overwrite in
`block_triangularize` (`matching.py:290`), and neither `n_dropped` nor
`stats={'n_match','n_block','n_agg'}` account for the loss (review repro:
`v0+v1+a==1`, `v0+v1==2`, `a==5` → `a` eliminated via the dense constraint,
the singleton match discarded, `n_agg=2` with `n_block=2`, `n_match=3`, and
the Theorem-2 chain `n_block ≤ n_agg ≤ n_match` reported but not actually
enforced by construction). Result validity is rescued by `_finalize`, but the
algorithm deviates from Algorithm 2 and the bookkeeping lies.

**Fix design.** Pass the block's variable set into the greedy call and filter
`linear_vars` to it (Algorithm 2 step 6(b): greedy **on the block**). Assert
(debug-level) that per-block picks are disjoint. Recompute `n_block` and
`n_agg` from the *final* matching (post-`_finalize`, post-guards) and report
the Theorem-2 chain only when no guard dropped anything; otherwise expose
`n_agg_pre_guards` so the chain refers to the algorithmic quantities the
theorem is about.

**Validation.** The 3-constraint repro now eliminates `a` via `a==5` (matching
Algorithm 2) and stats satisfy `n_block ≤ n_agg ≤ n_match` **by assertion in
the code path**, not just in tests; property test over 50 random models:
chain holds whenever guards are off; existing LM tests green.

**Files.** `strategies.py`, `matching.py` (duplicate-collapse now an error
path), tests.

### N4. Degree filters count only candidate variables (S–M)

**Problem.** `strategies.py:385-396` computes constraint degree over
`inc.constraint_vars` — which `all_vars` restricts to scalar-continuous
*candidates*. Confirmed: `y == x + i` (`i` integer) passes the D2 filter, so
"degree-2" constraints can carry integer/array passengers into targets,
violating both the paper's filter semantics and the README's
structure-preserving claim ("no remaining constraint gains variables").

**Fix design.** Degree = count of **all** variables in the constraint
(candidates or not) — requires `incidence` to retain a per-constraint
all-variables count even for non-candidates (cheap: collect during the same
walk, C3's `child_exprs` makes this uniform). This is a behavioral change to
the *default*; it is a fidelity fix, so default-on with a CHANGELOG entry.
Grep the test corpus for cases relying on the loose semantics before flipping.

**Validation.** `y == x + i` no longer passes d2/ld2/ecd2 filters; a pure
2-continuous-variable constraint still does; structural assertion on a mixed
model: after d2, no constraint's variable count exceeded its original
(the README claim, now actually tested); suite green (fix any tests that
depended on passenger eliminations, with a comment).

**Files.** `incidence.py`, `strategies.py`, tests, `CHANGELOG.md`.

### N5. `ordering.block_triangular_ordering(method=…)` is dead (S)

**Problem.** The `method` parameter ("lm"/"gr") is documented and recorded but
never used — the function always uses the raw maximum matching.

**Fix design.** Smallest honest fix: make the parameter real — dispatch to the
requested strategy's matching (lm = linear matching + block triangularize;
gr = greedy order), or, if that has no consumer, delete the parameter and note
it in the CHANGELOG. Decide by checking pounce/discopt call sites (there are
none in-repo → prefer delete + keep the field on the result as the constant
"lm" it actually is).

**Validation.** Either behavior difference test (lm vs gr orderings differ on
a model where the strategies differ) or removal + green suite.

**Files.** `ordering.py`, tests.

### N6. `differentiate_expression` crashes on array-valued constant exponents (S)

**Problem.** `derivatives.py:95-98` (`_const_value`) does `reshape(())`;
`x ** Constant([2, 3])` raises `ValueError`, violating the module contract
("returns None for unhandled constructs, never crash/wrong").

**Fix design.** `_const_value` returns `None` for any constant whose size ≠ 1;
the power rule already handles a `None` exponent by returning `None`.

**Validation.** `differentiate_expression(x ** dm.Constant([2,3]), x) is None`;
the 7 existing derivative tests green; scalar `Constant([[2]])` (size-1 nested)
still works.

**Files.** `derivatives.py`, tests.

### N7. Infinite-bound sentinel mismatch (S)

**Problem.** `transform.py:46` treats `|bound| >= 1e19` as infinite while
discopt's sentinel is ±9.999e19 — a genuine finite bound in (1e19, 9.999e19)
is silently dropped from the box→inequality conversion.

**Fix design.** Import/mirror discopt's actual sentinel (find the constant in
discopt; if not exported, use `>= 0.99e20` and comment the provenance). Apply
at every bound-infinity test in the plugin (`grep -rn "1e19" src/`).

**Validation.** A bound of `5e19` now produces the inequality; `1e20`-sentinel
bounds still treated as infinite.

**Files.** `transform.py` (+ any other hits), tests.

### N8. Drop-count accounting resets (S)

**Problem.** `strategies.py:215` (`drop_linearity_breaking`) and the
re-finalize at `strategies.py:131-164` reset `n_dropped`, while guard drops
(`guards.py:118-122`) accumulate — the reported drop count under combined
options is wrong.

**Fix design.** Thread the running count through `_finalize` (parameter
`n_dropped_prior=0`) or accumulate on the aggregation object; one convention,
applied everywhere a drop happens.

**Validation.** A model where pivot-drop + linearity-drop + guard-drop each
remove one elimination reports `n_dropped == 3`.

**Files.** `strategies.py`, `guards.py`, tests.

---

## Phase 4 — API & test hardening

### T1. `solve()` missing `tearing=` / `decomposable=` passthrough (S)

`api.py:248-260` vs `aggregate` at `api.py:103-116`. Add both parameters,
forward verbatim, mirror the docstrings. Test: `solve(m, method="lm",
tearing="exact")` on a cyclic model returns the same eliminations as
`aggregate(..., tearing="exact")` + manual solve; a keyword-signature test
asserting `set(sig(solve)) ⊇ set(sig(aggregate)) − {model-only params}` so the
next asymmetry cannot ship silently.

### T2. `compare_methods` rejects `d<k>` (S)

`metrics.py:164-166` validates against the `STRATEGIES` dict; route through
`strategies.resolve_strategy` instead. Test: `compare_methods(m,
methods=("d2","d3"))` returns both rows; unknown method still raises.

### T3. `compare_methods` mixed scalarization granularity (S–M)

`metrics.py:169-174`: the "original" row is computed on the un-scalarized
model while method rows scalarize internally — NNZ/Hessian columns are not
comparable on array models (`benchmark.py:72-78` already fixes this; metrics
doesn't). Fix: scalarize the baseline model once (respecting the same
`scalarize=` option) before computing the original row; document the
granularity in the docstring. Test: on an array model, original-row
`n_variables` equals the scalarized count and matches `benchmark_model`'s
baseline row.

### T4. Maximize-objective aggregate→solve coverage (S–M)

Zero tests exercise maximize through any rebuild path (`transform.py:275`,
`scalarize.py:259`, `implicit.py:423`). Add three round-trip tests
(one per path): maximize model, aggregate/scalarize/implicit-eliminate, solve,
assert objective and recovered point match the direct solve of the original
(and are feasible via C2's helper). If any path is broken, fix the sign in the
same item.

### T5. Example tests must assert (M)

24 of 30 tests in `test_aggregation_examples.py` are assertion-free "must not
raise". Refactor `examples.py` so each `example_*` returns a small dict of its
headline quantities (residuals, counts, statuses) *and* prints as before; the
tests then assert the meaningful ones: `example_dual_recovery` KKT residual
< 1e-6, `example_bound_pruning` prune count > 0, `example_exact_tearing`
kept ≥ greedy, `example_reliability` convergence fractions ∈ [0,1] and
original ≤ best-method, etc. Keep "runs clean" as one umbrella test. Update
CONTRIBUTING's convention note (it currently institutionalizes the
assertion-free pattern).

### T6. Guard-interplay and edge-path tests (S–M)

Add: `max_fill` with `lm` and with `recursive=True`; `max_fill` +
`max_def_nodes` + `max_condition` combined (drops accumulate — pairs with N8);
`tearing="exact"` + guards; integer/binary *array* variables through
`scalarize` (rebuilt un-scalarized, model still solves); `balanced_sum([])`
raises `ValueError`; `exact_hessian_nnz` returns `None` when the evaluator
can't build (monkeypatch); objective-less model through
`model_structure`/`solve`; `sweep._solve_instance` fully-determined branch.

### T7. Dead `slow`-marker config (S)

`pyproject.toml:49` deselects `-m 'not slow'` but no test carries the marker —
either register the marker (`[tool.pytest.ini_options] markers`) and apply it
to the protocol-scale sweep (P2), or drop the deselect. Prefer: register + use
in P2.

### T8. Schur e2e cannot detect a wrong block (S–M)

`test_aggregation_schur_e2e.py:39-52,59-83` assert only objective equality;
pounce's transparent fallback makes a wrong partition invisible. Fix in two
layers: (a) hand-checked index fixtures (C5 adds one; add an equality-only and
a mixed fixture with every offset annotated); (b) probe pounce's result/stats
for any field indicating the Schur path was taken (inspect
`solve_nlp_from_model`'s result object; the parity plan says timing hooks
landed via pounce#180). If a stats field exists → assert it; if not → add a
skip-marked test documenting the gap and file the stats request upstream
(house pattern: parked with a written note).

---

## Phase 5 — Study completeness vs the paper

### P1. Sweep must retain per-cell results; virtual best (M)

**Problem.** `sweep.py:67-86` discards per-cell statuses inside the loop —
the paper's Figs 9–11 (per-cell convergence grids) and Fig 12 (virtual best)
are unproducible; "virtual" appears nowhere in src/tests.

**Design.** `SweepResult` gains `cells: list[CellResult]` with
`(params: dict, status: str, iterations: int | None, wall: float)`. Add
`virtual_best(results: Mapping[str, SweepResult]) -> SweepResult` (per-cell
best across methods — the paper's Fig 12 quantity) and
`sweep_to_csv(results, path)` for artifacts. Keep the aggregate fields
backward-compatible.

**Validation.** 3×3 grid: `len(cells) == 9`, aggregate counts equal the
recount over cells; virtual-best convergence ≥ every single method (assert on
a crafted case where methods fail on disjoint cells); CSV round-trips.

**Files.** `sweep.py`, tests, `example_reliability` (now prints the grid),
`design/aggregation-stage2-results.md` (note the capability).

### P2. Protocol-scale reliability run (M, mostly compute)

**Design.** An opt-in, `slow`-marked test-or-script (`benchmark.py` gains
`run_protocol_sweep(problem, points=11, methods=ALL)`) that runs the paper's
11×11 × {original + 6 methods} on `gas_pipeline` and `recycle_loop` (the
non-convex problems), writing CSV + the Fig-9-style grid per method under a
`results/` directory (git-ignored). Include `gr`/`lm` — with `max_def_nodes`
set to the D2-guard default so the known Hessian blow-up doesn't hang the run
(document that choice; it is itself a finding). Report the Table-7 analogue
and virtual best via P1.

**Validation.** The `slow` suite completes under a stated budget (< 1 h
locally); determinism spot-check (rerun 5 random cells, same statuses — relies
on C6); "no method converges fewer than original" *reported*, not asserted.

**Files.** `benchmark.py` or new `protocol.py`, tests (marker), `.gitignore`.

### P3. Per-callback solve-time breakdown (M, spike-gated)

**Problem.** The paper's Table 6 %-time-in-func/jac/hess has no analogue; the
original S2 design promised evaluator-callback timers and was quietly rescoped.

**Design (spike first, 1 iteration).** Determine whether discopt exposes a
cyipopt (or any callback-boundary) NLP path where the plugin can wrap
objective/gradient/jacobian/hessian callables with timers *without touching
discopt source*. If yes → `benchmark.py` gains
`callback_breakdown(model, method) -> dict[str, float]` and the S2 story is
completed with a Table-6 analogue on one problem. If no → park with a written
note (house pattern) and add the timing request to the existing pounce#180
thread. Do not vendor a solver.

**Validation (if built).** Fractions sum to ≤ 100%; overhead < 2% (no-op
timer run); the gr-reduced reaction-diffusion shows Hessian fraction ≫
original (the R2 finding, now in Table-6 form).

### P4. A discretized-DAE test problem (M–L)

**Problem.** The paper's problem class (collocation DAE: distillation,
moving bed) has no counterpart; the S1 "DONE" swapped it for a
finite-difference QP. This is the biggest remaining completeness gap.

**Design.** One `TestProblem` built with discopt's DAE/collocation tooling
(per the original S1 design item 1: N-stage cascade with bilinear + rational
terms, two sweep parameters), sized so nominal local solve < 5 s. Register in
`PROBLEMS`; verify every structure-preserving method round-trips; run it
through P1's sweep at 3×3 in tests and include it in P2's protocol run.

**Validation.** Nominal solve optimal + objective regression-pinned;
`ld1/d2` round-trip objective match + full-space feasibility; structural
fingerprint pinned; `d2` eliminates a meaningful fraction (assert > 10% of
variables — the paper's DIST shows 36%).

**Files.** `testproblems.py`, tests, `examples.py`.

---

## Phase 6 — Docs & bibliography

### D1. Stale counts and layout claims (S)

- `README.md:85` "200 tests" and `:96` "300+ tests" → replace both with a
  non-numeric phrase ("the full suite") or a single number updated by the
  packaging test. Prefer non-numeric; numbers rot.
- `README.md:92-95` layout: add `schur` (and reconcile with
  `CONTRIBUTING.md:76`).
- `design/aggregation-parity-plan.md:374,380`: "342 tests", "74 exports" —
  append a dated correction note rather than rewriting history (the plans read
  as logs).
- Add to `tests/test_packaging.py`: assert `len(__all__)` equals a pinned
  constant that lives next to `__all__` (drift becomes a loud failure), and
  assert README contains no regex-matching stale `\d+ tests` claim if the
  non-numeric wording is chosen.

### D2. Bibliography corrections (S)

In `design/references.bib` (all verified against Crossref/arXiv July 2026):
- `regev2022hykkt`: `year = {2023}` (OMS issue year; online 2022 — keep a
  `note` if desired).
- `zhang2024aggregation`: authors `Zhang, Houshan and Yuan, Jianhua`.
- Key/`year` mismatches `pacaud2024condensed` (year=2022) and
  `gleixner2024papilo` (year=2022): pick one convention (rename keys to
  `pacaud2022condensed`/`gleixner2022papilo` and update the two design docs'
  cite keys — grep for the old keys).
- Tearing wording: the docs cite Baharev et al. (minimum feedback **arc** set)
  for `tearing.py`'s minimum feedback **vertex** set MILP. In
  `aggregation-parity-plan.md` S2 and the `tearing.py` docstring, say
  "adapting the lazy cycle-enumeration MFAS approach of Baharev et al. to the
  vertex variant".

### D3. Bibliography additions (S–M)

Add (verified in the review):
- Bongartz & Mitsos, "Deterministic global optimization of process flowsheets
  in a reduced space using McCormick relaxations", JOGO 69:761–796 (2017),
  DOI 10.1007/s10898-017-0547-4 — the major uncited neighbor (manual
  reduced-space elimination at scale; MAiNGO).
- Bestuzheva et al., "Global optimization of MINLPs with SCIP 8", JOGO (2023),
  DOI 10.1007/s10898-023-01345-1 — the MINLP-presolve anchor.
- Shin, Anitescu, Pacaud, JOTA (2023), DOI 10.1007/s10957-022-02129-5 and
  Pacaud et al., arXiv:2405.14236 — the current GPU condensed-KKT references
  (supersede the lone 2022 preprint).
- Parker et al., arXiv:2509.22462 (2025) — companion to
  `parker2026blocktriangular`.
- Optionally: PACE 2022 DFVS report (DOI 10.4230/LIPIcs.IPEC.2022.26) next to
  the tearing wording fix; Hoen et al. CPAIOR 2024 (presolve certification,
  motivates R4); Fourer & Gay 1994 (AMPL presolve); Puranik & Sahinidis 2017
  (bound-tightening survey, motivates R2).
Cite the new entries from the relevant design-doc sections (README's
References paragraph mentions the collection; keep it one line).

---

## Phase 7 — Research extensions (optional; each behind a flag or a spike)

Ranked by tractability × fit. Each follows the house pattern: spike →
go/no-go note → if go, opt-in flag + tests + worked example.

### R1. Fill-aware (Markowitz-style) elimination ordering (M)

**Idea.** Every current strategy maximizes elimination *count*; none minimizes
substitution fan-out — the measured R2/S2 Hessian pathology. Add a selection
score per candidate elimination:
`fill(e) = (n_vars(def_expr) − 1) · n_target_constraints(v)` (the F2 estimator
already computes this) and, for `gr`/`lm`, process candidates in ascending
fill order under a budget `total_fill_budget` (opt-in,
`aggregate(..., order="min_fill")`). Validation: on `reaction_diffusion`
n=8, `gr` with `order="min_fill"` + budget produces a reduced model whose
`model_expression_size` and local-solve wall time sit strictly between plain
`d2` and plain `gr` (the R2 numbers give the brackets); triangularity fuzz
suite green. Cite Achterberg et al. 2020 (fill guards in MIP aggregation).

### R2. Bound-tightening interplay: FBBT for the reduced model (M)

**Idea.** Interval-propagate through defining expressions to (a) derive
tighter bounds for kept variables in the reduced model and (b) *prove more
eliminated-variable boxes redundant* than the current per-bound check —
unlocking pruning the F2/G2 guards currently leave on the table, and directly
strengthening C1's feasibility check. Build on `interval_evaluate` (already
sound per the 883k-point fuzz). Opt-in `aggregate(..., fbbt=True)`.
Validation: soundness property (sampled feasible points of the original remain
feasible, same harness as G2); a case where a redundant box is only provable
after one round of propagation; cite Puranik & Sahinidis 2017, SCIP 8.

### R3. Native indexed incidence (L)

The measured quadratic build bottleneck (R3 profiling: scalarize+aggregate
0.045→1.272 s over 1k→8k vars) — stage (b) deferred from F1: treat
`(name, index)` pairs as incidence nodes and substitute at `IndexExpression`
granularity, avoiding full expansion, with single-pass substitution. Go/no-go
spike: prototype incidence-only (no transform) and measure; go only if the
kernel stays ~linear where the current path is quadratic. This is the largest
item in the plan; do not start it before Phases 1–4 are done.

### R4. Parked (write one-paragraph spike notes only, no code)

- **Aggregation × GPU condensed-KKT** — needs a GPU condensed-IPM path in the
  solver stack; measure aggregation's density penalty under condensation
  (cites: arXiv:2405.14236, Shin et al. EPSR 2024). Park until the stack has
  a condensed path.
- **Reduced-space export for global optimization** — `discopt.Model.solve()`
  *is* a global solver: a 1-iteration spike measuring spatial-B&B node counts
  original-vs-d2 on `gas_pipeline` decides go/no-go (cite Bongartz & Mitsos
  2017).
- **Learned aggregation selection** — the benchmark harness is the data
  generator; needs P2's protocol runs first for a training set (cites:
  L2P-MIP ICLR 2024, arXiv:2603.19417).
- **Certified aggregation** — machine-checkable substitution certificates
  extending `verify_composition` (cite Hoen et al. CPAIOR 2024).
- **Symmetry-aware / coupling-width-aware elimination** (cites: van Doornmalen
  & Hojny 2024; arXiv:2603.19417).

---

## Phase 7 — execution notes (what shipped)

### R1 — fill-aware (min-fill) elimination ordering — ✅ DONE
Shipped as `aggregate(..., order="min_fill")` (default `"index"`, a no-op).
Threaded through `build_incidence`/`IncidenceModel.order` and `_greedy`: under
`min_fill`, constraints are visited — and the variable within each is chosen —
in ascending estimated-fill order (`IncidenceModel.fill_estimate`, the F2
estimator), so the greedy prefers low-fan-out substitutions. Measured on
`reaction_diffusion(8)` with `gr`: the objective expression tree drops from
16871 to ~400 nodes (~40×) for the *same* 9 eliminations, and the reduced model
round-trips to the same optimum — directly countering the R2/S2 substitution
blow-up. Test: `test_fill_aware_order_shrinks_expression_without_losing_eliminations`.

### R2 — FBBT for the reduced model — ⏸ PARKED (spike note)
Interval-propagate through defining expressions to (a) tighten kept-variable
bounds in the reduced model and (b) prove more eliminated boxes redundant than
the current per-bound G2 check, unlocking pruning the pivot/bound guards leave
on the table (and strengthening C1's feasibility check). The primitive
(`recovery.interval_evaluate`, sound over 883k fuzz points) already exists, so
this is an `aggregate(..., fbbt=True)` post-pass, not new math. Go criterion: a
model where a box is redundant only after one propagation round, plus the G2
soundness property (sampled feasible points stay feasible). M-effort; parked
because the shipped guards already cover the common cases and no bundled model
needs the extra strength yet. Cites Puranik & Sahinidis 2017; SCIP 8.

### R3 — native indexed incidence + DAG coefficient extraction — ⏸ PARKED (spike note, go)
The measured bottleneck is twofold and now well understood:
(1) **build-time O(tree)**: `incidence.linear_vars` calls discopt's
`_isolated_affine_coeff` once per candidate per constraint, and that walker is
`O(tree)` — a single call on a substitution-expanded reduced constraint took
**28.7 s** (a 2.5M-node tree over a 5k-node DAG). Recursive `gr`/`lm` on a real
DAE therefore hangs *at aggregate time*, before any solve, which no `max_def_nodes`
guard prevents (the guard runs post-selection). This is the same root cause as
the metrics hang already fixed in Phase 2's follow-on (which replaced per-name
`_isolated_affine_coeff` with the DAG-memoised `exprwalk.classify_linearity`).
**Go decision:** extend `exprwalk` with a DAG-memoised *coefficient* extractor
(`affine_coefficients(body) -> {name: float | None}`, one pass, `None` for
nonlinear) and route `incidence.linear_vars` through it, making incidence
`O(distinct nodes)`. This alone makes `gr`/`lm` tractable on the DAE problems and
unblocks the full P2 protocol run with the approximate-maximum methods.
(2) **native `(name, index)` incidence** (F1 stage b): substitute at
`IndexExpression` granularity instead of full scalar expansion — the ~quadratic
scalarize+aggregate cost the R3 profiling measured. Larger; do (1) first (it is
the load-bearing half and reuses the Phase-2 pattern). L-effort overall; parked
as the single most valuable follow-on for *scaling* the study to the paper's
problem sizes.

### R4 — parked research directions (one-line spikes)
- **Aggregation × GPU condensed-KKT**: measure aggregation's density penalty
  under condensation (Cholesky/cuDSS), where condensation already densifies — the
  penalty may be nearly free while the system shrinks. Needs a GPU condensed path
  in the solver stack. Cites arXiv:2405.14236, Shin et al. EPSR 2024.
- **Reduced-space export for global optimization**: `discopt.Model.solve()` *is*
  a global solver; a 1-iteration spike measuring spatial-B&B node counts
  original-vs-`d2` on `gas_pipeline` decides go/no-go. Cites Bongartz & Mitsos 2017.
- **Learned aggregation selection**: the P2 protocol harness is the training-data
  generator; needs its runs first. Cites L2P-MIP (ICLR 2024), arXiv:2603.19417.
- **Certified aggregation**: machine-checkable substitution certificates
  extending `verify_composition`. Cites Hoen et al. CPAIOR 2024.
- **Symmetry-/coupling-width-aware elimination**: orbit-consistent choices; a
  coupling-width objective before ADMM/Schur decomposition. Cites van Doornmalen
  & Hojny 2024; arXiv:2603.19417.

---

## Final gate (release 0.3.0)

1. Full suite green under `PYTHONHASHSEED ∈ {0..9}`; ruff + format + mypy
   clean; `python -m discopt.aggregation.examples` exits 0.
2. Every C*/N* item has a regression test that fails on the pre-fix code
   (spot-check two by `git stash`-ing the fix).
3. `CHANGELOG.md` `[0.3.0]`: soundness fixes listed as **Fixed** (default-on),
   behavior changes (N4 degree semantics) called out, new options as **Added**
   (opt-in). Version bumped in `pyproject.toml` + `__init__.__version__`
   (consistency test already exists).
4. Design docs: dated correction notes for the stale claims (D1), scope notes
   for C7 (γ-terms) and P3 (if parked).
5. Re-run the review's acceptance repros end to end (they are inlined above)
   — all must show the fixed behavior.
