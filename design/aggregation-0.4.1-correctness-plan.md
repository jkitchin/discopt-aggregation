# discopt-aggregation — 0.4.1 correctness-review fixes plan

**Status:** NOT STARTED. Successor to `aggregation-gaps-and-integer-plan.md`
(complete at 0.4.0). Source: the 2026-07-11 full-repo review (four independent
reviewers over core, numerics, completeness, tests/packaging; every RC finding
below was **confirmed by execution**, and RC1/RC2/RC3 were re-confirmed
independently). The headline: the 0.4.0 gate is green (446 passed, ruff/mypy
clean, hash-seed robust) but five core bugs slipped through it, two of which
return **silently wrong "optimal" results under default settings** — a direct
violation of ground rule "soundness first". This plan verifies, fixes,
validates, and regression-tests each finding, then releases 0.4.1.

**Executor notes (read first — environment pitfalls already hit once).**

- Work items in phase order; within a phase, in listed order. One item per loop
  iteration; commit after each item's validation passes. Never `--no-verify`.
- **Regression-test-first, every RC item:** the *Verify* snippet is the repro.
  Step 1 of every item is to inline it as a test, run it, and watch it fail for
  the documented reason. Only then fix. If a repro does *not* fail as
  documented, stop and re-diagnose before touching src — the finding may have
  drifted.
- **Environment.** `.venv` is Python **3.12** (rebuilt 2026-07-11), with
  discopt installed editable from `/Users/jkitchin/projects/discopt`
  (0.6.0.dev0) and pounce importable. Invoke tools directly:
  `.venv/bin/python -m pytest`, `.venv/bin/ruff`, `.venv/bin/mypy`.
  - **NEVER run `uv run …` in this repo.** It project-syncs: resolves
    `discopt>=0.5` from PyPI against the default interpreter (3.14, whose
    pyo3 build fails) and **empties `.venv` on the failed sync**. This
    happened during the review. Rebuild recipe if it happens again:
    `rm -rf .venv && uv venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install
    -e /Users/jkitchin/projects/discopt -e ".[dev]"`.
- Gates after every item:
  `.venv/bin/ruff check src tests && .venv/bin/ruff format src tests &&
  .venv/bin/mypy src/discopt/aggregation` then
  `PYTHONUNBUFFERED=1 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`.
  Hash-seed gate before finishing a phase: `PYTHONHASHSEED` in `{0, 5, 7}`
  full-suite green.
- `tests/test_packaging.py` pins `len(__all__)` and the version string — update
  both when adding exports or bumping the version. `ruff format` after
  appending tests.
- File:line references below are against commit `4d77816` and may drift;
  locate by the quoted content.

**Ground rules (unchanged).** discopt/pounce/feral untouched (file upstream);
soundness fixes are **default-on** with a CHANGELOG note (they change behavior
only where the old behavior was *wrong*); everything else opt-in/default-off;
tests hash-seed-robust; the skill (`src/discopt/aggregation/skill/`) is updated
in the same item that changes behavior it describes.

Item IDs: `RC*` correctness, `RD*` docs/repo hygiene, `RT*` test hygiene &
coverage, `RX*` release. Effort: S=1, M=2–3, L=4+ iterations.

---

## 0. Phase overview and dependencies

| Phase | Theme | Items | Exit gate |
|---|---|---|---|
| 1 | Silent wrong answers | RC1 RC2 RC3 RC4 | all four repros return answers matching a direct solve (or decline); no existing test regresses |
| 2 | Crash paths | RC5 RC6 | infeasible-integer and coupled-block repros terminate with a correct status, never an exception |
| 3 | Secondary correctness | RC7 RC8 RC9 | docstring claims of duals/diagnostics/metrics all hold on the repros |
| 4 | Docs & repo hygiene | RD1 RD2 RD3 RD4 | skill self-consistent; symlink named right; plan-doc debts written down |
| 5 | Test hygiene & coverage | RT1 RT2 RT3 | `-m unit` is solver-free; named coverage gaps have tests; wheel contents tested |
| 6 | Release | RX1 | 0.4.1 gate |

Dependencies (finish-to-start):

```
RC1 ──> RC4      (same exprwalk collapse logic; RC1's fix determines RC4's shape)
RC5 ──> RC4-test (RC4's repro currently dies in RC5's AssertionError; fixing RC5
                  first turns RC4's observable from "crash" into "wrongly admitted
                  edge" — write RC4's test against admissibility, not the crash)
RD2 depends on RC-phase outcomes only in wording; do after Phase 3.
RT2(a) after RC-phase (guard×integer tests exercise fixed code paths).
```

Suggested order: RC1 RC2 RC3 RC5 RC4 RC6 RC7 RC8 RC9 RD1 RD2 RD3 RD4 RT1 RT2
RT3 RX1.

---

## Phase 1 — Silent wrong answers (soundness, default-on fixes)

### RC1. `SumExpression` pass-through yields per-element affine coefficients (M) — CRITICAL

**Problem.** `exprwalk.affine_coefficients` (`exprwalk.py:329-330`) and
`classify_linearity` (`exprwalk.py:184`) pass a `SumExpression` straight
through to its operand. A scalar variable broadcast against an array inside a
`dm.sum(...)` therefore reports its **per-element** coefficient instead of the
summed one; `incidence.linear_vars` feeds that wrong coefficient into the
substitution and the pipeline returns a wrong optimum, **status `optimal`,
under default `solve(m)` settings**. Scalarize does not save us: it rewrites
indexed uses, not un-indexed broadcast uses.

**Verify (confirmed 2026-07-11; becomes
`tests/test_aggregation_incidence.py::test_sum_broadcast_coefficient_sound`):**

```python
import discopt.modeling as dm
from discopt.aggregation import solve

m = dm.Model("f1")
y = m.continuous("y", lb=-10, ub=10)
x = m.continuous("x", shape=(3,), lb=1, ub=1)
m.subject_to(dm.sum(2 * y + x) == 0)   # true equation: 6y + 3 = 0 -> y = -0.5
m.minimize(y**2)
r = solve(m)
# BUG: returns status "optimal" with y = -1.5 (coefficient read as 2, not 6)
assert r.status == "optimal" and abs(r.x["y"] - (-0.5)) < 1e-6
```

**Fix.** In both `affine_coefficients` and `classify_linearity`, stop passing
`SumExpression` through blind. Sound minimal fix: treat the operand as
**nonlinear/opaque unless it is provably scalar** (shape `()` — mirror the
existing `SumOverExpression` collapse-nonlinear handling). That declines the
elimination rather than mis-eliminating: soundness first, per ground rules.
Do *not* attempt the "exact" fix (multiplying scalar-var coefficients by the
element count) in this item — element counts require shape propagation through
the affine walk, and a half-right version is exactly how this bug happens.
If shape propagation is wanted later, file it as a follow-up item with its own
oracle validation. Grep `exprwalk.py` for **every** `SumExpression` /
reduction-node branch and audit each the same way while in there.

**Validate.**
1. The repro returns `optimal`, `y == -0.5 ± 1e-6` (elimination declined, or
   correct if the surviving path still reduces it) and
   `feasibility_violations` on the recovered point is empty.
2. Sums over *scalar* expressions still eliminate (existing suite must not
   lose eliminations: compare `stats["n_eliminated"]` on the `PROBLEMS`
   registry before/after — write the counts down in the commit message).
3. Oracle harness, SC1-style: for every constraint body in every `PROBLEMS`
   model *and* their `ld1/ecd2/ld2/d2` reduced models, evaluate the affine
   claim numerically — sample 3 random points, check
   `body(p) - (const + Σ coeff·p_v)` ≈ 0 for bodies the walker calls affine
   (tolerance 1e-9). Zero misses required. Keep the harness as a `slow` test.

**Regression tests.** The repro above; a `linear_vars`-level unit test
asserting the coefficient of `y` in `dm.sum(2*y + x) == 0` is either absent
(declined) or `6.0`, never `2.0`; a `classify_linearity` unit test on the same
body; one array-in-sum case with `x` free (not fixed) proving the constraint
is *kept* and the solve matches the direct solve.

### RC2. Binary variables recreated in the reduced model without bounds (S) — CRITICAL

**Problem.** `transform.py:248` recreates surviving binaries as
`reduced.binary(v.name, shape=v.shape)`, dropping `lb`/`ub`, while the
continuous and integer recreation paths carry them. discopt binaries have
mutable bounds (`b.lb = [1.0]` fixes `b` and a direct solve honours it), so an
aggregated solve silently un-fixes the binary and can return an optimum that
is **infeasible in the original model**.

**Verify (confirmed; becomes
`tests/test_aggregation_transform.py::test_binary_bounds_carried_to_reduced`):**

```python
import discopt.modeling as dm
from discopt.aggregation import solve

m = dm.Model("f5")
b = m.binary("b")
b.lb = [1.0]                       # fix b = 1
y = m.continuous("y", lb=-10, ub=10)
m.subject_to(y == b + 1)
m.minimize(y)
r = solve(m)
# BUG: returns optimal b=0, y=1; direct solve gives b=1, y=2
assert abs(r.x["y"] - 2.0) < 1e-6 and round(float(r.x["b"])) == 1
```

**Fix.** Carry `lb`/`ub` in the binary recreation exactly as the
continuous/integer branches do. Then **audit every variable-recreation site**
for the same omission: the rest of `transform.py`, `scalarize.py`'s rebuild,
and `implicit.py`'s reduced-model construction — fix any sibling occurrence in
this same item.

**Validate.** Repro matches the direct solve; a structural unit test asserts
`reduced_model` binary bounds equal the originals for defaulted *and*
tightened binaries; full suite green (binaries with default [0,1] bounds are
unaffected, so nothing should move).

**Regression tests.** The repro; the structural bounds-equality test across
all three variable kinds (continuous/integer/binary) in one parametrized test.

### RC3. Fixed-variable feasibility check reads an interval *endpoint* as "the value" (S) — CRITICAL

**Problem.** `transform.py:292`
(`val = float(np.asarray(interval_evaluate(d, {})[0]))`) takes the **lower
bound of a conservative enclosure** as the fixed variable's value. Any
variable-free defining expression with a non-tight enclosure (`sin`/`cos` →
(−1, 1)) yields garbage: a feasible model is declared **infeasible** (with the
ld1 prepass, i.e. under defaults).

**Verify (confirmed; becomes
`tests/test_aggregation_transform.py::test_fixed_transcendental_value_not_interval_endpoint`):**

```python
import discopt.modeling as dm
from discopt.aggregation import solve

m = dm.Model("f2")
y = m.continuous("y", lb=0, ub=2)
m.subject_to(y == dm.sin(1))       # 0.8414…, feasible
m.minimize(y)
r = solve(m)
# BUG: warns "fixed variable(s) y=-1∉[0,2]" and returns status "infeasible"
assert r.status == "optimal" and abs(r.x["y"] - 0.84147) < 1e-4
```

**Fix.** The sound formulation: flag infeasible **only when the whole
enclosure lies outside the box** — `hi < lb − tol or lo > ub + tol` — never by
comparing one endpoint as if it were the value. Where the expression is
variable-free and exactly evaluable, prefer exact evaluation (tight interval)
so detection stays sharp; where evaluation is unsupported, the wide-interval
test simply (and correctly) doesn't fire. Check what else consumes `val`
downstream (recovery evaluates the defining expression itself, so it should
be unaffected — verify that assumption explicitly).

**Validate.** Repro solves optimal and matches the direct solve; the existing
true-positive stays: `tests/test_aggregation_transform.py::
test_infeasible_fixed_bound_detected` (`y == 5`, `y ∈ [0,1]`) must still
report infeasible; add `y == dm.sin(1)` with `y ∈ [0, 0.5]` — genuinely
infeasible, and exact evaluation should still catch it.

**Regression tests.** The three cases above, parametrized: feasible
transcendental fix, infeasible constant fix, infeasible transcendental fix.

### RC4. `collapse_nonlinear` silently drops variable-free nonlinear constants (M)

**Problem.** `exprwalk.py:280-287`: collapsing a subtree with **no variables**
(e.g. `dm.sin(2)`) yields `(0.0, {}, set())` — the constant 0.909… vanishes
*and* no nonlinear marker survives. Two consequences confirmed: (a)
`_integer_edge_admissible` (`incidence.py:100-121`) sees an all-integral
affine row with constant 0.0 and **wrongly admits** an integer elimination,
violating IA1's documented "constant term is integral" rule (repro:
`i + j + dm.sin(2) == 0` with `integer_aggregation=True` → admitted → dies in
recovery's integrality assertion); (b) potentially wrong *constants* for
continuous eliminations through the same walk — **check this first**, it
upgrades the severity.

**Verify (do BOTH; the continuous probe decides how loudly this fix is
CHANGELOG'd):**

```python
import discopt.modeling as dm
from discopt.aggregation import solve, aggregate

# (b) continuous probe: is the recovered value of y correct?
m = dm.Model("rc4b")
x = m.continuous("x", lb=-5, ub=5)
y = m.continuous("y", lb=-5, ub=5)
m.subject_to(y == x + dm.sin(2))          # y = x + 0.909…
m.minimize((x - 1) ** 2 + y**2)
r = solve(m)                               # compare against m.solve() directly

# (a) integer probe (write the test as "edge not admitted", NOT as a crash
#     assertion — RC5 changes the failure mode):
m2 = dm.Model("rc4a")
i = m2.integer("i", lb=-10, ub=10)
j = m2.integer("j", lb=-10, ub=10)
m2.subject_to(i + j + dm.sin(2) == 0)
m2.minimize(i**2 + j**2)
agg = aggregate(m2, integer_aggregation=True)
assert not agg.eliminated                  # BUG: i is eliminated today
```

**Fix.** A variable-free subtree must contribute its **exact value** as the
constant: evaluate it (`evaluate_expression(node, {})`); on evaluation failure,
return a nonlinear marker (decline), never `(0.0, {}, set())`. Coordinate with
RC1 — same function family, land RC1 first so this diff is small and reviewed
against the already-audited collapse logic.

**Validate.** Both probes: continuous solve matches direct; integer edge not
admitted (or admitted with the true constant and correctly *rejected* by the
integrality rule — either observable is fine, assert through the public
`aggregate` result). IA suite (`tests/test_aggregation_integer.py`, 599
lines) fully green — it is the richest consumer of this walk.

**Regression tests.** The two probes; a direct `collapse_nonlinear`/
`affine_coefficients` unit test asserting `sin(2)`'s value survives as the
constant of `x + sin(2)`.

---

## Phase 2 — Crash paths

### RC5. `solve()` raises `AssertionError` on the IA2 infeasibility it exists to report (S)

**Problem.** `api.py:351` — the infeasible early-return (every variable
eliminated) calls `result.recover_full({})`, and `RecoveryMap.recover`
(`recovery.py:278-285`) asserts integrality of eliminated integers. A
non-integral fixed integer — the *defining case* of IA2's
`infeasible_eliminations` — therefore **crashes** instead of returning
`status="infeasible"`.

**Verify (confirmed; becomes
`tests/test_aggregation_integer.py::test_nonintegral_fix_returns_infeasible_status`):**

```python
import discopt.modeling as dm
from discopt.aggregation import solve

m = dm.Model("rc5")
i = m.integer("i", lb=0, ub=10)
m.subject_to(i == 2.5)
m.minimize(i)
r = solve(m, integer_aggregation=True)     # BUG: AssertionError from recovery
assert r.status == "infeasible"
```

**Fix.** When `infeasible_eliminations` is non-empty, the solve path must not
re-assert what it already knows is violated: either skip recovery on the
infeasible early-return (return `x={}` plus the diagnostic), or teach
`recover` a mode that reports known-infeasible variables without asserting.
Prefer the smaller change at the `api.py` call site; keep the assertion for
the genuinely-unexpected case (a solver returning non-integral values for a
*kept* integer is still a bug worth failing loudly on — see RT2(e) note).

**Validate.** Repro returns `infeasible` with the variable named in the
warning/diagnostics; the existing IA2 unit tests on `infeasible_eliminations`
unchanged; `test_infeasible_fixed_bound_detected` unchanged.

**Regression tests.** The repro; a variant where only a *subset* of variables
is eliminated and one integer fix is non-integral (partial-elimination path).

### RC6. `eliminate_implicit` KeyErrors when cyclic blocks are coupled (M)

**Problem.** `implicit.py:395` — `u_exprs = [newvar_map[name] for name in
kept_inputs]` assumes every input of a block survives, but an input eliminated
in *another* cyclic block is absent from `newvar_map` → `KeyError`. The
documented contract is "a block with an unsupported construct is **left in
place**", not a crash. Same latent defect in `ImplicitElimination.recover`
(`implicit.py:305`): block inputs are read from `kept_values` instead of the
accumulated recovery dict, so chained blocks would also fail at recovery.

**Verify (confirmed; repro preserved from the review — becomes
`tests/test_aggregation_implicit_blocks.py::test_coupled_cyclic_blocks_decline_or_chain`):**

```python
import discopt.modeling as dm
from discopt.aggregation.implicit import eliminate_implicit

m = dm.Model("chain")
x1, x2 = m.continuous("x1"), m.continuous("x2")
y1, y2 = m.continuous("y1"), m.continuous("y2")
z = m.continuous("z", lb=0, ub=10)
m.subject_to(x1 + x2 - z == 0)     # cyclic block A {x1,x2}, input z
m.subject_to(x1 - x2 - 1 == 0)
m.subject_to(y1 + y2 - x1 == 0)    # cyclic block B {y1,y2}, input x1 (eliminated!)
m.subject_to(y1 - y2 + x1 == 0)
m.minimize((z - 3) ** 2 + y1**2)
elim = eliminate_implicit(m)       # BUG: KeyError: 'x1'
```

**Fix.** Conservative, contract-honouring: when a block's inputs include a
variable eliminated by another block, **leave that block in place** (decline
it) — matching the documented behavior for unsupported constructs. Fix
`recover` to read inputs from the accumulated `full` dict regardless (correct
and harmless independent of the decline). Full block-chaining (topological
ordering of block substitutions) is a *feature*, not this fix — park it with a
note in this plan if wanted.

**Validate.** Repro runs without exception; block A eliminated, block B
retained; end-to-end `solve` on the repro model matches the direct solve;
`tests/test_aggregation_implicit*.py` green.

**Regression tests.** The repro (decline path + e2e value check); a
three-block chain variant (A→B→C) proving no ordering assumption sneaks back.

---

## Phase 3 — Secondary correctness

### RC7. Dual recovery declines when aggregation eliminates *every* constraint (S)

**Problem.** `duals.py:300-301` — a constraint-free reduced model gets
`constraint_duals=None` from discopt, and `recover_duals` bails
`available=False, reason="missing_reduced_duals"`. But this case (single
defining equality, interior solution) is squarely in the module docstring's
scope, and the defining multipliers follow from stationarity alone — exactly
what `_solve_defining_duals` computes.

**Verify.** `y - (x+1) == 0`, `min x² + y²`, `lm`; reduced model has zero
constraints; `recover_duals(...).available` is False today. Hand-check the
expected multiplier via KKT (for this model: ∇f + λ∇c = 0 at the optimum).

**Fix.** Treat `constraint_duals is None` **when the reduced model has zero
constraints** as `{}` and proceed. Keep the decline for `None` with surviving
constraints (that is a genuinely missing input).

**Validate.** Recovered multiplier matches the hand KKT value to 1e-6 and the
built-in residual self-check passes; existing decline-reason tests unchanged.

**Regression tests.** The interior case (minimize *and* maximize — DU1's sign
normalization must hold here too); a `None`-with-surviving-constraints case
still declining with `missing_reduced_duals`.

### RC8. `diagnostics.free_variables` over-reports degrees of freedom (S)

**Problem.** `diagnostics.py:49-52` promises "variables with no defining
equality (the model's degrees of freedom)" but returns **all** right nodes of
the DM underdetermined block, matched or not: `x + y == 1` → `['x', 'y']`
(DOF is 1). The DM decomposition itself is correct; the accessor lies.

**Fix.** Return only the *unmatched* variables of the underdetermined block
(the actual DOF), and say so in the docstring. This is a default-on behavior
change to a diagnostic — CHANGELOG note, and check the skill's
`structural_diagnosis` documentation for wording that repeats the old
semantics (update in the same commit, per ground rules).

**Validate/regression tests.** `x + y == 1` → exactly one name;
square well-determined model → `[]`; an over-determined block unaffected;
grep tests/docs for consumers of `free_variables` and reconcile.

### RC9. `nnz_hessian_exact <= nnz_hessian` invariant is false for array variables (M)

**Problem.** `metrics.py:52-56` documents the structural count as a superset
of the exact AD Hessian sparsity ("always satisfies"), but the structural
pattern is counted at variable-*name* granularity (`metrics.py:115,127-137`)
while the exact count is per-scalar: `x` shape `(4,)` in `sum(x**2)` plus
`y**4` → structural 3 < exact 5. Any `compare_methods` row containing a
non-scalarized array participating nonlinearly can violate the documented
relation.

**Fix.** Make the invariant true: expand name-level pairs by element counts —
a name-pair `(u, v)` contributes `numel(u)·numel(v)` potential entries
(diagonal blocks `numel(u)²`, or the tighter elementwise diagonal where the
structure is provably elementwise — only if provable; otherwise the full
block). Overcounting is fine (it is documented as an upper bound); undercounting
is the bug. Scalar-only models must produce byte-identical numbers to today —
that is the no-regression bar (paper Table-4 parity tests).

**Validate/regression tests.** The 4-vector counterexample asserts
`exact <= structural`; all existing metrics/Table-4 tests unchanged; one mixed
scalar+array `compare_methods` smoke test asserting the invariant across all
six methods.

---

## Phase 4 — Docs & repo hygiene

### RD1. Rename the committed `.claude/skills/variable-aggregation 2` symlink (S)

**Problem.** Git tracks `.claude/skills/variable-aggregation 2` — a
Finder/Dropbox duplicate-name artifact — so the repo-local skill registers
under the name "variable-aggregation 2". README's "the two `.claude/skills` /
`skills` entries are symlinks" is broken on this side (`skills/
variable-aggregation` is correct).

**Fix.** `git mv ".claude/skills/variable-aggregation 2"
.claude/skills/variable-aggregation`. Then extend `tests/test_skill.py` with a
repo-layout test: both symlink paths exist under their exact expected names,
both resolve to `src/discopt/aggregation/skill`, and no sibling entry matching
`* [0-9]` exists in either directory (catches Finder-duplicate recurrence —
this repo lives in Dropbox, it *will* try again).

**Validate.** `claude` skill list shows `variable-aggregation` (manual check);
new layout test green.

### RD2. Skill docs contradict the shipped integer-aggregation feature; stale counts (S)

**Problem (all confirmed).** `SKILL.md:79` says integer/binary variables are
"**never** eliminated" twenty lines above the `integer_aggregation` option
row; `references/troubleshooting.md:96-99` ("Never eliminated by any method…
out of scope") sits directly above the section describing the feature.
`SKILL.md:159-160` and `references/study-tools.md:66-74` list 5 `PROBLEMS`
(missing `inventory_chain`); "30 in-tree worked examples"
(`SKILL.md:174`, `study-tools.md:94`) — there are 31.

**Fix.** Rewrite both "never eliminated" passages to: not eliminated **by
default**; opt-in via `integer_aggregation=True` with the IA1 admissibility
conditions, cross-referencing the section below. Add `inventory_chain` to both
problem listings; fix both example counts. Prefer wording that won't go stale
("31" → "every `example_*` in `examples.py`" where exactness isn't needed).

**Regression test.** Add to `tests/test_skill.py`: every key in
`testproblems.PROBLEMS` appears in `study-tools.md`; the string "never
eliminated" (case-insensitive) does not appear in the skill tree.

### RD3. Written-down plan debts: stale parity stats + dropped `tearing_fallbacks` (S)

**Problem.** (a) The 0.3.0 plan promised a dated correction note on
`design/aggregation-parity-plan.md:374,380` ("74 exports", "342 tests" —
actual at 0.4.0: 83 exports, ~447 tests); never added. (b) N2 promised
`tearing_fallbacks: int` on the aggregation stats; the fallback + status check
+ regression test shipped (`tearing.py:78-80`, `strategies.py:313-314`,
`tests/test_aggregation_strategies.py:315`) but the counter did not, and no
doc notes the drop.

**Fix.** (a) Append the dated correction note with current numbers. (b)
Implement the counter — it is small and was promised: increment a
`tearing_fallbacks` entry in `AggregationResult.stats` (`api.py:90`, populated
at `api.py:268`) when `strategies._exact_tear`'s fallback fires; thread the
count out from the strategy layer. Extend the existing monkeypatched fallback
test to assert `stats["tearing_fallbacks"] == 1` and the default path asserts
`0`/absent.

### RD4. `_skill_cli install --force` onto a symlink destination tracebacks (S)

**Problem.** `_skill_cli.py:47,90` — `--force` uses `shutil.rmtree`, which
raises uncaught `OSError` when the destination is a symlink (plausible in this
very repo, where `.claude/skills/<name>` *is* a symlink). `install --user` has
no test at all.

**Fix.** Detect symlink destinations and `unlink` instead of `rmtree`; print
the same "replaced" message. Tests: `--force`-onto-symlink round-trip in
`tmp_path`; `--user` via monkeypatched `Path.home()`.

---

## Phase 5 — Test hygiene & coverage

### RT1. Marker discipline: `unit` must be solver-free (M)

**Problem (confirmed).** File-level `pytest.mark.unit` sits on files running
real pounce solves: `tests/test_aggregation_duals_recursive.py` (7+
`model.solve()` calls), `test_aggregation_guards.py:110-111`,
`test_aggregation_scalarize.py:140-186`,
`test_aggregation_target_linearity.py:86`,
`test_aggregation_transform.py:242`, and in `test_aggregation_examples.py`
the file-level `unit` stacks onto solve-running example tests (`:74,150,176`).
`test_aggregation_integer.py` (MINLP solves) and
`test_aggregation_determinism.py` carry no marker at all;
`test_aggregation_protocol.py:16` unmarked. `pytest -m unit` on a solver-less
machine errors rather than running clean, defeating the marker's purpose
(CONTRIBUTING: "Keep `unit` tests solver-free").

**Fix.** Sweep every test file: solve-running tests get `smoke` (or `slow`),
solver-free tests keep/gain `unit`; remove file-level marks where a file
mixes kinds (mark per-test); mark the three unmarked files. Enforcement so it
sticks: a `conftest.py` collection hook (or a meta-test) that fails any item
carrying **both** `unit` and `smoke`/`slow`, plus — the real gate — CI runs
`pytest -m unit` as a separate step *before* dependencies that provide the
solver are exercised (see RT3 note; locally: temporarily hide pounce via a
`sys.modules` guard fixture if a cheap solver-less check is wanted).

**Validate.** `.venv/bin/python -m pytest -m unit -q` passes and completes in
seconds; `-m smoke` count roughly matches the number of moved tests; full
suite count unchanged (446 + whatever this plan added — update
`test_packaging.py` pins if touched).

### RT2. Close the named coverage gaps (M)

Add, one commit each or batched sensibly:

- **(a) Guard × integer interplay:** `max_fill` and `pivot_tol` each combined
  with `integer_aggregation=True` (zero existing hits) — assert guards still
  veto integer candidates and stats stay coherent. Place in
  `tests/test_aggregation_guard_interplay.py`.
- **(b) MILP tearing *exception* path:** the non-optimal-status fallback is
  tested (`test_aggregation_strategies.py:315`); a **raising** MILP solve is
  not. Monkeypatch `solve` to raise; assert greedy fallback + (post-RD3)
  `tearing_fallbacks == 1`.
- **(c) Dual recovery on a scalarized array model:** all dual tests are
  scalar-only; the base-model-space bookkeeping under `scalarize=True` +
  `recover_duals=True` is untested. Small array model, compare against the
  direct solve's duals.
- **(d) `solve()` infeasible-branch payload:** tests assert
  `status == "infeasible"` only; after RC5, assert the documented shape of
  `r.x` and the diagnostics on both infeasible branches (`api.py:351`
  region).
- **(e) Kept-integer integrality tolerance note:** `recovery.py`'s 1e-6
  assertion evaluates defining expressions at solver-returned kept-integer
  values; a MIP solver with a looser integrality tolerance can trip it on a
  legitimately feasible solve. Decide: round kept-integer inputs before
  evaluation (recommended) or document the tolerance contract. Test with a
  synthetic result dict containing `i = 2.0000004`.

### RT3. Wheel-content test — packaging that `test_packaging.py` doesn't cover (S)

**Problem (confirmed).** Deleting the whole `[tool.setuptools.package-data]`
section passes CI today: `test_skill.py` resolves resources from the editable
source tree, so a wheel missing every skill file goes undetected until a user
runs `discopt-aggregation-skill install`.

**Fix.** A `slow`-marked test that builds a wheel
(`.venv/bin/python -m pip wheel . --no-deps --no-build-isolation -w <tmp>` —
add `build`/nothing extra; pip suffices) and asserts via `zipfile`: `SKILL.md`,
`skill/README.md`, all `references/*.md`, both `examples/*.py` present under
`discopt/aggregation/`, and `plugin.json`'s version equals
`pyproject.toml`'s. Wire the same wheel-build into CI as a step (it is the
only honest test of package-data).

---

## Phase 6 — Release

### RX1. 0.4.1 release gate (S)

1. Re-run every RC repro from this plan one final time from a clean venv.
2. Bump `pyproject.toml` version, `discopt.aggregation.__version__`,
   `.claude-plugin/plugin.json` — and the `test_packaging.py` pins
   (version string *and* `len(__all__)` if exports changed).
3. `CHANGELOG.md` `[0.4.1]`: a **Fixed** section listing RC1–RC9 in user
   language, flagging RC1/RC2/RC3 as "previously returned wrong
   optimal/infeasible results" (these are default-on behavior corrections);
   Changed: RC8 semantics, RD3 `tearing_fallbacks` stat; Docs: RD1/RD2.
4. Full gate: ruff check + format + mypy + pytest, then hash-seed seeds
   `0 5 7`, then `python -m discopt.aggregation.examples` exit 0.
5. Commit; tag if the repo's release convention says so (0.4.0 was a plain
   commit — follow that).

---

## Appendix — review conclusions that need NO action

For the avoidance of re-review: the following were explicitly checked and
found correct — matching/SCC/DM kernel (randomized cross-checks vs brute
force), all six strategies' eligibility + triangularity, pivot_tol relative
test and negative-pivot signs, max_fill accounting, maximize paths, DU1/DU2
sign conventions (hand-verified against KKT), derivative rules (~3k-case fuzz
vs central differences, 0 mismatches), interval enclosures (~60k-sample fuzz,
0 escapes), bound-pruning direction, MILP tear optimality vs brute force on 25
random digraphs, Schur flat-offset indexing (and the integration boundary —
no pounce internals imported), scalarize round-tripping and partial-index
abstention, IA3 grid tightening direction, and `hessian_cost` tree/DAG counts.
`build/` and `egg-info/` are local residue, correctly untracked. The single
skipped test (`test_aggregation_schur.py:110`) is a documented upstream gap
(pounce#180), not a defect.
