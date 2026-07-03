---
name: variable-aggregation
description: Use when reducing a discopt nonlinear optimization model with variable aggregation (reduced-space presolve) — substituting variables defined by equality constraints to shrink the model, improve interior-point convergence reliability, or reproduce Naik et al. (arXiv:2502.13869). Covers method choice (ld1/ecd2/ld2/d2/gr/lm), the structure-preserving-vs-Hessian-cost tradeoff, numerical guards, primal/dual recovery, implicit elimination of cyclic blocks, structural diagnostics, and the study/benchmark harnesses. Triggers on the discopt.aggregation package, "aggregate variables", "reduced-space formulation", "presolve substitution", or eliminating intermediates from an NLP/DAE model.
---

# Variable aggregation (discopt.aggregation)

## Overview

**Variable aggregation** substitutes a variable that is *defined by an equality
constraint* into the rest of a model, eliminating both a variable and a
constraint to produce a smaller **reduced-space** formulation, then recovers the
eliminated variables from the reduced solution. It is a **presolve reduction**
for nonlinear programs, implementing Naik, Biegler, Bent & Parker, *Variable
aggregation for nonlinear optimization problems* (arXiv:2502.13869), as the
`discopt.aggregation` plugin.

```python
import discopt.modeling as dm
from discopt.aggregation import aggregate, solve

m = dm.Model("ex")
x = m.continuous("x", lb=-5, ub=5)
y = m.continuous("y", lb=-5, ub=5)
m.subject_to(y - (x + 1) == 0.0)      # y is DEFINED by x
m.minimize(x**2 + y**2)

res = solve(m, method="d2")            # aggregate -> solve -> recover
print(res.status, res.objective)       # optimal 0.5
print(res.x)                           # {'x': -0.5, 'y': 0.5}  (y recovered)
```

**Core value:** a smaller, often better-conditioned problem that interior-point
solvers converge on more reliably — *if* you pick a method that doesn't inflate
the Hessian. The whole skill is about that "if".

## The one thing to get right: structure-preserving vs approximate-maximum

This is the central tradeoff and the source of every pitfall.

| Family | Methods | Eliminates | Effect | Use when |
|---|---|---|---|---|
| **Structure-preserving** | `ld1`, `ecd2`, `ld2`, `d2` | ≤ ~60% (only ≤2-variable defining eqs) | keeps per-constraint density ~flat; per-iteration cost stays low | **default choice**; `d2` is the recommended balance |
| **Approximate-maximum** | `gr`, `lm` | 70–90% (any linearly-defined var) | substitutes multi-variable / nonlinear expressions everywhere → **denser, more nonlinear** constraints; **Hessian evaluation can become the bottleneck** | you want maximum size reduction AND have checked the reduced model isn't expression-blown-up (use `order="min_fill"`) |

**Rule of thumb:** start with `method="d2"` (the paper's recommendation and the
default). Only reach for `gr`/`lm` when you specifically need aggressive
elimination, and then pass `order="min_fill"` and/or `max_def_nodes=` to bound
the substitution blow-up. See `references/methods.md` and
`references/troubleshooting.md`.

## Two entry points

- `aggregate(model, method="d2", ...) -> AggregationResult` — build the reduced
  model + recovery map. Inspect `.reduced_model`, `.eliminated`, `.n_passes`,
  `.recover_full(kept_values)`.
- `solve(model, method="d2", ...) -> AggregatedSolveResult` — aggregate, solve
  the reduced model, recover the full-space solution in one call. Returns
  `.status`, `.objective`, `.x` (full-space, original shapes), `.duals`
  (if `recover_duals=True`).

Both accept the same aggregation options (below). `solve` forwards extra kwargs
to the reduced model's `.solve()`.

## Method selection (quick)

```
ld1   fixed variables  y = a                        (most conservative)
ecd2  equal-coeff 2-var y = x + a                    structure-preserving
ld2   linear 2-var     y = a*x + b                   structure-preserving
d2    degree-2         y = f(x)   (≤2 vars)          DEFAULT, recommended
d<k>  degree-k         y = f(x1..xk) (≤k vars)       d3, d4, ... (intermediate)
gr    greedy           y = f(w,x,…) (y linear)       approximate-maximum
lm    linear-matching  y = f(w,x,…) (y linear)       approximate-maximum
```
`ld1`/`ecd2`/`ld2`/`d2`/`d<k>` run fixed-variable elimination first, then the
degree-bounded pass, recursively to a fixed point. `gr`/`lm` are applied once.
Integer/binary variables are **never** eliminated. Full decision guidance:
`references/methods.md`.

## Options reference (when to use each)

All are keyword args to `aggregate`/`solve`. Defaults are safe no-ops unless noted.

| Option | Default | Use when |
|---|---|---|
| `method` | `"d2"` | pick the family (above) |
| `recursive` | `True` | `False` = exactly one pass of the named strategy |
| `pivot_tol` | `1e-6` | relative pivot guard; `0.0` = exact paper behavior |
| `prune_bounds` | `True` | omit box→inequality constraints proven redundant by interval propagation (keeps bounded-variable elimination a net win) |
| `scalarize` | `True` | rewrite array variables to scalars so aggregation reaches discretized/DAE models (no-op on scalar models) |
| `max_fill` | `None` | cap per-elimination Jacobian fill; drops high-fan-out eliminations |
| `max_def_nodes` | `None` | cap the substituted expression's node count — **use with `gr`/`lm` to bound Hessian blow-up** |
| `max_condition` | `None` | cap worst-case recovery-error amplification (drops ill-conditioned chains a large coefficient or tiny pivot would create) |
| `order` | `"index"` | `"min_fill"` = fill-aware greedy; **shrinks `gr`/`lm` expression graphs ~40× for the same eliminations** |
| `tearing` | `"greedy"` | `"exact"` = maximum-acyclic-subset tear of cyclic blocks (eliminates ≥ greedy per block) |
| `decomposable` | `False` | local-search the `lm` matching for a more decomposable (higher Theorem-2 bound) structure |
| `preserve_target_linearity` | `False` | drop an elimination that would turn a currently-linear constraint nonlinear |
| `recover_duals` | `False` | (`solve` only) also recover full-space Lagrange multipliers |

Deep dive with recipes: `references/options.md`.

## Safety and correctness (built in)

- **Infeasibility is surfaced, not hidden.** If a fixed-variable elimination's
  value violates its bounds, or a fully-determined system is inconsistent,
  `solve()` returns `status="infeasible"` (with a warning) instead of a bogus
  `optimal`.
- **No dangling references.** Variables inside opaque nodes (`CustomCall`,
  matrix multiply) are correctly tracked and substituted; genuinely unanalyzable
  nodes raise rather than silently corrupt the model.
- **Deterministic.** Results are reproducible across `PYTHONHASHSEED` (sorted
  graph iteration), so a run is repeatable.
- **Guards are sound.** `max_fill`, `max_def_nodes`, `max_condition` only ever
  *drop* eliminations; the remaining set stays validly lower-triangular.

## When aggregation helps — and when it hurts

**Helps:** convergence reliability (the paper's headline — more parameter
instances converge), and solve time when the reduced KKT system factorizes
faster. Best on models with many linearly-defined intermediates (DAE
discretizations, flowsheets, recycle loops).

**Hurts:** `gr`/`lm` on a model where the eliminated intermediates feed a
nonlinear objective/constraint — substitution compounds the expression graph
(worst case degree-`2^n`), and Hessian evaluation dominates. A **chain** of
2-variable-eliminable constraints is the classic trap. Mitigations, in order:
`order="min_fill"` → `max_def_nodes=` → use `d2` instead → keep the block
implicit (see below).

## Beyond primal recovery

- **Dual recovery** — `solve(..., recover_duals=True).duals` gives full-space
  Lagrange multipliers (surviving + eliminated defining equalities), validated
  by a KKT stationarity self-check. Reports `available=False` with a `reason`
  for out-of-scope cases (maximize objective, active eliminated-variable bound)
  rather than a wrong answer. See `references/recovery-and-implicit.md`.
- **Implicit elimination** — `eliminate_implicit(model)` eliminates *cyclic*
  (irreducible) blocks that explicit substitution can't, via a differentiable
  inner solve (`Model.implicit`), keeping the reduced model AD-differentiable.
  This is the reduced-space move for recycle loops / index-1 DAEs.
- **Structural diagnostics** — `structural_diagnosis(model)` runs a
  Dulmage–Mendelsohn decomposition to flag over/under-determined subsystems
  *before* eliminating (the square block is exactly what aggregation targets).
- **Schur/ordering** — `kkt_schur_indices(model)` and
  `block_triangular_ordering(model)` emit the reducible block / permutation for a
  structured-KKT (Schur-complement) linear solve.

## Study & reproduction harnesses (paper Tables 4/6/7)

- `compare_methods(model)` → Table-4 structural comparison (vars, cons, elim,
  NNZ/con, Hessian coupling) across all methods.
- `benchmark_model(model)` / `callback_breakdown(model, method)` → runtime and
  per-callback (func/grad/jac/hess) cost breakdown (Table-6 analogue).
- `reliability_sweep(problem, methods, points)` + `virtual_best(results)` +
  `sweep_to_csv(...)` → convergence-reliability grids (Figs 9–12, Table 7).
- `PROBLEMS` registry of parameterized test problems
  (`reaction_diffusion`, `unit_selection`, `gas_pipeline`, `recycle_loop`,
  `cstr_dynamic`).
Details and the local-vs-global-solve caveat: `references/study-tools.md`.

## Task routing

- **Which method / when** → `references/methods.md`
- **Every option, with recipes** → `references/options.md`
- **gr/lm is slow, hangs, or densifies; infeasibility; recovery failed** →
  `references/troubleshooting.md`
- **Dual recovery, implicit cyclic-block elimination, Schur/ordering** →
  `references/recovery-and-implicit.md`
- **Reproducing the paper's structural/runtime/reliability study** →
  `references/study-tools.md`
- **Runnable end-to-end scripts** → `examples/` (and
  `python -m discopt.aggregation.examples` for 30 in-tree worked examples)

## Installation / sanity check

```bash
uv pip install -e .          # in the discopt-aggregation repo; requires discopt >= 0.5
python -c "from discopt.aggregation import aggregate, solve; print('ok')"
python -m pytest tests/ -q   # the full suite
```
`discopt.aggregation` is a PEP-420 namespace subpackage that merges into the
installed `discopt` package at import time.
