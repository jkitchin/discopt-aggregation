# Study & reproduction harnesses (paper Tables 4/6/7, Figs 8–12)

The package reproduces Naik et al.'s empirical study on the discopt stack.

## Structural comparison — Table 4

```python
from discopt.aggregation import compare_methods
rows = compare_methods(model)                 # first row is "original"
# each row: method, n_eliminated, n_variables, n_constraints, n_equality,
#           nnz_jacobian, avg_nnz_per_con, avg_linear_nnz_per_con, nnz_hessian,
#           nnz_hessian_exact (if exact_hessian=True)
compare_methods(model, methods=["d2","d3","gr"], exact_hessian=True)
```
- `nnz_hessian` is the fast structural coupling *superset*; `nnz_hessian_exact`
  (opt-in) is the true AD Hessian nonzero count (the paper's Table 4 quantity).
- The `original` row is computed on the *scalarized* model so its Var./NNZ
  columns match the aggregated rows' per-scalar granularity.
- Structural metrics are DAG-memoised, so they stay fast even on gr/lm reduced
  models whose expression *trees* are huge.

## Runtime — Table 6 / Fig 8

```python
from discopt.aggregation import benchmark_model, callback_breakdown

rows = benchmark_model(model)                 # local solves; per-method BenchmarkRow
# BenchmarkRow: method, n_eliminated, n_variables, nnz_hessian, status,
#               iterations, solve_wall_time, aggregate_time, objective
benchmark_model(model, run_solve=False)       # cheap: structural only (for gr/lm)
benchmark_model(model, options={"max_iter": 3000})  # cap iterations (paper protocol)

b = callback_breakdown(model, method="gr")    # per-callback cost fractions
# {"func","grad","jac","hess","hess_over_func"} — the paper's func/jac/hess split
```
`benchmark_model` uses the **local** NLP solve (the paper's setting) and exposes
iteration counts. `callback_breakdown` times each evaluator callback per
evaluation (POUNCE exposes no solver-internal timing — a documented gap); the
Hessian-cost *trend* is robust at DAE scale, noisy at tiny sizes.

## Reliability — Table 7, Figs 9–12

```python
from discopt.aggregation import (reliability_sweep, virtual_best, sweep_to_csv,
                                  PROBLEMS)

res = reliability_sweep(PROBLEMS["gas_pipeline"], methods=("ld1","d2"), points=11,
                        solver_options={"max_iter": 3000})  # paper's iteration cap
# solver_options forwards to every local solve; with {"max_iter": N} an instance
# that stops at the cap counts as *not converged* (paper's reliability protocol).
# solver_options=None (default) leaves solves uncapped.
# {method: SweepResult}; SweepResult has n_converged, pct_converged,
#   avg_iterations, avg_wall_time, and .cells (per-grid-point CellResult) for the
#   Figs 9–11 convergence grids.
vb = virtual_best(res)                         # paper Fig 12: per-cell best method
sweep_to_csv(res, "sweep.csv")                 # every per-cell outcome
```
Grids default to `points=3` (fast); the paper's protocol is `points=11`
(121-point) — that is a `slow` run, keep it opt-in. Use structure-preserving
methods for sweeps; `gr`/`lm` reduced models can be slow/blow-up (guard with
`order="min_fill"` + `max_def_nodes` or leave them out).

## Test problems

`PROBLEMS` maps names to `TestProblem` (builder + nominal params + sweep ranges):
- `reaction_diffusion` — 1-D diffusion-reaction least-squares collocation
  (convex QP / NLP).
- `unit_selection` — on/off units with quadratic cost (MIQP / MINLP; binaries
  never aggregated).
- `gas_pipeline` — Weymouth gas line (nonconvex NLP).
- `recycle_loop` — torn steady-state recycle (parallel units; aggregation
  collapses the loop).
- `cstr_dynamic` — time-discretized CSTR optimal control with rational kinetics
  (discretized-DAE; `d2` eliminates the algebraic rate variables ~42%).

`recycle_cycle` / `RECYCLE_CYCLE` is a **closed** recycle loop used as the RL2
reliability probe. It is exported but **not** in `PROBLEMS`: a bounded search for a
"tuned-to-fail" instance showing aggregation converging *more* instances was an
**honest negative** on this stack. Explicit `d2`/`lm` are sound but never faster
(on the closed cycle `d2` needs *more* Newton iterations, a reliability loss); the
implicit path (`eliminate_implicit`) shows a raw convergence gap under a tight
`max_iter`, but it is confounded — the nonconvex cycle has two roots and the
bound-unaware inner Newton recovers the spurious negative one on ~43% of the grid.
See `design/aggregation-stage2-results.md` (Reliability results, RL2) and
`tests/test_aggregation_reliability.py`.

```python
from discopt.aggregation import PROBLEMS
p = PROBLEMS["cstr_dynamic"]
m = p.build(n=8, k=1.0, c_in=1.0)      # nominal or overridden
grid = p.sweep_grid(points=11)         # 121 parameter dicts
```

## 30 in-tree worked examples

```bash
python -m discopt.aggregation.examples    # runs every feature end to end, prints output
```
The `examples.py` functions double as executable documentation for every method,
guard, recovery path, diagnostic, and study tool.
