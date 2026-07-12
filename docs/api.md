# API reference

Everything below is importable from the top-level `discopt.aggregation` namespace,
e.g. `from discopt.aggregation import aggregate, solve`. The package exports 83
public names; this page covers the ones you reach for directly. See
[Options](options.md) for a per-option reference and
[Recovery & diagnostics](recovery.md) for the recovery/analysis surfaces.

## Core

### `aggregate(...) -> AggregationResult`

```python
aggregate(
    model,
    method="d2",
    recursive=True,
    pivot_tol=1e-6,
    max_fill=None,
    prune_bounds=True,
    scalarize=True,
    preserve_target_linearity=False,
    tearing="greedy",
    decomposable=False,
    order="index",
    max_def_nodes=None,
    max_condition=None,
    integer_aggregation=False,
)
```

Build the reduced-space model and the recovery map without solving. The returned
`AggregationResult` exposes `reduced_model`, `eliminated`, `n_eliminated`,
`stats` (including `tearing_fallbacks`), `infeasible_eliminations`, and
`recover_full(kept_values)`.

### `solve(...) -> AggregatedSolveResult`

```python
solve(model, ..., recover_duals=False, **solve_kwargs)
```

Aggregate, solve the reduced model, and recover the full solution. Takes every
`aggregate` argument plus `recover_duals` and any extra `**solve_kwargs`
forwarded to the reduced model's `.solve()`. The result exposes `status`,
`objective`, `x` (full-space, original shapes), `duals`, `reduced_result`, and
`aggregation`.

See [Options](options.md) for every keyword.

## Structural analysis

| Function | Returns | Purpose |
|---|---|---|
| `compare_methods(model, methods=None, exact_hessian=False)` | `list[dict]` | Paper Table 4: per-method structural metrics. |
| `model_structure(model, exact_hessian=False)` | `ModelStructure` | Structural metrics of one model. |
| `structural_diagnosis(model)` | `StructuralDiagnosis` | Dulmage–Mendelsohn well-posedness (`free_variables`, `degrees_of_freedom`, over-determined blocks). |
| `build_incidence(model, ...)` | `IncidenceModel` | The bipartite variable–constraint incidence. |
| `model_expression_size(model)` | `(tree, dag)` | Detect substitution expression blow-up. |

## Recovery & duals

| Function | Returns | Purpose |
|---|---|---|
| `recover_duals(reduced, reduced_result)` | `DualRecovery` | Full-space multipliers for a single `apply_aggregation`. |
| `recover_duals_from_result(result, reduced_result)` | `DualRecovery` | Recursive (multi-pass) dual recovery. |
| `kkt_stationarity_residual(model, full_x, constraint_duals, bound_duals=None)` | `float` | Independent KKT self-check (≈0 when duals are correct). |

See [Recovery & diagnostics](recovery.md).

## Implicit (cyclic block) elimination

| Function | Returns | Purpose |
|---|---|---|
| `implicit_blocks(model)` | `list[ImplicitBlock]` | The irreducible cyclic blocks. |
| `eliminate_implicit(model, *, pivot_tol=1e-6, x0_value=1.0)` | `ImplicitElimination` | Replace nonsingular cyclic blocks with `Model.implicit` nodes. |
| `kkt_schur_indices(model, *, method="lm", return_model=False)` | `list[int]` | KKT-space indices of the reducible Schur block. |
| `block_triangular_ordering(model)` | `BlockTriangularOrdering` | Fill-reducing block permutation. |

## Study & benchmark harnesses

| Function | Returns | Purpose |
|---|---|---|
| `benchmark_model(model, methods=..., run_solve=True, options=None)` | `list[BenchmarkRow]` | Paper Table 6 / Fig 8 runtime (local solves). |
| `callback_breakdown(model, method)` | `dict` | Per-callback (func/grad/jac/hess) cost split. |
| `reliability_sweep(problem, methods=..., points=3, solver_options=None)` | `dict[str, SweepResult]` | Paper Table 7 / Figs 9–12 convergence grids. |
| `virtual_best(results)` | per-cell best method | Paper Fig 12. |
| `sweep_to_csv(results, path)` | — | Export every per-cell outcome. |
| `PROBLEMS` | `dict[str, TestProblem]` | The parameterized test problems. |

## Test problems

`PROBLEMS` maps names to `TestProblem` (a builder plus nominal params and sweep
ranges): `reaction_diffusion`, `unit_selection`, `gas_pipeline`, `recycle_loop`,
`cstr_dynamic`, `inventory_chain`. Each builder is also exported directly (e.g.
`from discopt.aggregation import cstr_dynamic`). See
[Study & benchmarks](study.md).
