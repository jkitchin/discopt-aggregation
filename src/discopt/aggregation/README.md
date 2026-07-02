# discopt.aggregation

Variable aggregation (reduced-space presolve) for nonlinear optimization models,
following Naik, Biegler, Bent & Parker, *Variable aggregation for nonlinear
optimization problems* ([arXiv:2502.13869](https://arxiv.org/abs/2502.13869)).

Variable *aggregation* substitutes a variable that is defined by an equality
constraint into the rest of the model, eliminating both a variable and a
constraint to yield a smaller **reduced-space** formulation. This can improve
convergence reliability and (often) solve time; the trade-off is that aggressive
aggregation can make constraints denser and Hessian evaluation more expensive.

This package is pure Python, runs on the `discopt.modeling` expression DAG before
the Rust/JAX engine, and adds no dependencies beyond discopt's core
(numpy/scipy) — so it works out of the box under `uv`.

## Quick start

```python
import discopt.modeling as dm
from discopt.aggregation import aggregate, solve

m = dm.Model("example")
x = m.continuous("x", lb=-5, ub=5)
y = m.continuous("y", lb=-5, ub=5)
m.subject_to(y - (x + 1) == 0.0)   # y is defined by x
m.minimize(x**2 + y**2)

# One-shot: aggregate, solve the reduced model, recover the full solution.
res = solve(m, method="d2")
print(res.status, res.objective)   # optimal 0.5
print(res.x)                       # {'x': -0.5, 'y': 0.5}  (y recovered)

# Or inspect the reduced model / recovery map yourself.
agg = aggregate(m, method="d2")
print(agg.eliminated)              # ['y']
reduced = agg.reduced_model
```

## Methods

Ordered from most conservative to most aggressive (`method=` argument):

| key    | name                        | defining constraints eliminated  | paper |
|--------|-----------------------------|----------------------------------|-------|
| `ld1`  | fixed-variable (degree-1)   | `y = a`                          | Alg 3 |
| `ecd2` | equal-coefficient degree-2  | `y = x + a`                      | Alg 6 |
| `ld2`  | linear degree-2             | `y = a*x + b`                    | Alg 5 |
| `d2`   | degree-2 (**default**)      | `y = f(x)` (≤ 2 variables)       | Alg 4 |
| `gr`   | greedy                      | `y = f(w, x, …)` (`y` linear)    | Alg 1 |
| `lm`   | linear-matching             | `y = f(w, x, …)` (`y` linear)    | Alg 2 |

`ld1`/`ecd2`/`ld2`/`d2` are **structure-preserving** (they never add variables to
a remaining constraint). `gr`/`lm` are **approximate-maximum** — they eliminate
the most variables (70–90% in the paper's models) but can increase density and
nonlinearity. The paper recommends `d2` as the best balance; it is the default.
The degree-2 filter generalises to a **degree-k** family: `method="d3"`, `"d4"`,
… (any `k ≥ 2`) eliminate using defining constraints with at most `k` variables.

## Options

`aggregate(model, method=..., **opts)` and `solve(...)` accept:

| option | default | effect |
|--------|---------|--------|
| `recursive` | `True` | apply the §3.3 pipeline to a fixed point (fixed vars first, then the method) |
| `pivot_tol` | `1e-6` | refuse eliminations whose pivot is tiny relative to the row (ill-conditioning guard); `0.0` = exact paper behaviour |
| `max_fill` | `None` | cap the estimated new Jacobian incidences per elimination |
| `prune_bounds` | `True` | drop box→inequality constraints proven redundant by interval propagation |
| `scalarize` | `True` | rewrite array variables into scalars so aggregation reaches discretized models (no-op on scalar models) |
| `preserve_target_linearity` | `False` | refuse an elimination whose nonlinear substitution would turn a currently-linear constraint nonlinear |
| `tearing` | `"greedy"` | `"exact"` uses the maximum-acyclic-subset tear on cyclic blocks (`gr`/`lm`) |

## Correctness

Every strategy returns a strictly lower-triangular aggregation set (Lemma 1), so
each eliminated variable is an explicit function of the kept variables and is
recovered post-solve by back-substitution — never an iterative solve. Only scalar
continuous variables are eligible; integer/binary variables are never
eliminated. An eliminated variable's bounds become up to two inequality
constraints on its defining expression (Section 2.4).

## Dual recovery

`recover_duals(reduced, reduced_result)` reconstructs the full-space Lagrange
multipliers from a reduced-space solve: surviving-constraint multipliers map back
directly, and each eliminated defining equality's multiplier is recovered by
reverse-order stationarity back-substitution (using exact symbolic gradients from
`differentiate_expression`). `kkt_stationarity_residual` validates the recovered
point. Scope: single-pass, minimise objective, no active bounds on eliminated
variables — otherwise it reports `available=False` rather than a wrong result.

## Structural study

`compare_methods(model)` reproduces the paper's Table 4 (variables, constraints,
eliminations, Jacobian density, and structural Hessian coupling) across all six
methods, for reproducing the structure/solve-time/reliability analysis on
discopt models.

## Examples

`aggregation/examples.py` is runnable, commented documentation covering all six
methods, the recursive pipeline, end-to-end solve + recovery, the structural
comparison, the pivot/fill safeguards, and a **performance/scaling demo**:

```bash
python -m discopt.aggregation.examples
```

The performance example (`example_performance`) shows the solve-time benefit and
how it scales — e.g. on a chained model with univariate defining equations,
eliminating 2n intermediates gives ~1.2× at 300→100 variables and ~1.65× at
600→200 variables (solver-internal time; the speedup grows with size as KKT
factorization comes to dominate). It also documents the honest caveat: aggressive
methods that densify the Hessian or eliminate bounded variables (adding
inequalities) may *not* help solve time — the smaller problem is the guaranteed
benefit, raw speed is problem- and solver-dependent.

## Modules

- `matching` — graph kernel: Hopcroft–Karp matching, Tarjan SCC,
  block-triangularization (Alg 7–9). Stdlib-only.
- `incidence` — build the bipartite incidence graph from a model (Table 1).
- `strategies` — the six selection policies (Alg 1–6).
- `transform` / `recovery` — apply an aggregation set and recover eliminated
  variables.
- `api` — `aggregate` / `solve` public entry points (recursive Section 3.3
  pipeline, all options above).
- `metrics` — structural metrics harness (Section 5.1 / Table 4), incl. exact
  AD Hessian nonzeros.
- `scalarize` — rewrite array variables into scalars so aggregation reaches
  discretized (indexed) models.
- `derivatives` — exact symbolic differentiation over the expression DAG.
- `duals` — full-space Lagrange-multiplier recovery + KKT residual.
- `testproblems` — parameterized NLP/MINLP/nonconvex problems (`reaction_diffusion`,
  `unit_selection`, `gas_pipeline`) for the empirical study.
- `benchmark` — local-solve runtime instrumentation (iterations, wall time).
- `sweep` — convergence-reliability sweeps over parameter grids.

See `../../../design/aggregation-stage2-results.md` for the reproduction
write-up, and `aggregation/examples.py` (`python -m discopt.aggregation.examples`)
for runnable walkthroughs of every feature.

## Reference

This package implements the methods of:

> Sakshi Naik, Lorenz Biegler, Russell Bent, and Robert Parker.
> **"Variable aggregation for nonlinear optimization problems."**
> arXiv:2502.13869 (2026). <https://arxiv.org/abs/2502.13869>

```bibtex
@article{naik2026variable,
  title   = {Variable aggregation for nonlinear optimization problems},
  author  = {Naik, Sakshi and Biegler, Lorenz and Bent, Russell and Parker, Robert},
  journal = {arXiv preprint arXiv:2502.13869},
  year    = {2026},
  url     = {https://arxiv.org/abs/2502.13869},
}
```

Section and algorithm numbers referenced throughout the source (e.g. "Alg 2",
"Table 4", "Lemma 1", "Section 3.3") point into this paper.
