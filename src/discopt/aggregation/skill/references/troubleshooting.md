# Troubleshooting & pitfalls

## `gr`/`lm` is slow, hangs, or the reduced model won't evaluate

**Cause:** substitution compounds the expression graph. Eliminating a chain of
2-variable-eliminable constraints (`y0=f(x)`, `y1=g(y0)`, …) inlines each into
the next, so the *tree* size grows toward degree-`2^n` even though the shared
DAG is small. discopt's expression→jaxpr lowering re-traces shared nodes, so a
few-thousand-node DAG can have a multi-million-node tree — Hessian evaluation
then dominates (the paper's "Hessian evaluation becomes the bottleneck").

**Fixes, in order:**
1. `order="min_fill"` — visits low-fan-out candidates first (~40× smaller
   expressions on DAE models). Cheapest, always try first.
2. `max_def_nodes=<N>` — hard cap on substituted expression size (drops the
   offending eliminations).
3. Use `method="d2"` (or `d3`) instead — structure-preserving methods can't
   compound because each definition has ≤2 variables.
4. If the culprit is a *cyclic* block, use `eliminate_implicit` (keeps it
   implicit, no substitution) — see `recovery-and-implicit.md`.

**Diagnose it:**
```python
from discopt.aggregation import aggregate, model_expression_size
a = aggregate(m, method="gr")
print(model_expression_size(a.reduced_model))   # (tree, dag); tree >> dag => blowup
```

## Local vs global solve (important for benchmarking nonconvex models)

`discopt.Model.solve()` runs a **global** spatial branch-and-bound on nonconvex
NLPs — slow, and NOT the paper's setting (local IPOPT). For runtime/reliability
work use the **local** solver:
```python
from discopt.solvers.nlp_pounce import solve_nlp_from_model   # local NLP
# or the harness helper:
from discopt.aggregation.benchmark import _local_solve        # (status, iters, wall, obj)
```
`solve(m, method=...)` calls the reduced model's `.solve()` (global). On a
nonconvex reduced model with a big expression this can appear to "hang" — it's
the global solver, not aggregation. Use `_local_solve(aggregate(m, ...).reduced_model)`
to time the paper's setting.

## `solve()` returns `status="infeasible"` unexpectedly

This is intentional and usually correct: a fixed-variable elimination produced a
value outside that variable's bounds, or a fully-determined system is
inconsistent. Check:
```python
a = aggregate(m, method="ld1")
print(a.infeasible_eliminations)   # [(var, value, lb, ub), ...] -> the model IS infeasible
```
If you believe the model is feasible, your bounds or defining equations are
inconsistent — inspect them. Aggregation is telling you the truth earlier than
the solver would.

## Dual recovery reports `available=False`

`res.duals.reason` tells you why:
- `"maximise"` — maximize objectives are out of scope (sign convention).
- `"active_eliminated_bound:<name>"` — an eliminated variable sits on a finite
  bound at the optimum; the γ-term isn't modeled. Recovery is unavailable, not
  wrong.
- `"missing_reduced_duals"` — the backend didn't expose constraint duals.
- `"unsupported_derivative_or_singular"` / `"residual_check_failed"` — a
  derivative couldn't be formed, or the KKT self-check failed; recovery declines
  rather than return a wrong multiplier.
Single-pass and recursive (multi-pass) paths are both supported for
minimize + interior solutions.

## Nothing gets eliminated

- The model may have no variable *defined by* an equality (aggregation needs
  `y - f(...) == 0` forms, not just any constraint).
- Array variables: ensure `scalarize=True` (default). If an array is used whole
  (`sum(a)`, a slice, a matmul) it abstains — index it elementwise to make it
  eliminable.
- `pivot_tol` too high for a poorly-scaled model — try `pivot_tol=0.0`.
- The defining variable appears *nonlinearly* in its own equation (e.g.
  `y + y**3 == x`) — it can't be isolated, so it's not a candidate.

## Results differ between runs

They shouldn't — the pipeline is deterministic across `PYTHONHASHSEED`. If you
see variation, you're likely comparing across a code change or a different
`method`/`order`. Confirm with:
```bash
for s in 0 5 7; do PYTHONHASHSEED=$s python -c "from discopt.aggregation import aggregate; from discopt.aggregation.testproblems import reaction_diffusion; print(sorted(aggregate(reaction_diffusion(8), method='lm').eliminated))"; done
```

## MINLP: binaries/integers

Never eliminated by any method. If you need an integer relationship substituted,
that's out of scope — aggregation targets scalar continuous variables only.
