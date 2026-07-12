# Method selection

Six strategies plus a degree-k generalization, from most conservative to most
aggressive. `method=` selects one; the structure-preserving families run
fixed-variable elimination (`ld1`) first, then their degree-bounded pass,
recursively to a fixed point. `gr`/`lm` run once.

## The methods

### `ld1` — fixed-variable (degree-1), paper Alg 3
Eliminates `y = a` (a variable that is the only one in a linear equality).
Never increases density or nonlinearity. Always safe; run it as a cheap first
pass on any model. Typically removes a few % of variables.

### `ecd2` — equal-coefficient degree-2, paper Alg 6
Eliminates `y = x + a` (two variables, equal-magnitude coefficients). The most
restrictive structure-preserving method: substituting `x + a` leaves Jacobian
entries unchanged when the target doesn't already contain `x`. Preserves
linearity. Use when you want to touch the Jacobian pattern as little as possible.

### `ld2` — linear degree-2, paper Alg 5
Eliminates `y = a*x + b` (two variables, both linear). Preserves linearity (a
linear constraint stays linear) but linear coefficients can change. Analogous to
Pyomo's `linear_presolve=True`.

### `d2` — degree-2 (**DEFAULT, recommended**), paper Alg 4
Eliminates `y = f(x)` where the defining constraint has ≤2 variables and `y`
appears linearly. `f` may be nonlinear (`y = x**2 + 2`), so this can *introduce*
a nonlinearity into a previously-linear target — but it never increases the
*number* of variables in any constraint. The paper's recommended balance:
meaningful reduction, no density inflation, fast reduced solve.

### `d<k>` — degree-k (`d3`, `d4`, …)
Generalization of `d2`: eliminate using constraints with ≤k total variables.
An intermediate between `d2` and the approximate-maximum methods. Coverage is
monotone in `k` only when eliminations are acyclic (heuristic cycle-breaking
means a larger `k` can occasionally eliminate fewer). Accepted anywhere a method
string is (`aggregate(m, method="d3")`, `compare_methods(m, methods=["d2","d3"])`).

### `gr` — greedy (approximate-maximum), paper Alg 1
Iterate constraints; eliminate the first linearly-appearing variable not yet
used. Eliminates any linearly-defined variable regardless of how many other
variables its defining expression has — so a 4-variable definition gets
substituted into every target, **inflating density and nonlinearity**. Removes
70–90% of variables. Inspired by AMPL's `substout=1`.

### `lm` — linear-matching (approximate-maximum), paper Alg 2
Maximum matching on the linear bipartite graph → block-triangularize →
per-block greedy tear. Comes with a Theorem-2 bound `n_block ≤ n_agg ≤ n_match`
on how many it eliminates. Similar coverage to `gr`; the matching choice can
make its reduced model milder (lower density) than `gr` on some problems.

## Decision tree

1. **Default:** `d2`. It is the paper's recommendation and almost always the
   right first call.
2. **Want to touch structure as little as possible** (e.g. preserve a convex/GP
   structure a solver exploits, or keep constraints linear): `ecd2` → `ld2`.
   These preserve linearity; `d2` does not.
3. **Only clean fixed variables** (`y = const`): `ld1`.
4. **Want more reduction than `d2` but bounded degree:** `d3`, `d4`.
5. **Want maximum reduction** (smallest variable count): `gr` or `lm` — **but**
   always pair with `order="min_fill"`, and consider `max_def_nodes=` and a
   *local* solver, because the reduced model can become expensive to evaluate
   (see `troubleshooting.md`). Prefer `lm` if `gr` densifies too much; compare
   with `compare_methods`.
6. **Cyclic (irreducible) blocks** you want gone: explicit methods *drop* them;
   use `eliminate_implicit` (see `recovery.md`).

## Integer aggregation (opt-in)

By default only scalar *continuous* variables are eliminated. Pass
`integer_aggregation=True` to also eliminate a scalar **integer/binary** variable
— but only when its integrality is *implied* (standard MIP presolve, Achterberg
et al. 2020): the defining equality is **fully affine**, the **pivot is ±1**,
**every other participant is integer/binary**, and **every other coefficient and
the constant are integral**. Example that qualifies: an all-integer stock balance
`s[t] = s[t-1] + q[t] - d[t]`. What abstains: a continuous variable in the
definition, a non-unit pivot (`y = 2*i` can define `y` but cannot eliminate `i`),
a fractional coefficient or constant, or a nonlinear body. Recovery returns
exactly integral values (a non-integral result raises); dual recovery reports
`available=False` over eliminated integers. The flag is default-off and a proven
no-op with the flag off. Use it for MINLPs with integer accounting/counting
chains, where fewer integers means a smaller branching space. It does nothing for
models whose integers appear only in inequalities (e.g. `unit_selection`'s
on/off binaries).

## What each method will NOT eliminate

- Integer or binary variables **unless `integer_aggregation=True`** and
  integrality is implied (see above); by default they are kept — MINLP-safe.
- A variable in a genuinely opaque/unanalyzable constraint node (refused).
- A variable whose defining pivot is tiny relative to its neighbors
  (`pivot_tol`, default `1e-6`).
- A variable in a cyclic block (explicit methods) — needs implicit elimination.

## Structural comparison

To see what each method does to a specific model before committing:
```python
from discopt.aggregation import compare_methods
for row in compare_methods(m):                     # paper Table 4
    print(row["method"], "elim", row["n_eliminated"],
          "vars", row["n_variables"], "nnz/con", round(row["avg_nnz_per_con"], 2),
          "Hess", row["nnz_hessian"])
```
Watch `avg_nnz_per_con` and `nnz_hessian`: if they jump for `gr`/`lm`, that model
is a densification case — use `order="min_fill"` or stay with `d2`.
