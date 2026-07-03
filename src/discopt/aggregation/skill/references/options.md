# Options — detailed, with recipes

Every option is a keyword arg to both `aggregate` and `solve`. Defaults are safe.

## Aggregation shape

### `recursive` (default `True`)
`True` applies the strategy pipeline to a fixed point (paper §3.3). `False` does
exactly one pass of the named strategy — useful to inspect a single step or when
you want a bounded, predictable reduction.

### `order` (default `"index"`) — fill-aware greedy (R1)
`"min_fill"` visits constraints and picks variables in ascending estimated-fill
order, so `gr`/`lm` prefer low-fan-out substitutions.
```python
solve(m, method="gr", order="min_fill")   # same eliminations, ~40x smaller
                                           # substituted expression on DAE models
```
**Use whenever you use `gr`/`lm`.** It is the single cheapest defense against the
substitution expression-graph blow-up. No effect on structure-preserving methods
(their definitions are already low-fan-out).

### `tearing` (default `"greedy"`)
How `gr`/`lm` tear cyclic blocks. `"exact"` keeps the maximum acyclic subset
(a minimum-feedback-vertex-set MILP for larger blocks, brute force for small),
eliminating at least as many per block as greedy.
```python
aggregate(m, method="lm", tearing="exact")
```
Falls back to greedy on any MILP failure (never degrades).

### `decomposable` (default `False`)
Local-search the `lm` maximum matching for a more decomposable block structure
(higher Theorem-2 lower bound) before tearing. Marginal on small models; helps
where the matching has freedom (large models).

## Numerical guards

### `pivot_tol` (default `1e-6`)
A variable is eliminable only if `|coeff| >= pivot_tol * max(1, max_j|coeff_j|)`
in its defining constraint — the relative pivot guard Achterberg et al. flag for
MIP aggregation. `0.0` restores exact paper behavior (eliminate any nonzero
pivot). Raise it if you suspect ill-conditioned substitutions.

### `max_condition` (default `None`)
Cap the worst-case recovery-error amplification `max_k |∂D_v/∂k|` bounded over
the variable boxes — this composes both the pivot *and* the coefficient
magnitudes (unlike `pivot_tol`, which is per-pivot). Catches e.g. `v = 1e6*u`
(unit pivot, huge amplification). Drops over-cap eliminations.
```python
aggregate(m, method="gr", max_condition=1e3)   # drop chains amplifying error >1000x
```

### `max_fill` (default `None`)
Cap the estimated new Jacobian incidences from an elimination
(`(#other vars in def) * (#target constraints)`). Drops high-fan-out
eliminations. A structural (not expression-size) cap.

### `max_def_nodes` (default `None`)
Cap the *resolved* (inlined) defining-expression node count. This is the direct
guard against the Hessian/expression blow-up: it bounds how big any substituted
expression can get.
```python
aggregate(m, method="gr", max_def_nodes=200)   # never inline an expr >200 nodes
```
Compose with `order="min_fill"`: the order keeps expressions small, the cap
enforces a hard ceiling.

## Model reach & bounds

### `scalarize` (default `True`)
Rewrites fully-indexed array variables (`m.continuous("x", shape=(n,))`) into
per-element scalars so aggregation reaches discretized/DAE models — the paper's
entire test set. No-op on scalar-only models. Arrays used *whole* (`sum(a)`, a
slice, a matrix multiply) are left as arrays (conservative abstention). Recovery
reassembles arrays so `res.x["x"]` has the original shape.

### `prune_bounds` (default `True`)
Each bounded eliminated variable would add up to two box→inequality constraints
(paper §2.4). When interval propagation proves the defining expression's range
stays inside the box, those inequalities are omitted as redundant. This is what
keeps eliminating *bounded* intermediates a net size win. Turn off only to
reproduce exact paper counts.

### `preserve_target_linearity` (default `False`)
Drop an elimination whose *nonlinear* defining expression would turn a
currently-*linear* constraint nonlinear (the degradation Amarger et al. warn
about). Use when downstream steps rely on constraints staying linear.

## Integer aggregation

### `integer_aggregation` (default `False`) — opt-in MIP-style elimination

Also eliminate scalar integer/binary variables whose integrality is *implied*:
```python
solve(m, method="lm", integer_aggregation=True)   # MINLP: collapse an integer chain
```
Admitted only when the defining equality is fully affine, the pivot is ±1, every
other participant is integer/binary, and every other coefficient and the constant
are integral. Recovered integers are exactly integral. Default-off and a proven
no-op with the flag off. Pair with `order="min_fill"` on long chains (though an
all-integer, unit-coefficient chain is affine and does not blow up). See
`references/methods.md` for the full rule and what abstains.

## Solve-only

### `recover_duals` (default `False`, `solve` only)
Also recover full-space Lagrange multipliers into `res.duals`. See
`recovery-and-implicit.md`.

### `**solve_kwargs`
Any extra kwargs to `solve()` are forwarded to the reduced model's `.solve()`.

## Composing guards — a robust aggressive recipe

```python
res = solve(
    m,
    method="lm",
    order="min_fill",       # keep substituted expressions small
    max_def_nodes=500,      # hard ceiling on expression size
    max_condition=1e4,      # drop ill-conditioned recovery chains
    tearing="exact",        # maximize per-block elimination
)
```
If the reduced solve is still slow, the model is a densification case — fall back
to `method="d2"`.
