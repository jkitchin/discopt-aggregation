# Primal/dual recovery, implicit elimination, Schur/ordering

## Primal recovery (automatic)

`aggregate` returns an `AggregationResult` whose `recover_full(kept_values)`
reconstructs the eliminated variables by evaluating their defining expressions in
order (no iterative solve — Lemma 1 guarantees strict lower-triangularity).
`solve()` does this for you; `res.x` is the full-space point at the original
variable shapes.
```python
a = aggregate(m, method="d2")
kept = a.reduced_model.solve().x
full = a.recover_full(kept)     # {name: value} incl. eliminated, arrays reassembled
```

## Dual (Lagrange-multiplier) recovery

```python
res = solve(m, method="d2", recover_duals=True)
dr = res.duals
if dr.available:
    dr.constraint_duals      # {original_cid: multiplier}  (surviving + defining)
    dr.bound_duals           # {eliminated_var: γ}  active-bound multipliers (DU2)
    dr.full_x                # the point used
else:
    print(dr.reason)         # why not (see troubleshooting.md)
```
The eliminated defining equalities' multipliers are recovered by solving the
defining-multiplier system (a linear solve — recursive passes can couple, so it
is *not* a back-substitution). A **KKT stationarity self-check** gates every
result: `available=True` means the full-space multipliers satisfy stationarity to
~1e-6, so a residual algebra gap becomes an honest `available=False`, never a
wrong number.

**Active eliminated-variable bounds (DU2).** When an eliminated variable sits on a
finite bound at the optimum, its box→inequality (`d(kept) <= ub` / `>= lb`)
survives in the reduced model and carries a multiplier. By substitution
invariance that reduced dual **equals** the base-model bound multiplier γ, so
recovery collects it into `dr.bound_duals[var]` (discopt normalises `v >= lb` to
`lb - v` and `v <= ub` to `v - ub`, so the term added to that variable's
stationarity row is `-μ` for a low bound, `+μ` for a high bound), feeds it into
the defining-multiplier system, and the self-check includes it. An *interior* or
pruned bound simply contributes γ ≈ 0.

**Scope:** minimize **or** maximize objective; active **or** interior
eliminated-variable bounds (their γ-terms are recovered). A maximize objective's
multipliers are reported by discopt in the `minimize(-f)` convention; recovery
normalises the objective-gradient sign so recovered duals stay directly
comparable to a direct solve. A still-active bound whose inequality was *pruned*
as redundant (so its γ is unobservable — which cannot actually happen for a truly
active bound) is reported defensively as `available=False`.

Validate independently with (pass `bound_duals` so active bounds are accounted):
```python
from discopt.aggregation import kkt_stationarity_residual
kkt_stationarity_residual(
    res.aggregation.base_model, dr.full_x, dr.constraint_duals, dr.bound_duals
)  # ~0
```

## Implicit elimination of cyclic blocks

Explicit methods **drop** irreducible cyclic blocks (mutually-dependent defining
equations — e.g. a closed recycle loop, an index-1 DAE). To eliminate them, use
the implicit-function reduced-space move: solve the block for its variables
inside a differentiable inner solve.

```python
from discopt.aggregation import eliminate_implicit, implicit_blocks

# inspect what's cyclic
for blk in implicit_blocks(m):
    print(blk.variables)                 # the mutually-dependent variables

el = eliminate_implicit(m)               # reduced model uses Model.implicit nodes
reduced = el.reduced_model               # solve this (AD-differentiable through the block)
full = el.recover(reduced.solve().x)     # reconstruct block variables post-solve
```
Only *nonsingular* blocks over scalar variables with JAX-supported ops are
eliminated; others are left in place. Eliminated block variables' bounds are
re-expressed as inequalities on the node components (so the reduced solve can't
violate them). `ImplicitBlock` also exposes the primitive directly
(`solve`, `is_nonsingular`, `implicit_derivative` via the implicit function
theorem) for analysis.

**When to use:** a recycle loop or DAE where explicit aggregation eliminates
everything *except* one irreducible cycle, and you want that gone too. This is
Parker et al.'s implicit-function formulation.

## Structural diagnostics (Dulmage–Mendelsohn)

Run this *before* eliminating to understand a model's structure:
```python
from discopt.aggregation import structural_diagnosis
d = structural_diagnosis(m)
d.free_variables               # degrees of freedom (unmatched under-determined vars)
d.degrees_of_freedom           # DOF count (under-determined excess)
d.overdetermined_constraints   # structurally over-determined
d.is_structurally_singular     # bool
for w in d.warnings():
    print(w)
```
The **square block** of the DM decomposition is exactly what aggregation can
eliminate; over/under-determined blocks signal a modeling error the solver would
otherwise hit as an infeasibility or singular KKT.

## Schur / block-triangular KKT (structured linear solve)

The linear-matching aggregation identifies a large nonsingular
block-lower-triangular submatrix of the KKT system — a natural Schur pivot block.

```python
from discopt.aggregation import kkt_schur_indices, block_triangular_ordering

idx = kkt_schur_indices(sm)                    # KKT-space indices of the reducible block
#   -> pass to a solver's set_kkt_schur_block, e.g. via discopt's pounce bridge:
#      solve_nlp_from_model(sm, x0, kkt_schur_block=idx)
perm = block_triangular_ordering(m).permutation  # a fill-reducing / block permutation
```
`kkt_schur_indices` computes indices in the *scalarized* model's flat KKT layout
(`[x, slack, eq-dual, ineq-dual]`); solve that same model. Pass
`return_model=True` to get `(indices, analyzed_model)` and avoid a mismatch.
This is plugin-side analysis; handing indices to the solver belongs in discopt's
pounce bridge (correctness-safe — the solver falls back if the block is
unsuitable).
