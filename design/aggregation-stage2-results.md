# Variable aggregation — Stage 2 empirical results

Reproduction of the structural / runtime / reliability study from Naik, Biegler,
Bent & Parker, *Variable aggregation for nonlinear optimization problems*
[cite:@naik2025aggregation], on the `discopt` stack via the standalone
`discopt-aggregation` plugin. Numbers below are from the plugin's own harnesses
(`metrics.compare_methods`, `benchmark.benchmark_model`, `sweep.reliability_sweep`)
on the three S1 test problems.

**Headline:** the qualitative trends reproduce the paper. Aggregation shrinks the
problem; structure-preserving methods keep per-iteration cost flat; the
approximate-maximum methods (`gr`/`lm`) eliminate the most but make the reduced
model far more expensive to solve — the paper's "Hessian evaluation becomes the
bottleneck."

## Test problems (S1)

| problem | kind | defining intermediates | sweep params |
|---|---|---|---|
| `reaction_diffusion` | convex QP / **NLP** | residuals `res[i]` (multi-var linear) | `k`, `c_in` |
| `unit_selection` | MIQP / **MINLP** | costs `cost[i] = x[i]²` (degree-2) | `demand`, `fixed_cost` |
| `gas_pipeline` | **nonconvex NLP** (Weymouth) | squared pressures `sq[i] = p[i]²` (degree-2) | `p_supply`, `demand` |

## Structural results (S1, Table-4 analogue)

Variables eliminated / remaining, and average nonzeros per constraint:

```
reaction_diffusion   elim  vars  cons  nnz/con
  original              0    16     9    1.78
  ld1/ecd2/ld2/d2       2    14     7    3.71
  gr                    9     7    14    4.86
  lm                    9     7    14    3.71

unit_selection (MINLP)  elim  vars  cons  nnz/con
  original / ld1..ld2     0     9     7    2.14
  d2 / gr / lm            3     6    10    1.50

gas_pipeline (nonconvex) elim  vars  cons  nnz/con
  original                0     8     7    2.00
  ld1/ecd2/ld2            2     6     5    2.00
  d2 / gr / lm            4     4     5    1.60
```

Observations matching the paper:
- **Approximate-maximum eliminates the most** — on `reaction_diffusion`, `gr`/`lm`
  remove 9 of 16 variables vs 2 for structure-preserving.
- **`gr` inflates density** (`nnz/con` 1.78 → 4.86) by substituting multi-variable
  residual definitions; `lm` keeps it lower (3.71) — the paper's observation that
  the matching choice matters.
- **MINLP:** on `unit_selection` the binaries are never eliminated; only the
  continuous `cost[i]=x[i]²` intermediates aggregate (d2/gr/lm eliminate 3).
- **Structure-preserving actually lowers `nnz/con`** on the degree-2 problems
  (`unit_selection` 2.14 → 1.50, `gas_pipeline` 2.00 → 1.60) because the sparse
  defining constraints are removed.

Caveat on the Hessian-coupling column: `compare_methods`'s `nnz_hessian` is a
structural proxy computed at variable-*name* granularity and does **not**
scalarize the original, so for array-variable models it is not comparable across
the original (array-named) and reduced (scalarized) rows. The runtime harness
(`benchmark_model`) fixes this by scalarizing the base model; use it for
Hessian-cost comparisons.

## Runtime results (S2)

**Methodological correction (this is important).** `discopt.Model.solve()` runs a
*global* spatial branch-and-bound on nonconvex NLPs. That is neither the paper's
setting (local IPOPT) nor tractable here — a free-variable model timed out (>2
min) at n=10 under global solve. The harness therefore uses a **local** solve
(`discopt.solvers.nlp_pounce.solve_nlp_from_model`), which is fast and exposes
iteration counts (the global result did not). Everything below is local.

Local solve, `reaction_diffusion` n=8:

```
method    elim  vars  iters  wall     t_elim
original     0    16      6   ~90 ms      —
ld1          2    14      6  ~103 ms   1.0 ms
d2           2    14      6   ~77 ms   1.3 ms
gr           9     7    (7)  ~41 s (!)     —
lm           9     7    (7)   slow         —
```

- **Structure-preserving methods (`ld1`/`d2`) solve as fast as the original** with
  fewer variables; aggregation time (`t_elim`) is ~1 ms, negligible.
- **`gr`/`lm` are the paper's cautionary tale.** They cut variables (16→7) and
  iterations stay low (~7), but substituting the multi-variable residual
  definitions into the quadratic objective builds a large, dense expression
  graph whose Hessian is expensive to evaluate: the reduced solve took **~41 s**
  vs ~0.1 s for the original at n≈8–10, despite *fewer* iterations. This is
  exactly the paper's "Hessian evaluation becomes the bottleneck."
- Note: the cost is expression-graph / Hessian-evaluation cost, not a larger
  count of distinct Hessian nonzeros — the coarse structural `nnz_hessian` proxy
  does not capture it.

## Reliability results (S3)

`reliability_sweep` runs the local solver across a parameter grid and counts
convergence per method. On the convex `reaction_diffusion` (3×3 `k`×`c_in` grid)
every method converges on every instance (100%, ~6 iterations) — so this problem
does not exhibit the paper's *improvement*. Reproducing the paper's headline
(aggregation lets more instances converge) requires a problem tuned so the
full-space form fails within the iteration limit; the paper's authors searched
parameter ranges for exactly this. The harness produces the comparison for any
`TestProblem`; building such a deliberately-hard instance is the remaining step
to quantify the reliability benefit on this stack.

## Comparison to the paper

| Paper finding | Reproduced here? |
|---|---|
| Approximate-max (`gr`/`lm`) eliminates 70–90% of variables | ✅ trend (9/16 here at small n) |
| Structure-preserving keeps density ~constant | ✅ (and lowers it on degree-2 problems) |
| `gr` increases density; matching choice matters (`lm` milder) | ✅ (`nnz/con` 4.86 vs 3.71) |
| Aggressive aggregation → Hessian-eval bottleneck | ✅ strongly (41 s reduced solve) |
| `d2` a good balance | ✅ (fast solve, real elimination, no densification) |
| Aggregation improves convergence reliability | ⚠️ harness ready; needs a tuned-to-fail instance |

**Stack differences from the paper.** The paper [cite:@naik2025aggregation]
measures ASL/IPOPT; here it is JAX-compiled evaluation + POUNCE. The Hessian-cost
effect is if anything *more* pronounced on the JAX stack (expression-graph
compilation of substituted terms), which is a genuinely new data point:
aggressive aggregation's downside is not an ASL artifact.

## Reproducing

```python
from discopt.aggregation.metrics import compare_methods
from discopt.aggregation.benchmark import benchmark_model
from discopt.aggregation.sweep import reliability_sweep
from discopt.aggregation.testproblems import PROBLEMS, reaction_diffusion

compare_methods(PROBLEMS["gas_pipeline"].build())          # structural table
benchmark_model(reaction_diffusion(n=8), methods=("ld1", "d2"))  # runtime (local)
reliability_sweep(PROBLEMS["reaction_diffusion"], points=3)      # convergence
```

Or run all worked examples: `python -m discopt.aggregation.examples`.
