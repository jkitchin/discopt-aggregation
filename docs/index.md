---
hide:
  - navigation
---

<p align="center">
  <img src="assets/logo-wordmark.svg#only-light" alt="discopt-aggregation" width="620">
  <img src="assets/logo-wordmark-dark.svg#only-dark" alt="discopt-aggregation" width="620">
</p>

<p align="center">
  <a href="https://github.com/jkitchin/discopt-aggregation/actions/workflows/ci.yml"><img src="https://github.com/jkitchin/discopt-aggregation/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://arxiv.org/abs/2502.13869"><img src="https://img.shields.io/badge/paper-arXiv%3A2502.13869-b31b1b" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
</p>

**Variable aggregation** is a reduced-space presolve for nonlinear optimization: it
substitutes variables defined by equality constraints into the rest of a model,
eliminating both a variable and a constraint to yield a smaller reduced-space
formulation, then recovers the eliminated variables (and their duals) from the
reduced solution.

`discopt-aggregation` is a standalone **plugin for
[discopt](https://github.com/jkitchin/discopt)** that installs into the
`discopt.aggregation` namespace. It implements Naik, Biegler, Bent & Parker,
*Variable aggregation for nonlinear optimization problems*
([arXiv:2502.13869](https://arxiv.org/abs/2502.13869)).

## Quickstart

```python
import discopt.modeling as dm
from discopt.aggregation import aggregate, solve

m = dm.Model("example")
x = m.continuous("x", lb=-5, ub=5)
y = m.continuous("y", lb=-5, ub=5)
m.subject_to(y - (x + 1) == 0.0)          # y is defined by x
m.minimize(x**2 + y**2)

res = solve(m, method="d2")                # aggregate → solve → recover
print(res.status, res.objective)          # optimal 0.5
print(res.x)                              # {'x': -0.5, 'y': 0.5}  (y recovered)
```

<div class="grid cards" markdown>

-   :material-rocket-launch: **Get started**

    ---

    Install the plugin and run your first reduced-space solve.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-function-variant: **Choose a method**

    ---

    Six strategies from conservative fixed-variable elimination to
    approximate-maximum matching.

    [:octicons-arrow-right-24: Methods](methods.md)

-   :material-tune: **Tune & recover**

    ---

    Numerical guards, bound pruning, array support, primal & dual recovery.

    [:octicons-arrow-right-24: Options & recovery](options.md)

-   :material-chart-line: **Study & reproduce**

    ---

    Structural metrics, benchmarks, and reliability sweeps reproducing the
    paper's tables.

    [:octicons-arrow-right-24: Study & benchmarks](study.md)

</div>

## Why aggregate?

A model often carries intermediate variables that are *defined* by an equality
constraint — a discretized DAE state, a recycle stream, an algebraic rate. Each
such definition is a variable **and** a constraint the solver must carry. Substituting
the definition removes both, shrinking the problem the interior-point solver sees
and often improving its conditioning and convergence — at zero cost to the optimum,
because every strategy produces a strictly lower-triangular aggregation set (Lemma 1)
with explicit, solve-free post-solve recovery.

## Features

- **Six strategies**, most conservative to most aggressive: `ld1`, `ecd2`, `ld2`,
  `d2` (default), `gr`, `lm`. See [Methods](methods.md).
- **Numerical safeguards**: a relative pivot tolerance against ill-conditioned
  substitutions, an optional fill-in cap, and derivative-amplification / condition
  guards.
- **Redundant-bound pruning** by interval propagation (on by default).
- **Array support**: fully-indexed array variables are scalarized so aggregation
  reaches discretized (DAE) models, then reassembled on recovery.
- **Primal & dual recovery**, including recursive multi-pass duals.
- **Integer aggregation** (opt-in, beyond the paper): eliminate a scalar
  integer/binary variable when its integrality is *implied*. See
  [Integer aggregation](integer-aggregation.md).
- **Structural metrics & study harnesses** reproducing the paper's Table 4 and the
  reliability figures.

## Citation

> Sakshi Naik, Lorenz Biegler, Russell Bent, Robert Parker.
> *Variable aggregation for nonlinear optimization problems.*
> arXiv:2502.13869 (2025). <https://arxiv.org/abs/2502.13869>
