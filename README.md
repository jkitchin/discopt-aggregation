# discopt-aggregation

Variable aggregation (reduced-space presolve) for nonlinear optimization —
a standalone **plugin for [discopt](https://github.com/jkitchin/discopt)** that
installs into the `discopt.aggregation` namespace.

It implements Naik, Biegler, Bent & Parker, *Variable aggregation for nonlinear
optimization problems* ([arXiv:2502.13869](https://arxiv.org/abs/2502.13869)):
substitute variables defined by equality constraints into the rest of a model,
eliminating both a variable and a constraint to yield a smaller reduced-space
formulation, then recover the eliminated variables from the reduced solution.

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

## Installation

```bash
uv pip install -e .            # or: pip install -e .
```

Requires `discopt >= 0.5`. The plugin is a `discopt.aggregation` **namespace
package**: it merges into the installed `discopt` package at import time. This
needs `discopt/__init__.py` to extend its search path, i.e. to contain:

```python
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
```

discopt 0.5+ includes this line. (If you use an older discopt, add it once.)

## Claude skill

This package ships a [Claude Code](https://code.claude.com/docs/en/skills) skill
that documents how to use aggregation (method choice, the structure-preserving vs
Hessian-cost tradeoff, every option, recovery, diagnostics, the study harnesses).
Claude Code does **not** auto-discover skills from installed Python packages, so
after `pip install` run the bundled installer once to copy it into a
`.claude/skills/` directory Claude scans:

```bash
discopt-aggregation-skill install            # -> ./.claude/skills/  (this project)
discopt-aggregation-skill install --user     # -> ~/.claude/skills/  (every project)
discopt-aggregation-skill path               # where the packaged skill lives
```

Working *inside this repo*, the skill is already at `.claude/skills/` (no install
needed). The repo is also a Claude Code **plugin** (`.claude-plugin/plugin.json`
+ `skills/`), so it can be loaded with `claude --plugin-dir <repo>` or published
to a marketplace. The single source of truth is
`src/discopt/aggregation/skill/`; the two `.claude/skills` / `skills` entries are
symlinks to it.

## Methods

`method=` selects the strategy, most conservative to most aggressive:

| key    | name                        | eliminates                       | paper |
|--------|-----------------------------|----------------------------------|-------|
| `ld1`  | fixed-variable (degree-1)   | `y = a`                          | Alg 3 |
| `ecd2` | equal-coefficient degree-2  | `y = x + a`                      | Alg 6 |
| `ld2`  | linear degree-2             | `y = a*x + b`                    | Alg 5 |
| `d2`   | degree-2 (**default**)      | `y = f(x)` (≤ 2 variables)       | Alg 4 |
| `gr`   | greedy                      | `y = f(w, x, …)` (`y` linear)    | Alg 1 |
| `lm`   | linear-matching             | `y = f(w, x, …)` (`y` linear)    | Alg 2 |

`ld1`/`ecd2`/`ld2`/`d2` are structure-preserving; `gr`/`lm` are
approximate-maximum. The paper recommends `d2`; it is the default.

## Features

- **Six strategies**, each producing a strictly lower-triangular aggregation set
  (Lemma 1) with explicit, solve-free post-solve recovery.
- **Numerical safeguards**: a relative pivot tolerance (`pivot_tol`) against
  ill-conditioned substitutions, and an optional fill-in cap (`max_fill`).
- **Redundant-bound pruning** (`prune_bounds`, on by default): box→inequality
  constraints proven redundant by interval propagation are omitted.
- **Array support** (`scalarize`, on by default): fully-indexed array variables
  are scalarized so aggregation reaches discretized (DAE) models, then arrays
  are reassembled on recovery.
- **Structural metrics** (`compare_methods`) reproducing the paper's Table 4,
  including the exact AD Hessian nonzero count.

## Examples

`python -m discopt.aggregation.examples` runs commented, executable walkthroughs
of every method, the recursive pipeline, end-to-end solve + recovery, the
structural comparison, the safeguards, array scalarization, and a
performance/scaling demo.

## Tests

```bash
uv pip install -e ".[dev]"
python -m pytest tests/ -q          # the full suite
```

## Layout

```
src/discopt/aggregation/    # the plugin (PEP 420 namespace subpackage)
  core:        matching · incidence · strategies · transform · recovery · api
  walking:     exprwalk (single DAG-walk authority for all analyses)
  structure:   scalarize · metrics · diagnostics (Dulmage–Mendelsohn) · ordering
  numerics:    guards · hessian_cost · duals · derivatives · implicit · tearing · schur
  study:       testproblems · benchmark · sweep · profiling · presolve · examples
tests/                      # test_aggregation_*.py
design/                     # follow-up roadmap · parity plan · references.bib
```

## References

Primary:

> Sakshi Naik, Lorenz Biegler, Russell Bent, Robert Parker.
> *Variable aggregation for nonlinear optimization problems.*
> arXiv:2502.13869 (2025). <https://arxiv.org/abs/2502.13869>

The full set of related work — reduced-space / implicit elimination
(Parker et al. 2022), block-triangular KKT (Parker, Garcia & Bent 2026), exact
tearing / minimum feedback set (Baharev et al. 2021), Dulmage–Mendelsohn
debugging (Parker et al. 2023), and the MIP-presolve line (Achterberg et al.
2020) — is collected as verified BibTeX in
[`design/references.bib`](design/references.bib), and cited from the design docs
under [`design/`](design/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the lint/type/test gate, and
the project conventions (keep discopt/pounce/feral untouched, new behavior
opt-in/default-off, a worked example per feature, hash-seed-robust tests).

## License

Eclipse Public License 2.0 (EPL-2.0), matching discopt.
