# variable-aggregation skill

Guidance for using the `discopt.aggregation` package — variable aggregation
(reduced-space presolve) for nonlinear optimization, implementing Naik, Biegler,
Bent & Parker (arXiv:2502.13869).

- `SKILL.md` — overview, the central structure-preserving-vs-Hessian-cost
  tradeoff, entry points, options table, and task routing.
- `references/methods.md` — the six methods (ld1/ecd2/ld2/d2/gr/lm) + `d<k>`, and
  a decision tree.
- `references/options.md` — every option with recipes (guards, `order="min_fill"`,
  tearing, scalarize, prune_bounds, …).
- `references/troubleshooting.md` — gr/lm blow-up, local-vs-global solve,
  infeasibility, recovery `available=False`, "nothing eliminated".
- `references/recovery-and-implicit.md` — primal/dual recovery, implicit cyclic-
  block elimination, Dulmage–Mendelsohn diagnostics, Schur/ordering.
- `references/study-tools.md` — compare_methods / benchmark_model /
  callback_breakdown / reliability_sweep / test problems (paper Tables 4/6/7).
- `examples/` — runnable scripts (`quickstart.py`, `aggressive_safely.py`).
