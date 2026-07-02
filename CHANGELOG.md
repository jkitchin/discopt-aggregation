# Changelog

All notable changes to `discopt-aggregation` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-07-02

Second release: the post-Stage-1 follow-up plan (paper-derived extensions,
empirical study, numerical/structural tooling). All non-upstream-gated parity
items are complete; the remaining work is gated on upstream features
(`discopt#379`, `discopt#383`, `feral#107`, `pounce#180`).

### Added

- **Strategies.** Degree-`k` family (`d3`, `d4`, …) generalising `d2`.
- **Exact tearing of cyclic blocks** (E3): brute-force maximum-acyclic-subset
  (`tearing="exact"`) for small blocks, and a **scalable lazy-cycle MILP** tear
  (S2, `tearing.py`) for larger blocks — minimum feedback vertex set via lazy
  cycle enumeration solved with discopt's own MILP solver (exact past the
  brute-force cap; greedy fallback never degrades).
- **Decomposable matchings** (E4): `aggregate(..., decomposable=True)`
  local-searches the linear-matching for a higher Theorem-2 block count.
- **Implicit-block primitive** (E5, `ImplicitBlock`): Newton solve of a
  nonsingular cyclic block with exact implicit-function-theorem derivatives;
  plus **cyclic-block extraction** (K1, `implicit_blocks`) and the guarded
  reduced-model rewrite `eliminate_implicit` (awaits `discopt#379`).
- **Block-triangular / Schur KKT ordering emitter** (K2,
  `block_triangular_ordering`): emits the LM submatrix's block-lower-triangular
  permutation for a future KKT ordering hint (awaits `feral#107`/`pounce#180`).
- **Recursive-path dual recovery** (D1, `recover_duals_from_result`,
  `solve(..., recover_duals=True)`): composes per-pass constraint-origin maps to
  recover full-space multipliers across all aggregation passes; certified by the
  KKT stationarity residual.
- **Expression-graph / Hessian-cost diagnostics** (R2, `hessian_cost.py`):
  `expression_tree_size`, `expression_dag_size`, `expression_depth`,
  `sharing_ratio`, `model_expression_size` (root-caused the gr/lm slowdown →
  `discopt#383`).
- **Build-time profiling at scale** (R3, `profiling.py`): `profile_build`,
  `build_scaling`, `indep_scaling_model`, `balanced_sum`.
- **Elimination guards** (D2, `guards.py`): opt-in `max_def_nodes`
  (expression-size cap) and `max_condition` (chain-conditioning cap), plus
  `conditioning_estimate`.
- **Dulmage–Mendelsohn structural diagnostics** (S1, `diagnostics.py`,
  `matching.dulmage_mendelsohn`): over-/under-determined/square blocks and
  structural-singularity warnings.
- **Native-presolve overlap audit** (S3, `presolve.py`):
  `native_presolve_overlap`, `recommended_ordering`, `verify_composition`
  (composition with discopt's native presolve is exact).
- **Target-linearity filter** (`preserve_target_linearity`) and additional
  **numerical safeguards**.
- **Parameterized test problems** (`reaction_diffusion`, `unit_selection`,
  `gas_pipeline`, `recycle_loop`), **runtime benchmark** (`benchmark_model`),
  and **reliability sweep** (`reliability_sweep`).
- `__version__` on the package; worked examples for every feature (wired into
  `python -m discopt.aggregation.examples`); test suite grown to 300+.
- `design/references.bib` — verified bibliography of the related literature.

### Notes

- Default behaviour is unchanged: every new guard/option is opt-in and
  default-off; `d2` remains the default method and `tearing="greedy"` the default
  tear.
- discopt/pounce/feral source remains untouched apart from the one-line
  namespace shim in discopt's `__init__.py`.

## [0.1.0] - 2026-07-01

### Added

- Initial standalone release of the variable-aggregation plugin for discopt,
  installing into the `discopt.aggregation` namespace. Implements Naik, Biegler,
  Bent & Parker, *Variable aggregation for nonlinear optimization problems*
  (arXiv:2502.13869).
- Public API: `aggregate(model, method=...)` and `solve(model, method=..., **kw)`
  with the recursive Section 3.3 pipeline; `compare_methods` structural harness.
- Six selection strategies — `ld1`, `ecd2`, `ld2`, `d2` (default), `gr`, `lm` —
  each producing a strictly lower-triangular aggregation set (Lemma 1) with
  explicit post-solve recovery (no iterative solve).
- Graph kernel (`maximum_matching`, `strongly_connected_components`,
  `block_triangularize`), incidence analysis, expression substitution/recovery,
  and structural metrics (`model_structure`, `compare_methods`, exact AD Hessian
  nonzero count).
- Numerical safeguards: relative pivot tolerance (`pivot_tol`) and fill-in cap
  (`max_fill`).
- Redundant-bound pruning via sound interval propagation (`prune_bounds`,
  default on).
- Array-variable scalarization (`scalarize`, default on) so aggregation reaches
  discretized/indexed models, with array reassembly on recovery; plus a guard
  for fully-determined (empty reduced) models.
- Runnable examples (`python -m discopt.aggregation.examples`) and a 200-test
  suite.

### Requires

- `discopt >= 0.5` (whose `discopt/__init__.py` extends `__path__` via
  `pkgutil.extend_path`, letting this namespace plugin merge in).
