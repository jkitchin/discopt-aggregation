# Changelog

All notable changes to `discopt-aggregation` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-07-02

Third release: the post-audit review-fixes plan (`design/aggregation-review-fixes-plan.md`).
A four-track adversarial review surfaced correctness bugs, hash-seed
nondeterminism, guard/fidelity gaps, and study-completeness holes; this release
closes them. Soundness fixes are **default-on** (returning `optimal` for an
infeasible model is a bug, not a preserved behavior). The full suite passes on
`PYTHONHASHSEED` 0–9.

### Fixed

- **Infeasibility no longer masked as `optimal`.** A fixed-variable elimination
  whose constant value violates the eliminated variable's box (C1), and an
  over-determined fully-determined system (C2), now return `infeasible` from
  `solve()` (with a warning), instead of solving a reduced model that dropped the
  violated constraint. New `recovery.feasibility_violations` helper.
- **Opaque expression nodes no longer drop hidden variables (C3).** New
  `exprwalk` module is the single DAG-walk authority: it walks `CustomCall`
  (`.args`) and `MatMulExpression` (`.left/.right`), so a variable used only
  inside one is counted and substituted (no dangling references), and a
  genuinely unrecognised node refuses loudly. `scalarize` abstains for an array
  used whole inside such a node. Walkers are memoised by node id, making
  structural analysis `O(distinct nodes)` instead of `O(tree)` — a substitution-
  expanded body can have a multi-million-node tree over a few-thousand-node DAG.
- **`eliminate_implicit` re-expresses eliminated block variables' bounds (C4)** as
  inequalities on the implicit-node components (previously dropped, letting the
  reduced solve violate them).
- **`kkt_schur_indices` uses flat variable offsets (C5),** so a surviving
  (abstained) array shifts later variables' KKT indices correctly.
- **Deterministic matching (C6).** `maximum_matching` / `block_triangularize`
  iterate sorted node/adjacency, so all `lm`-based results (eliminations, Schur
  indices, orderings, implicit-block selection) are reproducible across processes.
- **Multi-pass dual recovery is correct (C7).** `recover_duals_from_result` solves
  the defining-equality multipliers as one linear system (recursive passes can
  couple, so no back-substitution order exists); adds active-eliminated-bound
  detection and a KKT-residual self-check that reports `available=False` (with a
  `reason`) rather than a wrong answer.
- **Sound conditioning estimate (N1):** worst-case recovery-error amplification
  `max_k |∂D_v/∂k|` bounded over boxes, composing pivot *and* coefficients (the
  old `1/|coeff|` product reported 1.0 for `v = 1e6·u`).
- **Exact tearing checks MILP status (N2)** and falls back to greedy on a
  non-optimal solve; **block greedy runs on the block's induced subgraph (N3)**
  (can't steal another block's variable), with the Theorem-2 chain asserted;
  **degree filters count all variables (N4),** so `y = x + i` (integer `i`) no
  longer passes degree-2 (default-on fidelity change); array-exponent derivatives
  return `None` instead of crashing (N6); bound-infinity threshold `1e19 → 9.9e19`
  so real bounds up to `9.9e19` are respected (N7); drop counts accumulate across
  filters (N8).

### Added

- `solve()` forwards `tearing=` / `decomposable=` / `max_def_nodes` /
  `max_condition` (T1); `compare_methods` accepts `d<k>` and computes the
  `original` row on the scalarized model (T2, T3).
- `sweep`: per-cell retention (`CellResult`), `virtual_best` (paper Fig. 12), and
  `sweep_to_csv` (P1); `benchmark.callback_breakdown` — a per-callback cost
  breakdown, the paper's Table-6 analogue (P3); `cstr_dynamic` — a
  discretized-DAE optimal-control test problem (P4); a `slow`-marked
  protocol-scale sweep test (P2).
- `aggregate(..., order="min_fill")` — fill-aware elimination ordering (R1):
  `gr`/`lm` visit constraints and pick variables in ascending estimated-fill
  order, cutting the substituted-expression tree ~40× on `reaction_diffusion`
  (gr: 16871 → ~400 nodes) for the same eliminations. Default `"index"` (no-op).
- **Claude skill**, shipped in the wheel under `discopt/aggregation/skill/`
  (SKILL.md + references + examples), with a `discopt-aggregation-skill` console
  script that installs it into `./.claude/skills/` or `~/.claude/skills/` (Claude
  Code does not auto-discover skills from site-packages). The repo is also a
  Claude Code plugin (`.claude-plugin/plugin.json` + `skills/`); single source of
  truth is the packaged skill, with symlinks for the repo/plugin discovery paths.
- Regression tests for every fix; ~48 new tests. Bibliography corrections and
  additions (Bongartz & Mitsos 2017, SCIP 8, the GPU condensed-KKT line).

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
