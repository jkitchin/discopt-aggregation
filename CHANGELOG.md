# Changelog

All notable changes to `discopt-aggregation` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-07-03

Fourth release: the NLP-gaps + integer-aggregation plan
(`design/aggregation-gaps-and-integer-plan.md`). Closes the scaling bottleneck
that blocked the aggressive methods at DAE scale (SC1), completes dual recovery
for maximize and active-bound cases (DU1/DU2), adds iteration-cap reliability
plumbing with an honest-negative probe (RL1/RL2), and introduces **opt-in integer
variable aggregation** (IA1-IA6). Soundness/scaling fixes are default-on; integer
aggregation is default-off and a proven no-op with the flag off.

### Added

- **Integer variable aggregation (opt-in, IA1-IA6).** `aggregate(...,
  integer_aggregation=True)` / `solve(...)` also eliminates scalar integer/binary
  variables when integrality is *implied* -- the standard MIP-presolve rule
  (Achterberg et al. 2020): fully-affine defining equality, pivot `+-1`, all other
  participants integer/binary, integral other coefficients and constant. A fixed
  integer set to a fractional value is reported infeasible (via the existing C1
  path); box->inequality conversion tightens to the integer grid (IA3); recovered
  eliminated integers are asserted integral and returned rounded (IA4, a
  non-integral result raises rather than shipping a wrong value); dual recovery
  declines over eliminated integers (`reason="eliminated_integer_variable"`). New
  `inventory_chain` MINLP test problem: an all-integer stock chain collapses
  end-to-end and round-trips to the same optimum with integral values (IA5).
  **Default-off and provably a no-op with the flag off** -- eliminated lists on
  the whole corpus are byte-identical, and `unit_selection` binaries are never
  eliminated even flag-on. Worked example `example_integer_aggregation` (IA6).
- **Iteration-cap plumbing for local solves (RL1).** The study harness can now
  count convergence *within an iteration limit* — the paper's reliability
  protocol (3000 iterations). A new `options: dict | None = None` argument threads
  through `benchmark._local_solve` and `benchmark_model`, and
  `reliability_sweep(..., solver_options=None)` forwards it to every local solve
  (`solve_nlp_from_model(options=...)`, e.g. `{"max_iter": 3000}`). A run that
  stops at the cap reports a non-`OPTIMAL` status (`ITERATION_LIMIT`) and so
  counts as **not converged**. Default (`options=None`/`solver_options=None`)
  leaves solves uncapped and behavior unchanged.
- **RL2 reliability probe `recycle_cycle` (an honest negative).** A bounded search
  for a bundled instance demonstrating the paper's "aggregation converges more
  instances" headline found **no sound demonstration** on this stack, so no
  `HARD_INSTANCE` is registered. The searched closed-recycle builder is shipped as
  `recycle_cycle` / `RECYCLE_CYCLE` (a `TestProblem`, deliberately **not** in
  `PROBLEMS`) for reproducibility. Findings, pinned as regression tests
  (`tests/test_aggregation_reliability.py`) and written up in
  `design/aggregation-stage2-results.md`: explicit `d2`/`lm` never converge more
  instances (on the closed recycle `d2` is *sound* but needs *more* Newton
  iterations, a reliability loss); the implicit path (`eliminate_implicit`) shows
  a large *raw* convergence gap under a tight cap, but it is **confounded** by root
  multiplicity — the bound-unaware inner Newton recovers a spurious negative root
  (`R = -3.86` vs the full space's `R = 9.33`) on ~43% of the grid, so it reports a
  different, sometimes unphysical solution rather than a genuine reliability win.

### Fixed

- **Dual recovery now supports maximize objectives (DU1).** `recover_duals` /
  `recover_duals_from_result` previously returned `available=False,
  reason="maximise"`. discopt reports a maximize objective's multipliers in the
  `minimize(-f)` convention (verified empirically), so recovery now normalises the
  objective-gradient sign (`obj_sign = -1` for maximize) in both the defining-
  multiplier solve and `kkt_stationarity_residual`. Recovered duals match a direct
  solve of the maximize model to 1e-6 and pass the KKT self-check; minimize-path
  behavior is unchanged. Default-on.
- **Dual recovery now handles active eliminated-variable bounds (DU2).** An
  eliminated variable sitting on a finite bound at the optimum previously returned
  `available=False, reason="active_eliminated_bound:<name>"`. The box→inequality
  emitted for that bound survives in the reduced solve and carries a multiplier;
  by substitution invariance it *equals* the base-model bound multiplier γ, so
  recovery now collects it (`-μ` for a low bound, `+μ` for a high bound, matching
  discopt's `lb - v` / `v - ub` normalisation) into a new
  `DualRecovery.bound_duals: {var: γ}` field, feeds each γ into the defining-
  multiplier system and the (extended) `kkt_stationarity_residual`, and reports
  `available=True`. The active-bound repro now recovers the defining-equality
  multiplier (≈ -2.536) with KKT residual < 1e-6. Interior/pruned bounds
  contribute γ ≈ 0; a still-active bound with no observable multiplier stays
  `available=False` defensively. Default-on.

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
