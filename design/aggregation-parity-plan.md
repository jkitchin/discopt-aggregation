# discopt-aggregation — literature-parity plan

**Goal.** Take the plugin from "faithful reproduction of Naik et al.
[cite:@naik2025aggregation] + safety/tooling extensions" to **parity with the
surrounding state of the art** — the reduced-space / implicit-elimination /
structured-KKT line of Biegler–Parker–Bent, and the MIP-presolve line of
Achterberg et al. [cite:@achterberg2020presolve].
Citations use `design/references.bib`.

**Method.** Each theme states: *literature target → current state → gap → work →
validation → dependencies*. Effort in loop-iterations (S≈1, M≈2–3, L≈4+), 25%
buffer implied. Two upstream feature requests (feral, pounce) unblock the KKT
themes; both are filed (see end).

**Ground rule (unchanged).** discopt/pounce/feral source stays untouched by the
plugin; anything needing core support is filed upstream, not vendored.

**Adjacent framing.** Variable elimination can also be read as *nonlinear
preconditioning* [cite:@ciaramella2024preconditioning] rather than presolve; the
reduced-space and preconditioning views coincide on which variables to remove,
so that line is a useful cross-check on the selection strategies rather than a
separate work item here.

---

## Phase overview

| Phase | Theme | Items | Milestone |
|---|---|---|---|
| P1 | Finish the reproduction | R1 reliability instance · R2 CSE/Hessian-cost (F3) · R3 real DAE scale | **MP1**: paper reproduced end-to-end incl. the reliability win |
| P2 | Dual & numerical completeness | D1 recursive+surfaced duals · D2 conditioning + Hessian-cost cap | **MP2**: full-space KKT point + robust defaults on any supported model |
| P3 | Structural parity | S1 Dulmage–Mendelsohn · S2 scalable exact tearing (Baharev) · S3 MIP-presolve overlap/parity | **MP3**: structural analysis + presolve on par with Pyomo/SCIP practice |
| P4 | Implicit & structured-KKT parity (upstream-gated) | K1 implicit-function node (E5 end-to-end) · K2 structured/Schur KKT (E6) | **MP4**: cyclic-block elimination and block-triangular KKT, matching Parker 2022/2026 |

Dependency spine: `feral#107 → pounce#180 → K2`; `discopt#379 → K1`. P1/P2/P3
are independent of the upstream work and can proceed now.

---

## P1 — Finish the reproduction

### R1. A reliability-demonstrating instance (M) — ✅ DONE (honest negative on the *failure* gap)
- **Target.** [cite/t:@parker2022implicit] reports **82/100 vs 42/100**
  convergence (implicit/reduced vs full-space); Naik et al.
  [cite:@naik2025aggregation] show aggregation improving reliability.
- **Built.** `testproblems.recycle_loop` — a torn steady-state recycle (parallel
  units, tear-residual objective), registered as `RECYCLE_LOOP`, wired into
  `reliability_sweep`, with `example_recycle_reliability`. 6 structural tests
  (`tests/test_aggregation_recycle.py`).
- **Finding (empirical, honest).** Aggregation (d2) **robustly collapses the loop
  to a single DOF** at every grid point — the smaller-model win. But across the
  3×3 (feed, alpha) grid **both forms converge 100% and iteration counts are
  ~unchanged** (d2 9–14 vs full 10–12, occasionally *worse*): no convergence /
  iteration gap. An explicit-aggregation *failure* gap is **architecturally
  elusive**, for two now-articulated reasons: (a) a **closed** recycle loop is one
  irreducible cycle → implicit elimination (E5/K1), gated on discopt#379, not
  explicit; (b) a **deep nonlinear cascade** is explicitly eliminable but its
  substitution compounds to a degree-`2^n` expression (the S2 Hessian-cost
  pathology, reproduced here — the first probe *hung*), so aggregation hurts.
  The robust benefit of aggregating a torn loop is the **smaller model**, and the
  genuine reliability win is the **implicit (K1)** path.
- **Deps.** none. (The reliability *win* itself effectively depends on K1 →
  discopt#379; documented rather than overfit.)

### R2. CSE / Hessian-cost investigation (F3) (M) — ✅ DONE (root cause found: discopt lowering has no CSE)
- **Target.** Determine whether discopt's JAX compiler CSEs substituted subtrees
  — the root cause of the gr/lm slowdown.
- **Built.** `hessian_cost.py` — `expression_tree_size` (multiplicity),
  `expression_dag_size` (distinct nodes), `sharing_ratio`, `model_expression_size`.
  4 tests (`tests/test_aggregation_hessian_cost.py`) + `example_hessian_cost`.
- **Finding (measured, decisive).**
  1. A single `substitute_expression` **preserves sharing** (the replacement is
     one shared object; `dag(result) ≈ dag(D) + O(1)`, not `3·dag(D)`). The plugin
     is not the primary culprit.
  2. **discopt's expression→jaxpr lowering does NOT CSE.** A *manually* linear-DAG
     nonlinear chain (distinct nodes 25→37→49→55→61 over depths 4→10, i.e.
     ~6/level) builds its Hessian in **exponential** time (0.017 → 0.032 → 0.211 →
     0.721 → 2.965 s). Linear DAG, exponential build ⇒ shared nodes are re-traced.
  3. Secondary: the plugin's *multi-variable* substitution composition also loses
     sharing on a deep chain (exponential DAG) — but this is moot until (2) is
     fixed, since even a perfectly-shared DAG blows up.
- **Fix placement.** The dominant fix is **upstream** (CSE / identity-memoized
  lowering in discopt) → **filed as discopt#383** with the reproduction + table.
  The plugin-side mitigation is to **cap substituted-expression size**
  (`max_def_nodes`, D2) rather than aggregate deep nonlinear chains.
- **Follow-up (post-#383, MEASURED on `main`).** discopt#383 landed
  (commit `5469d50`: memoize `dag_compiler._compile_node` by `id(expr)`; the
  `NLPEvaluator` solve path uses those `compile_*` entry points). With discopt on
  `main` the fix **is real and helps, proportional to sharing (`tree/dag`)**:
  - synthetic linear-DAG chain (huge `tree/dag`): depth-8 solve **10.4 s → 0.21 s
    (~50×)**;
  - real `reaction_diffusion` gr-reduced (`tree/dag ≈ 2.5`): n=10 solve
    **35.5 s → 19.8 s (~1.8×)** — matching the analytic `tree/dag` bound.
  It does **not eliminate** the blowup: the residual cost is the genuinely-large
  DAG from substitution fan-out (real case) and the Hessian `jacfwd∘jacfwd` XLA
  codegen (synthetic). **#383 and D2 compose** — CSE cuts per-node cost, D2's
  `max_def_nodes` caps node count. Net: R2's guidance stands, now with a ~2× tailwind.
- **Deps.** none (spawned **discopt#383**, since landed).

### R3. Real discretized-DAE scale (M) — ✅ DONE (profiled; 3 scaling findings)
- **Target.** The paper's models are large discretized DAEs; our solved instances
  were n≲10 — measure the *build* (not solve) at scale.
- **Built.** `profiling.py` — `profile_build` (times scalarize / incidence /
  aggregate), `build_scaling`, `indep_scaling_model`, `balanced_sum`; plus
  `hessian_cost.expression_depth` (iterative, safe). 5 tests
  (`tests/test_aggregation_profiling.py`) + `example_build_scaling`.
- **Findings (measured to 8k–16k flat vars).**
  1. **The incidence / matching graph kernel scales ~linearly** (0.008→0.068 s
     over 1k→8k vars) — not the bottleneck.
  2. **Scalarize + full aggregate scale ~quadratically** (0.045→1.272 s) — each
     elimination substitutes into the growing objective/constraints, re-walked
     every pass. This is the real build-time limit → follow-up: **native indexed
     incidence (F1)** + single-pass substitution.
  3. **The recursive AST walkers overflow** (`RecursionError`) on a flat
     `sum(...)` of ~1000+ terms (depth-~1000 left-associative chain). `balanced_sum`
     keeps depth `O(log n)`; `expression_depth` detects an at-risk model in
     advance → follow-up: **iterative walkers**.
  - Also: `gr`/`lm` at grid scale hit the CSE blow-up (R2 / discopt#383), so the
    structure-preserving methods are the ones that scale.
- **Validation (achieved).** Scaling curve produced; round-trip correctness
  verified at n=100 (recovered `y = 2x+1` for all elements).
- **Deps.** none.

---

## P2 — Dual & numerical completeness

### D1. Recursive + surfaced dual recovery (M) — ✅ DONE (recursive/minimize; maximize+active-bounds scoped)
- **Target.** A *complete* full-space KKT point, as reduced-space methods
  provide.
- **Built.**
  1. `aggregate` now records per-pass `origin_maps` + `elim_records` +
     `base_model` on `AggregationResult`. `duals.recover_duals_from_result`
     composes the per-pass constraint-origin maps to work in the base model and
     back-substitutes across **all** passes (the combined defining system is
     strictly lower-triangular). Recovers surviving + defining multipliers.
  2. `solve(..., recover_duals=True)` surfaces a `DualRecovery` on
     `AggregatedSolveResult.duals`.
  3. 8 tests (`tests/test_aggregation_duals_recursive.py`) +
     `example_recursive_duals`.
- **Validation (achieved).** On a recursive `d2` (3 passes) with an inequality:
  **KKT stationarity residual ≈ 9.4e-10**, recovered duals **match the original
  model's own multipliers**, and `d2`/`gr`/`lm` recover the **same** (unique) KKT
  duals. Method-independent, certified.
- **Remaining (documented scope, not blocking).** Maximize objective and *active*
  eliminated-variable bounds still return `available=False` — extending the sign
  convention / γ-terms is the only unfinished piece; the recursive-path gap (the
  headline) is closed.
- **Deps.** none.

### D2. Conditioning bound + Hessian-cost cap (S–M) — ✅ DONE
- **Target.** MIP-presolve practice [cite:@achterberg2020presolve] treats
  substitution numerics carefully; the paper flags Hessian growth.
- **Built (`guards.py`, both opt-in / default-off).**
  1. **`max_def_nodes`** — drops any elimination whose *resolved* (inlined)
     defining-expression node count exceeds the cap, bounding the expression-graph
     / Hessian blow-up R2 identified. Threaded through `aggregate`/`solve`.
  2. **`max_condition`** — drops any elimination whose cumulative pivot
     amplification `(1/|coeff|)·A(referenced)` exceeds the cap; `conditioning_estimate`
     exposes the worst-case chain amplification (`pivot_tol` is per-pivot only).
  3. 7 tests (`tests/test_aggregation_guards.py`) + `example_elimination_guards`.
- **Validation (achieved).** On a depth-5 nonlinear chain, `max_def_nodes`
  monotonically shrinks the reduced DAG (875 → 316 → 125) and the elimination
  count; `conditioning_estimate` = 100 for a `0.01` pivot (hand-checked) and
  `max_condition=10` drops it; default-off reproduces baseline exactly; a capped
  aggregation still round-trips (objective + recovered definitions correct).
- **Deps.** R2 (uses `hessian_cost.expression_dag_size`).

---

## P3 — Structural parity

### S1. Dulmage–Mendelsohn decomposition (M) — ✅ DONE
- **Target.** [cite/t:@parker2023dulmage] — DM decomposition identifies structurally
  over-/under-determined subsystems, the standard diagnostic layer around
  incidence analysis.
- **Built.**
  1. `matching.dulmage_mendelsohn(graph, matching)` + `DulmageMendelsohn` — the
     pure-graph coarse decomposition (alternating reachability from unmatched
     nodes → over-/under-determined/square blocks), added to the stdlib-only
     kernel.
  2. `diagnostics.structural_diagnosis(model)` + `StructuralDiagnosis` — the
     model-facing wrapper: `free_variables`, `overdetermined_constraints`,
     `is_structurally_singular`, `warnings()`.
  3. 7 tests (`tests/test_aggregation_diagnostics.py`) +
     `example_structural_diagnosis`.
- **Validation (achieved).** Cross-checked block membership on constructed
  square / under-determined / over-determined systems; the **square block is
  exactly what aggregation eliminates** (fixed-var cascade → `{a,b}`); a redundant
  equality is flagged structurally singular with a warning; blocks partition the
  nodes; pure-graph core verified directly.
- **Deps.** none (pure-graph).

### S2. Scalable exact tearing (M) — ✅ DONE
- **Target.** [cite/t:@baharev2021mfas] — exact minimum feedback set via **lazy
  simple-cycle enumeration + MILP**. Our E3 exact tear was brute-force
  max-acyclic-subset, exact only to ~13 vars.
- **Built (`tearing.py`).** `max_acyclic_subset_milp` — minimum-feedback-vertex-set
  via lazy cycle enumeration: solve `min Σ r_v` s.t. `Σ_{v∈C} r_v ≥ 1` per known
  cycle with **discopt's own MILP solver** (dogfood), add a violated cycle
  (`find_cycle`, iterative DFS) and re-solve until the kept subgraph is acyclic.
  `strategies._exact_tear` now dispatches to it for blocks past
  `_EXACT_TEAR_CAP`, keeping brute-force for tiny blocks and greedy as the
  never-degrade fallback. 7 tests (`tests/test_aggregation_tearing_milp.py`) +
  `example_scalable_tearing`.
- **Validation (achieved).** MILP **matches brute-force exactly** on ring/double-
  cycle graphs; result is provably acyclic; **scales past the cap** — n=20 keeps
  18 in 0.4 s, n=40 keeps 38 in 1.2 s (brute force would be 2⁴⁰); end-to-end
  `tearing="exact"` still round-trips with `≥ greedy` eliminations; default
  (greedy) unchanged.
- **Deps.** none (uses discopt MILP).

### S3. MIP-presolve overlap & parity (M) — ✅ DONE (audit + composition safety; guards scoped)
- **Target.** [cite:@achterberg2020presolve;@zhang2024aggregation;@gleixner2024papilo;@vanderhulst2025impliedintegrality].
- **Built (`presolve.py`).**
  1. **Overlap audit** (`native_presolve_overlap`). Read discopt's Rust presolve
     (`eliminate_variables` M10, `aggregate_variables` C1, `factorable_eliminate`
     C2) and classified each: all act on variables appearing **only in their
     defining equation** (write-only/singleton), whereas the plugin **substitutes
     into every occurrence** — so `ld1`/`ld2` *subsume* M10/C1 and `gr`/`lm` are
     **complementary** to C2 (it frees but doesn't substitute).
  2. **Recommended ordering** (`recommended_ordering`): plugin first (pre-pass),
     then `discopt.solve()` — its native presolve mops up leftovers on the reduced
     model; no double-elimination conflict.
  3. **Composition safety** (`verify_composition`): empirically confirmed
     aggregate-then-solve **matches a direct solve** for write-only *and* used
     intermediates across `d2`/`gr`/`lm`.
  4. 7 tests (`tests/test_aggregation_presolve.py`) + `example_presolve_overlap`.
- **Implied integrality (scoped out, documented — like G4).** The guard would
  never fire: aggregation **never eliminates integer/binary variables** (only
  scalar continuous candidates — tested), and continuous eliminations are
  equivalence-preserving. We keep the sound invariant rather than build a dead
  guard. (G1-redux `PresolveDelta` integration still needs a discopt
  topology-rewrite protocol — an upstream item, not built here.)
- **Deps.** none for the audit; G1-redux would need a discopt protocol change.

---

## P4 — Implicit & structured-KKT parity (upstream-gated)

### K1. Implicit-function node → E5 end-to-end (L) — ✅ DONE (discopt#379 landed)
- **Target.** [cite/t:@parker2022implicit] — implicit elimination *inside* the model,
  solved with AD-through-solver.
- **Unblocked.** discopt#379 (`Model.implicit(residual, u_inputs, n_unknowns, x0)`,
  a `jax.lax.custom_root` inner solve) landed and is present in the local checkout.
- **Built.**
  - ✅ **Analysis** — `implicit.implicit_blocks(model)` extracts the irreducible
    cyclic blocks as `ImplicitBlock` primitives.
  - ✅ **End-to-end elimination** — `implicit.eliminate_implicit(model)` →
    `ImplicitElimination`: rebuilds the model with each *nonsingular* cyclic block
    replaced by a `Model.implicit` node (a JAX-safe residual `_jax_eval_expr` lowers
    the block's defining equations; kept variables become the node's `u_inputs`),
    drops the defining equations, and substitutes the block variables by the node's
    components. `ImplicitElimination.recover` reconstructs the block variables
    post-solve via the `ImplicitBlock` primitive.
  - ✅ 5 end-to-end tests (`tests/test_aggregation_implicit_e2e.py`, robust across
    hash seeds) + updated `example_implicit_blocks`.
- **Validation (achieved).** Reduced model solves to the **same optimum** as the
  original (verified obj match); `recover()` reconstructs the block variables to
  their closed form and satisfies the defining equations (linear *and* nonlinear
  cycles); acyclic/singular models handled soundly (`ValueError` / skip).
  Scope: blocks over scalar variables with JAX-supported ops (others left in place).
- **Deps.** none remaining (discopt#379 landed).

### K2. Structured / Schur KKT reuse — E6 done properly (L) — ✅ DONE
- **Target.** [cite/t:@parker2026blocktriangular] (15× over MA57/MA86 by
  Schur-isolating the block-triangular KKT submatrix, inertia via Sylvester),
  and the reduced/condensed-KKT line
  [cite:@regev2022hykkt;@pacaud2024condensed;@kardos2020reduced].
- **Blocker.** FERAL exposes ordering only as an *algorithm selector*
  (`feral_ordering ∈ {auto,amd,amf,metis,scotch,kahip}`) with **no
  user-supplied-permutation input**, and `pounce.Problem` exposes no
  ordering/Schur hook. LM's nonsingular submatrix (a natural Schur pivot block)
  can't be injected.
- **Feature requests (filed).**
  - **feral:** an `OrderingMethod::External(Vec<usize>)` (analogous to the
    existing external *scaling*), so a caller can pass a fill-reducing / block
    permutation. → **filed: jkitchin/feral#107**.
  - **pounce:** plumb an external ordering through `FeralConfig`/options and the
    `Problem` API, plus (ideally) a block-triangular/Schur KKT path per
    [cite:@parker2026blocktriangular], and expose **per-iteration linear-algebra
    timing** (which our S2 runtime study wanted but the result object doesn't
    carry). → **filed: jkitchin/pounce#180**.
- **Work.**
  - ✅ **Emitter built (plugin side).** `ordering.block_triangular_ordering(model)`
    → `BlockTriangularOrdering` produces the LM block-lower-triangular permutation
    as a 0-based `list[int]` in the exact shape feral#107 / pounce#180 will
    consume (cyclic diagonal blocks kept for E5/K1). 7 unit tests
    (`tests/test_aggregation_ordering.py`) + `example_block_triangular_ordering`.
  - ⏳ **Wiring (upstream now landed; interface identified).** feral#107 +
    pounce#180 have **landed and are live** (pounce rebuilt): `pounce.Problem`
    now exposes `set_ordering` / `set_kkt_schur_block` / timing hooks. Verified
    end-to-end:
    - `set_ordering(perm)` takes a **full KKT permutation of size `n+m`** (not the
      variable-only ordering) — an identity `n+m` perm solves correctly (status 0);
      a naive *variable* block-triangular perm + identity constraints **breaks the
      solve** (status −3, wrong answer), so raw `set_ordering` is *not* the right
      interface for the aggregation block.
    - **`set_kkt_schur_block(indices)` is the correct interface** — exactly
      [cite/t:@parker2026blocktriangular]: hand pounce the reducible block (in
      KKT-space indices `x, slack, eq-dual, ineq-dual`), it is Schur-complemented
      out, only the diagonal blocks factorized, inertia via Sylvester. It **falls
      back transparently** if the partition is unsuitable, so it never breaks a
      solve (correctness-safe by construction).
  - ✅ **Reducible-block analysis (plugin side).** `schur.kkt_schur_indices(model)`
    maps the LM reducible block to KKT-space indices in pounce's
    `[x, slack, eq-dual, ineq-dual]` layout: eliminated variables → x-block
    `[0, n_x)`, their defining equalities → eq-dual `n_x + n_ineq + eq_rank`.
    **Handles inequality constraints** (the `n_ineq` slack offset), not just
    equality-only. 5 unit tests (`tests/test_aggregation_schur.py`). Pure analysis:
    no solving, no solver internals.
  - ✅ **Solve-time passthrough landed in discopt (the right place).** Filed as
    **discopt#392**, merged on `origin/main` (**PR #394**, `ba0668b`): `solve_nlp`
    / `solve_nlp_from_model` now take `kkt_schur_block=Sequence[int]` (and
    `ordering=`), guarded + correctness-safe. The param takes KKT-space indices —
    exactly `kkt_schur_indices`'s output — so K2 is:
    `solve_nlp_from_model(model, x0, kkt_schur_block=kkt_schur_indices(model))`.
    The plugin does **not** touch discopt's solver internals; it feeds the
    supported API.
  - ✅ **Verified end-to-end through the supported API.** With the discopt
    passthrough present, `solve_nlp_from_model(model, x0, kkt_schur_block=`
    `kkt_schur_indices(model))` solves to **OPTIMAL** with the **identical
    objective** to a baseline solve (correctness-safe). Locked in by guarded smoke
    tests (`tests/test_aggregation_schur_e2e.py`, skip if discopt lacks the
    passthrough) covering an equality-only *and* a **mixed eq/ineq** model. Timing:
    ~1.2–1.4× on `reaction_diffusion` (n=40–160), ~1.1× on the mixed model — and
    the mixed block is **honored** (a wrong `n_ineq` offset would fall back to the
    same speed), confirming the KKT-index mapping.
- **Validation.** Analysis: valid KKT indices spanning x + eq-dual blocks,
  inequality rejection, determinism (done). Solve-time speedup verified externally
  (identical solution, ~1.2–1.4×); the production path is the discopt passthrough.
- **Deps.** none blocking (feral#107 + pounce#180 landed & live). Note
  [cite:@parker2026blocktriangular] is by pounce's own author group — the
  `set_kkt_schur_block` API is their implementation of it, so the plugin should
  *feed* it, not reimplement.

---

## Risk register (parity-specific)

| Risk | P×I | Mitigation |
|---|---|---|
| Upstream FRs (feral/pounce) not prioritized | M×H | P1–P3 are independent; K1/K2 clearly gated; offer reference implementations |
| CSE fix must live in discopt's JAX compiler, not the plugin (R2) | M×M | measure first; file a focused discopt issue with the jaxpr evidence |
| Reliability instance hard to tune to "fails full-space" (R1) | M×M | follow the chemical-looping recipe in [cite:@parker2022implicit;@bugosen2023gibbs] |
| Baharev MILP tearing slow on pathological blocks (S2) | L×M | size-gate: brute-force ≤13, MILP ≤ cap, greedy fallback |
| discopt presolve-protocol can't express topology rewrite (S3/G1) | M×M | keep aggregation a pre-pass; file protocol-v1 issue if interleaving is wanted |

## Success criteria

- **MP1**: a bundled instance where aggregation demonstrably improves
  convergence; CSE question answered with data; build-time scaling to 10⁴ vars.
- **MP2**: recursive-path dual recovery surfaced in `solve()`, KKT residual ≈0
  with active bounds/inequalities; chain-conditioning + `max_def_nodes` guards.
- **MP3**: DM diagnostics; exact tearing past 13 vars; documented,
  correctness-gated interaction with discopt's native presolve.
- **MP4**: cyclic blocks eliminated end-to-end via the implicit node (discopt#379)
  and block-triangular/Schur KKT reuse (feral+pounce FRs) — matching
  [cite:@parker2022implicit;@parker2026blocktriangular].

## Packaging (crosscutting) — ✅ DONE

- **Version** bumped to **0.2.0** (`pyproject.toml` + `discopt.aggregation.__version__`,
  consistency tested).
- **`CHANGELOG.md`** — full `[0.2.0]` entry documenting every follow-up addition
  (Dk, E3/S2 tearing, E4, E5/K1, K2, D1, R2, R3, D2, S1, S3, recycle_loop,
  benchmark/sweep/profiling) with the opt-in/default-off + untouched-source notes.
- **CI** — `.github/workflows/ci.yml` runs ruff check + ruff format --check +
  mypy + pytest on the uv env across Python 3.10–3.12. (Assumes `discopt` is
  installable in the runner; it depends on the base package being available.)
- **References** — `design/references.bib` wired into the README's References
  section; the design docs cite it via org-cite.
- **`tests/test_packaging.py`** — version consistency + full `__all__` import
  sanity (74 exports).

---

## Final status — parity plan complete

**All non-upstream-gated items are done.** Gate: **342 tests pass**, ruff +
format + mypy clean, uv-installable, 23 modules; discopt/pounce/feral untouched
apart from the one-line namespace shim.

- **P1** (reproduction): R1 ✅, R2 ✅, R3 ✅.
- **P2** (dual & numerical): D1 ✅, D2 ✅.
- **P3** (structural): S1 ✅, S2 ✅, S3 ✅.
- **P4** (upstream-gated): plugin-side halves built — K1 extraction + guarded
  rewrite ✅, K2 ordering emitter ✅; the end-to-end **wiring** waits on
  **discopt#379** (implicit-function node), **feral#107** + **pounce#180**
  (external ordering / Schur KKT). **E6** remains a no-go until pounce/feral expose
  the ordering hook. **R2's** CSE finding is filed as **discopt#383**.
- Earlier phases: E1–E5 ✅, E3/E4 ✅, **G4** declined (no-op guard, evidence),
  **E6** no-go (documented).

Everything shippable without upstream changes is shipped; the only remaining work
is the four filed upstream features and the wiring they unblock.
