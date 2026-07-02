"""Variable aggregation (reduced-space presolve) for discopt models.

Variable *aggregation* substitutes a variable that is defined by an equality
constraint into the rest of the model, eliminating both a variable and a
constraint to produce a smaller *reduced-space* formulation.  This package
formalises the approach of Naik, Biegler, Bent & Parker, "Variable aggregation
for nonlinear optimization problems" (arXiv:2502.13869), as a model transform
that runs on the :mod:`discopt.modeling` expression DAG *before* the Rust
presolve / branch-and-bound engine.

The paper's key soundness condition (Lemma 1) is that a set of variables /
defining equalities may be eliminated *explicitly* — with no iterative solve —
iff the Jacobian of the defining equations with respect to the eliminated
variables is strictly lower triangular.  Under that condition the eliminated
variables can be recovered post-solve by back-substitution in reverse
elimination order.

Layering (built incrementally):

- :mod:`discopt.aggregation.matching` — pure-graph kernel: maximum matching,
  strongly-connected components, and block-triangularization (paper Alg. 7-9).
  Stdlib-only, no discopt/JAX/Rust dependency.
- ``incidence`` — bipartite incidence graph over (scalar variables x equality
  constraints), with linear/nonlinear edge labelling (paper Table 1).
- ``strategies`` — the six selection policies (LD1, ECD2, LD2, D2, GR, LM).
- ``transform`` / ``recovery`` — apply an aggregation set and recover the
  eliminated variables from a reduced-space solution.

This module keeps imports light on purpose: importing
:mod:`discopt.aggregation` must not drag in JAX or the Rust extension, so the
graph kernel can be used (and tested) standalone.
"""

from __future__ import annotations

__version__ = "0.2.0"

from discopt.aggregation.api import (
    AggregatedSolveResult,
    AggregationResult,
    aggregate,
    solve,
)
from discopt.aggregation.benchmark import BenchmarkRow, benchmark_model
from discopt.aggregation.derivatives import differentiate_expression
from discopt.aggregation.diagnostics import (
    StructuralDiagnosis,
    structural_diagnosis,
)
from discopt.aggregation.duals import (
    DualRecovery,
    kkt_stationarity_residual,
    recover_duals,
    recover_duals_from_result,
)
from discopt.aggregation.guards import (
    apply_elimination_guards,
    conditioning_estimate,
)
from discopt.aggregation.hessian_cost import (
    expression_dag_size,
    expression_depth,
    expression_tree_size,
    model_expression_size,
    sharing_ratio,
)
from discopt.aggregation.implicit import (
    ImplicitBlock,
    ImplicitElimination,
    eliminate_implicit,
    implicit_blocks,
)
from discopt.aggregation.incidence import (
    IncidenceModel,
    all_vars,
    build_incidence,
    candidate_variables,
    linear_vars,
)
from discopt.aggregation.matching import (
    BipartiteGraph,
    DulmageMendelsohn,
    block_triangularize,
    dulmage_mendelsohn,
    maximum_matching,
    strongly_connected_components,
    topological_order,
)
from discopt.aggregation.metrics import (
    ModelStructure,
    compare_methods,
    exact_hessian_nnz,
    model_structure,
)
from discopt.aggregation.ordering import (
    BlockTriangularOrdering,
    block_triangular_ordering,
)
from discopt.aggregation.presolve import (
    CompositionCheck,
    PresolveOverlap,
    native_presolve_overlap,
    recommended_ordering,
    verify_composition,
)
from discopt.aggregation.profiling import (
    balanced_sum,
    build_scaling,
    indep_scaling_model,
    profile_build,
)
from discopt.aggregation.recovery import (
    RecoveryMap,
    evaluate_expression,
    interval_evaluate,
)
from discopt.aggregation.scalarize import ScalarizationMap, scalarize_model
from discopt.aggregation.schur import kkt_schur_indices
from discopt.aggregation.strategies import (
    STRATEGIES,
    AggregationSet,
    Elimination,
    aggregation_set,
)
from discopt.aggregation.sweep import SweepResult, reliability_sweep
from discopt.aggregation.tearing import find_cycle, max_acyclic_subset_milp
from discopt.aggregation.testproblems import (
    PROBLEMS,
    TestProblem,
    gas_pipeline,
    reaction_diffusion,
    recycle_loop,
    unit_selection,
)
from discopt.aggregation.transform import (
    ReducedModel,
    apply_aggregation,
    defining_expressions,
    substitute_expression,
)

__all__ = [
    "__version__",
    # public API
    "aggregate",
    "solve",
    "AggregationResult",
    "AggregatedSolveResult",
    # graph kernel
    "BipartiteGraph",
    "block_triangularize",
    "maximum_matching",
    "strongly_connected_components",
    "topological_order",
    "DulmageMendelsohn",
    "dulmage_mendelsohn",
    # structural diagnostics (S1)
    "structural_diagnosis",
    "StructuralDiagnosis",
    # incidence
    "IncidenceModel",
    "build_incidence",
    "candidate_variables",
    "all_vars",
    "linear_vars",
    # strategies
    "aggregation_set",
    "AggregationSet",
    "Elimination",
    "STRATEGIES",
    # transform + recovery
    "apply_aggregation",
    "ReducedModel",
    "defining_expressions",
    "substitute_expression",
    "RecoveryMap",
    "evaluate_expression",
    "interval_evaluate",
    "differentiate_expression",
    # implicit block elimination (E5) + cyclic-block extraction/elimination (K1)
    "ImplicitBlock",
    "implicit_blocks",
    "eliminate_implicit",
    "ImplicitElimination",
    # block-triangular / Schur KKT ordering (E6 / K2 emitter)
    "block_triangular_ordering",
    "BlockTriangularOrdering",
    # expression-graph / Hessian-cost diagnostics (R2)
    "expression_tree_size",
    "expression_dag_size",
    "expression_depth",
    "sharing_ratio",
    "model_expression_size",
    # build-time profiling at scale (R3)
    "profile_build",
    "build_scaling",
    "indep_scaling_model",
    "balanced_sum",
    # elimination guards: expression-size + conditioning caps (D2)
    "apply_elimination_guards",
    "conditioning_estimate",
    # dual / multiplier recovery
    "recover_duals",
    "recover_duals_from_result",
    "kkt_stationarity_residual",
    "DualRecovery",
    # structural metrics
    "model_structure",
    "compare_methods",
    "exact_hessian_nnz",
    "ModelStructure",
    # scalarization
    "scalarize_model",
    "ScalarizationMap",
    # test problems (S1)
    "TestProblem",
    "PROBLEMS",
    "reaction_diffusion",
    "unit_selection",
    "gas_pipeline",
    "recycle_loop",
    # runtime benchmark (S2)
    "benchmark_model",
    "BenchmarkRow",
    # reliability sweep (S3)
    "reliability_sweep",
    "SweepResult",
    # scalable exact tearing (S2)
    "max_acyclic_subset_milp",
    "find_cycle",
    # structured / Schur KKT reducible-block indices (K2, plugin-side analysis)
    "kkt_schur_indices",
    # native-presolve overlap audit + composition safety (S3)
    "native_presolve_overlap",
    "recommended_ordering",
    "verify_composition",
    "PresolveOverlap",
    "CompositionCheck",
]
