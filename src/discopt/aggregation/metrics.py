"""Structural metrics for studying the effect of aggregation (paper Section 5.1).

The paper evaluates aggregation methods on solver-agnostic *structural*
properties that do not depend on the values of variables: numbers of variables
and constraints, the density of the constraint Jacobian, and the number of
nonzeros in the Lagrangian Hessian (Table 4).  :func:`model_structure` computes
these for a single model and :func:`compare_methods` tabulates them across the
six aggregation strategies, so the trends the paper reports can be reproduced on
discopt models.

Metric definitions
------------------
- ``n_variables`` — flat variable count (sum of variable sizes).
- ``n_constraints`` / ``n_equality`` — number of (equality) algebraic constraints.
- ``nnz_jacobian`` — total variable-constraint incidences (distinct variables per
  constraint, summed); ``avg_nnz_per_con`` is its average.
- ``avg_linear_nnz_per_con`` — average number of *linearly*-participating
  variables per constraint.
- ``nnz_hessian`` — size of the structural Lagrangian-Hessian coupling pattern:
  the set of unordered variable pairs ``{i, j}`` (including diagonal ``{i}``)
  that co-participate *nonlinearly* in the objective or any constraint.  This is
  a well-defined superset of the exact second-derivative sparsity (it couples
  every pair among a function's nonlinearly-appearing variables); it captures the
  paper's finding that aggressive aggregation inflates Hessian coupling.

Metrics are computed at variable-*name* granularity, which is exact for models
whose variables are scalar (each distinct name in a constraint is one
incidence).  Array variables are counted once per name per constraint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from discopt.modeling.core import Constraint

from discopt.aggregation.api import aggregate
from discopt.aggregation.exprwalk import classify_linearity, collect_variable_names
from discopt.aggregation.scalarize import scalarize_model
from discopt.aggregation.strategies import resolve_strategy

_DEFAULT_METHODS = ["ld1", "ecd2", "ld2", "d2", "gr", "lm"]


@dataclass
class ModelStructure:
    """Solver-agnostic structural properties of a model.

    ``nnz_hessian`` is the fast, solver-free structural coupling *superset*
    (every pair among a function's nonlinearly-appearing variables).
    ``nnz_hessian_exact`` is the true Lagrangian-Hessian lower-triangle nonzero
    count from automatic differentiation (the paper's Table 4 quantity),
    populated only when requested; it always satisfies
    ``nnz_hessian_exact <= nnz_hessian``.
    """

    n_variables: int
    n_constraints: int
    n_equality: int
    nnz_jacobian: int
    avg_nnz_per_con: float
    avg_linear_nnz_per_con: float
    nnz_hessian: int
    nnz_hessian_exact: int | None = None


def _function_bodies(model: Any) -> list[Any]:
    bodies: list[Any] = []
    if getattr(model, "_objective", None) is not None:
        bodies.append(model._objective.expression)
    for c in model._constraints:
        if isinstance(c, Constraint):
            bodies.append(c.body)
    return bodies


def exact_hessian_nnz(model: Any) -> int | None:
    """Exact Lagrangian-Hessian lower-triangle nonzero count via AD.

    Uses discopt's :class:`~discopt._jax.nlp_evaluator.NLPEvaluator` to obtain the
    true second-derivative sparsity (the paper's Table 4 quantity).  Returns
    ``0`` for a model with no variables and ``None`` if the evaluator cannot be
    built (e.g. no objective, or an unsupported construct) so callers degrade
    gracefully rather than failing.
    """
    if not model._variables:
        return 0
    try:
        from discopt._jax.nlp_evaluator import NLPEvaluator

        rows, _cols = NLPEvaluator(model).hessian_structure()
        return int(len(rows))
    except Exception:
        return None


def model_structure(model: Any, exact_hessian: bool = False) -> ModelStructure:
    """Compute the structural metrics of ``model``.

    When ``exact_hessian`` is true, also compute the exact Lagrangian-Hessian
    nonzero count via automatic differentiation (slower; builds a JAX
    evaluator).
    """
    n_vars = sum(int(getattr(v, "size", 1)) for v in model._variables)
    cons = [c for c in model._constraints if isinstance(c, Constraint)]
    n_con = len(cons)
    n_eq = sum(1 for c in cons if c.sense == "==")

    # discopt classifies *array* variables as nonlinear conservatively; match that
    # so numbers are identical, while using the DAG-memoised classifier (O(distinct
    # nodes)) instead of per-name _isolated_affine_coeff (O(tree)) — the latter is
    # intractable on substitution-expanded bodies (a reduced constraint can have a
    # multi-million-node tree over a few-thousand-node DAG).
    array_names = frozenset(v.name for v in model._variables if int(getattr(v, "size", 1)) > 1)

    total_nnz = 0
    total_lin = 0
    for c in cons:
        names, analyzable = collect_variable_names(c.body)
        if not analyzable:  # opaque body: cannot analyse, contribute nothing
            continue
        total_nnz += len(names)
        linmap = classify_linearity(c.body, array_names)
        total_lin += sum(1 for nm in names if linmap.get(nm, False))

    # Structural Lagrangian-Hessian coupling pattern.
    pairs: set[frozenset] = set()
    for body in _function_bodies(model):
        names, analyzable = collect_variable_names(body)
        if not analyzable:
            continue
        linmap = classify_linearity(body, array_names)
        nl = [nm for nm in names if not linmap.get(nm, False)]
        for i in range(len(nl)):
            for j in range(i, len(nl)):
                pairs.add(frozenset({nl[i], nl[j]}))

    return ModelStructure(
        n_variables=n_vars,
        n_constraints=n_con,
        n_equality=n_eq,
        nnz_jacobian=total_nnz,
        avg_nnz_per_con=(total_nnz / n_con) if n_con else 0.0,
        avg_linear_nnz_per_con=(total_lin / n_con) if n_con else 0.0,
        nnz_hessian=len(pairs),
        nnz_hessian_exact=exact_hessian_nnz(model) if exact_hessian else None,
    )


def compare_methods(
    model: Any,
    methods: list[str] | None = None,
    recursive: bool = True,
    exact_hessian: bool = False,
) -> list[dict]:
    """Tabulate structural metrics for the original model and each method.

    Returns a list of rows (dicts).  The first row is the original model
    (``method="original"``, ``n_eliminated=0``); each subsequent row applies one
    aggregation method and reports the reduced model's structure.  This mirrors
    the paper's Table 4.  ``exact_hessian`` adds the AD-computed Hessian nonzero
    count to every row (slower).

    The ``original`` row is computed on the *scalarized* model, matching the
    aggregated rows (which scalarize internally), so the Var./NNZ/Hessian columns
    are on the same per-scalar granularity — otherwise an array-variable original
    undercounts against its reduced rows (T3).  Method names accept ``d<k>``
    (``d3``, ``d4``, …) in addition to the six named strategies (T2).
    """
    methods = methods or list(_DEFAULT_METHODS)
    for meth in methods:
        resolve_strategy(meth)  # validate (accepts d<k>; raises for unknown)

    baseline_model, smap = scalarize_model(model)
    baseline_model = baseline_model if smap.scalarized else model
    rows: list[dict] = [
        {
            "method": "original",
            "n_eliminated": 0,
            **asdict(model_structure(baseline_model, exact_hessian)),
        }
    ]
    for meth in methods:
        result = aggregate(model, method=meth, recursive=recursive)
        struct = model_structure(result.reduced_model, exact_hessian)
        rows.append({"method": meth, "n_eliminated": result.n_eliminated, **asdict(struct)})
    return rows


__all__ = ["ModelStructure", "model_structure", "compare_methods", "exact_hessian_nnz"]
