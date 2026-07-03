"""Bipartite incidence graph of a discopt model (paper Section 2.3, Table 1).

Given a :class:`discopt.modeling.core.Model`, :func:`build_incidence` constructs
the bipartite graph whose left nodes are *equality constraints* and whose right
nodes are *scalar continuous variables*, with an edge whenever a variable
participates in a constraint.  Each edge is labelled **linear** or **nonlinear**:
a variable participates *linearly* in a constraint when it appears affinely with
a constant, nonzero coefficient (so the equality can be solved for it without
introducing a nonlinearity in that variable).

Only scalar (``size == 1``) continuous variables are eligible aggregation
targets — integer/binary variables are handled by integer branching, and
indexed/array references are not the per-scalar defining-equation pattern.  This
matches the scope of the three existing Rust presolve passes
(``eliminate`` / ``aggregate`` / ``factorable_elim``) and of
:mod:`discopt._jax.dependent_vars`, whose affine-coefficient analyzers we reuse
here rather than re-deriving.

The two subroutines the paper names in Table 1 are exposed directly:

- :func:`all_vars` — the candidate variables that participate in a constraint.
- :func:`linear_vars` — the candidate variables that participate *linearly*,
  together with their constant coefficients.

References
----------
Naik, Biegler, Bent, Parker, arXiv:2502.13869, Section 2.3 and Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from discopt.modeling.core import Constraint, VarType

from discopt.aggregation.exprwalk import affine_coefficients, collect_variable_names
from discopt.aggregation.matching import BipartiteGraph

_INTEGER_TYPES = (VarType.INTEGER, VarType.BINARY)


def candidate_variables(model: Any, integer_aggregation: bool = False) -> dict[str, Any]:
    """Map name -> Variable for every scalar variable eligible for aggregation.

    Scalar *continuous* variables are always candidates.  When
    ``integer_aggregation`` is True, scalar *integer* and *binary* variables are
    admitted too (IA1); their edges are then further filtered by the integrality-
    implication rule in :func:`linear_vars`, so admitting them here is necessary
    but not sufficient for an eliminable edge.  With the flag off (the default),
    integer/binary variables are excluded, so the candidate set — and hence the
    whole pipeline — is byte-identical to the continuous-only behaviour.
    """
    candidates: dict[str, Any] = {}
    for v in getattr(model, "_variables", []):
        if int(getattr(v, "size", 1)) != 1:
            continue
        vt = getattr(v, "var_type", None)
        if vt == VarType.CONTINUOUS:
            candidates[v.name] = v
        elif integer_aggregation and vt in _INTEGER_TYPES:
            candidates[v.name] = v
    return candidates


def integer_variable_names(model: Any) -> frozenset[str]:
    """Names of every scalar integer/binary variable in ``model``.

    Tracked regardless of the ``integer_aggregation`` flag: the integrality-
    implication filter (IA1) needs to know which of a constraint's *other*
    variables are integer, whether or not they are aggregation candidates.
    """
    names: set[str] = set()
    for v in getattr(model, "_variables", []):
        if int(getattr(v, "size", 1)) != 1:
            continue
        if getattr(v, "var_type", None) in _INTEGER_TYPES:
            names.add(v.name)
    return frozenset(names)


def _integer_edge_admissible(
    pivot_name: str,
    pivot_coeff: float,
    const: float,
    coeffs: dict[str, float | None],
    integer_names: frozenset[str],
    int_tol: float = 1e-9,
) -> bool:
    """Whether integer ``pivot_name`` may be eliminated via this affine body.

    Enforces the integrality-implication rule (Achterberg et al. 2020 §4;
    Zhang & Yuan 2024): the body is fully affine, the pivot coefficient is
    ``±1``, every *other* participating variable is integer/binary, and every
    other coefficient and the constant term are integral.  Only then is
    integrality of the eliminated variable implied by the surviving ones, so
    substituting it back always yields an integer.
    """
    # Fully affine: no variable may participate nonlinearly (None coefficient).
    if any(c is None for c in coeffs.values()):
        return False
    # Pivot coefficient must be a unit (the general "pivot divides all" rule is
    # deliberately scoped to ±1).
    if abs(abs(pivot_coeff) - 1.0) > int_tol:
        return False
    for name, coeff in coeffs.items():
        if name == pivot_name:
            continue
        if name not in integer_names:
            return False  # a continuous companion breaks the implication
        assert coeff is not None  # guaranteed by the None-check above
        if abs(coeff - round(coeff)) > int_tol:
            return False  # non-integral coefficient
    return abs(const - round(const)) <= int_tol


def all_vars(constraint: Constraint, candidate_names: set[str]) -> set[str]:
    """Candidate variables that participate in ``constraint`` (paper ``all_vars``).

    Returns the subset of ``candidate_names`` that appear anywhere in the
    constraint body.  Variables inside opaque-but-walkable nodes (``CustomCall``,
    ``MatMulExpression``) are counted — treating them as absent is the *unsafe*
    direction for occurrence counting, since it would let such a variable be
    eliminated via another constraint and leave a dangling reference.  A
    genuinely unrecognised node raises :class:`NotImplementedError` (refuse
    rather than risk a hidden variable).
    """
    names, analyzable = collect_variable_names(constraint.body)
    if not analyzable:
        raise NotImplementedError(
            "aggregation cannot analyse an unrecognised expression node in a "
            "constraint body; refusing rather than risk eliminating a variable "
            "hidden inside it"
        )
    return names & candidate_names


def linear_vars(
    constraint: Constraint,
    candidate_names: set[str],
    pivot_tol: float = 0.0,
    integer_names: frozenset[str] = frozenset(),
) -> dict[str, float]:
    """Candidate variables that participate *linearly* (paper ``linear_vars``).

    Returns a mapping ``{name: coeff}`` for each candidate variable that appears
    in the constraint body affinely with a constant, finite, nonzero coefficient
    ``coeff``.  A variable that occurs nonlinearly (under a power, in a product
    with another variable, in a denominator, inside a transcendental, ...) is
    excluded, because eliminating it via this equality would require an implicit
    solve rather than an explicit substitution.

    ``pivot_tol`` guards against ill-conditioned substitutions: a variable is
    kept only if its coefficient is not tiny relative to the constraint's other
    linear coefficients, i.e. ``|coeff| >= pivot_tol * max(1, max_j |coeff_j|)``.
    Eliminating a variable whose coefficient is far smaller than its neighbours'
    divides the defining equation by a near-zero pivot, the numerical hazard
    Achterberg et al. flag for aggregation.  The default ``0.0`` disables the
    test (exact paper behaviour); :func:`build_incidence` applies a nonzero
    default.

    ``integer_names`` are the model's integer/binary variables (IA1).  A
    *continuous* candidate keeps the rule above unchanged.  An *integer*
    candidate (its name in ``integer_names``) is admitted only when integrality
    of the eliminated variable is implied — see :func:`_integer_edge_admissible`
    — so an integer edge never introduces a fractional value on back-substitution.
    """
    present = all_vars(constraint, candidate_names)
    out: dict[str, float] = {}
    # One DAG-memoised pass extracts every variable's affine coefficient at once
    # (O(#distinct nodes)), replacing an O(tree) walk per candidate — the walk that
    # took minutes on substitution-expanded bodies whose shared DAG is small.
    extracted = affine_coefficients(constraint.body)
    const: float = 0.0 if extracted is None else extracted[0]
    coeffs: dict[str, float | None] = {} if extracted is None else extracted[1]
    for name in present:
        coeff = coeffs.get(name)
        if coeff is None or coeff == 0.0 or not np.isfinite(coeff):
            continue
        if name in integer_names and not _integer_edge_admissible(
            name, coeff, const, coeffs, integer_names
        ):
            continue
        out[name] = float(coeff)
    if pivot_tol > 0.0 and out:
        scale = max(1.0, max(abs(c) for c in out.values()))
        threshold = pivot_tol * scale
        out = {n: c for n, c in out.items() if abs(c) >= threshold}
    return out


@dataclass
class IncidenceModel:
    """Incidence graphs and lookup tables derived from a model.

    Attributes:
        model: the source model.
        constraints: ``{cid: Constraint}`` for each equality constraint, where
            ``cid`` is the index of the constraint in ``model._constraints``.
        variables: ``{name: Variable}`` for the scalar continuous candidates.
        graph: bipartite graph over all ``(cid, varname)`` incidences.
        linear_graph: bipartite graph over the *linear* incidences only.
        linear_coeff: ``{(cid, varname): coeff}`` constant coefficient of a
            variable in a constraint, for linear edges.
        var_degree: ``{varname: count}`` number of constraints (any sense) plus
            the objective in which each candidate variable appears; used to
            estimate the fill introduced by eliminating it.
        pivot_tol: the relative pivot tolerance used to build ``linear_graph``.
        max_fill: optional cap on the estimated fill of an elimination
            (``(#other vars in defining constraint) * (#target constraints)``);
            ``None`` disables the cap.  Read by the selection strategies.
    """

    model: Any
    constraints: dict[int, Constraint] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    graph: BipartiteGraph = field(default_factory=BipartiteGraph)
    linear_graph: BipartiteGraph = field(default_factory=BipartiteGraph)
    linear_coeff: dict[tuple[int, str], float] = field(default_factory=dict)
    var_degree: dict[str, int] = field(default_factory=dict)
    total_var_count: dict[int, int] = field(default_factory=dict)
    pivot_tol: float = 0.0
    max_fill: float | None = None
    tearing: str = "greedy"  # cyclic-block tear: "greedy" or "exact"
    decomposable: bool = False  # local-search matching for higher n_block (E4)
    order: str = "index"  # greedy candidate order: "index" or "min_fill" (R1)
    integer_names: frozenset[str] = frozenset()  # all scalar integer/binary vars (IA1)

    def fill_estimate(self, cid: int, var: str) -> int:
        """Estimated new Jacobian incidences from eliminating ``var`` via ``cid``.

        Substituting ``var``'s defining expression (the other variables of
        constraint ``cid``) into each other constraint that contains ``var``
        introduces roughly ``(#other vars) * (#target constraints)`` incidences.
        """
        n_other_vars = max(len(self.constraint_vars(cid)) - 1, 0)
        n_targets = max(self.var_degree.get(var, 1) - 1, 0)
        return n_other_vars * n_targets

    def is_linear_edge(self, cid: int, var: str) -> bool:
        """Whether variable ``var`` participates linearly in constraint ``cid``."""
        return (cid, var) in self.linear_coeff

    def constraint_vars(self, cid: int) -> set[str]:
        """All candidate variables incident to constraint ``cid``."""
        return self.graph.neighbors_of_left(cid)

    def constraint_linear_vars(self, cid: int) -> set[str]:
        """Candidate variables incident to ``cid`` via a linear edge."""
        return self.linear_graph.neighbors_of_left(cid)

    def constraint_total_degree(self, cid: int) -> int:
        """Count of *all* variables in ``cid`` (candidate or not).

        Degree filters (paper Table 2) bound the *total* number of variables in a
        defining constraint — ``y = x + i`` with an integer ``i`` is a three-
        variable constraint even though only ``x, y`` are aggregation candidates.
        Falls back to the candidate count if the total was not recorded.
        """
        return self.total_var_count.get(cid, len(self.constraint_vars(cid)))


def build_incidence(
    model: Any,
    pivot_tol: float = 1e-6,
    max_fill: float | None = None,
    tearing: str = "greedy",
    decomposable: bool = False,
    order: str = "index",
    integer_aggregation: bool = False,
) -> IncidenceModel:
    """Build the bipartite incidence graphs for ``model``.

    Left nodes are the integer indices of equality constraints within
    ``model._constraints``; right nodes are scalar continuous variable names.
    Non-equality constraints and non-candidate variables are ignored.

    ``pivot_tol`` (default ``1e-6``) is the relative pivot tolerance for the
    linear graph (see :func:`linear_vars`).  ``max_fill`` is stored on the
    result for the selection strategies to enforce (see
    :meth:`IncidenceModel.fill_estimate`).  ``tearing`` (``"greedy"`` or
    ``"exact"``) selects how cyclic blocks are torn in linear-matching.
    """
    candidates = candidate_variables(model, integer_aggregation=integer_aggregation)
    candidate_names = set(candidates)
    integer_names = integer_variable_names(model)

    inc = IncidenceModel(
        model=model,
        variables=candidates,
        pivot_tol=pivot_tol,
        max_fill=max_fill,
        tearing=tearing,
        decomposable=decomposable,
        order=order,
        integer_names=integer_names,
    )

    # Total degree of each candidate across every constraint and the objective.
    # Walk with the deep collector so occurrences inside opaque-but-walkable
    # nodes are counted, and an unrecognised node in *any* constraint/objective
    # (not just equalities) refuses loudly rather than hiding a variable.
    var_degree = {name: 0 for name in candidate_names}
    for c in getattr(model, "_constraints", []):
        if isinstance(c, Constraint):
            names, analyzable = collect_variable_names(c.body)
            if not analyzable:
                raise NotImplementedError(
                    "aggregation cannot analyse an unrecognised expression node "
                    "in a constraint body; refusing rather than risk eliminating "
                    "a variable hidden inside it"
                )
            for nm in names & candidate_names:
                var_degree[nm] += 1
    obj = getattr(model, "_objective", None)
    if obj is not None:
        names, analyzable = collect_variable_names(obj.expression)
        if not analyzable:
            raise NotImplementedError(
                "aggregation cannot analyse an unrecognised expression node in "
                "the objective; refusing rather than risk eliminating a hidden "
                "variable"
            )
        for nm in names & candidate_names:
            var_degree[nm] += 1
    inc.var_degree = var_degree

    for cid, c in enumerate(getattr(model, "_constraints", [])):
        if not isinstance(c, Constraint):
            continue
        if c.sense != "==":
            continue
        present = all_vars(c, candidate_names)
        if not present:
            continue
        inc.constraints[cid] = c
        # Total variable count (candidates and non-candidates alike) for the
        # degree filters, so e.g. y = x + i (integer i) counts as three variables.
        all_names, _ = collect_variable_names(c.body)
        inc.total_var_count[cid] = len(all_names)
        lin = linear_vars(c, candidate_names, pivot_tol=pivot_tol, integer_names=integer_names)
        for name in present:
            inc.graph.add_edge(cid, name)
            if name in lin:
                inc.linear_graph.add_edge(cid, name)
                inc.linear_coeff[(cid, name)] = lin[name]

    return inc


__all__ = [
    "IncidenceModel",
    "build_incidence",
    "candidate_variables",
    "integer_variable_names",
    "all_vars",
    "linear_vars",
]
