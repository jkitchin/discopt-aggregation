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
from discopt._jax.dependent_vars import _collect_var_names, _isolated_affine_coeff
from discopt.modeling.core import Constraint, VarType

from discopt.aggregation.matching import BipartiteGraph


def candidate_variables(model: Any) -> dict[str, Any]:
    """Map name -> Variable for every scalar continuous variable in ``model``.

    These are the only variables eligible for explicit aggregation.
    """
    candidates: dict[str, Any] = {}
    for v in getattr(model, "_variables", []):
        if getattr(v, "var_type", None) != VarType.CONTINUOUS:
            continue
        if int(getattr(v, "size", 1)) != 1:
            continue
        candidates[v.name] = v
    return candidates


def all_vars(constraint: Constraint, candidate_names: set[str]) -> set[str]:
    """Candidate variables that participate in ``constraint`` (paper ``all_vars``).

    Returns the subset of ``candidate_names`` that appear anywhere in the
    constraint body.  Returns an empty set for an opaque body that cannot be
    analysed (the safe, conservative direction).
    """
    present = _collect_var_names(constraint.body)
    if present is None:  # opaque node -> cannot analyse; contribute nothing
        return set()
    return present & candidate_names


def linear_vars(
    constraint: Constraint, candidate_names: set[str], pivot_tol: float = 0.0
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
    """
    present = all_vars(constraint, candidate_names)
    out: dict[str, float] = {}
    for name in present:
        coeff = _isolated_affine_coeff(constraint.body, name)
        if coeff is None or coeff == 0.0 or not np.isfinite(coeff):
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
    pivot_tol: float = 0.0
    max_fill: float | None = None
    tearing: str = "greedy"  # cyclic-block tear: "greedy" or "exact"
    decomposable: bool = False  # local-search matching for higher n_block (E4)

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


def build_incidence(
    model: Any,
    pivot_tol: float = 1e-6,
    max_fill: float | None = None,
    tearing: str = "greedy",
    decomposable: bool = False,
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
    candidates = candidate_variables(model)
    candidate_names = set(candidates)

    inc = IncidenceModel(
        model=model,
        variables=candidates,
        pivot_tol=pivot_tol,
        max_fill=max_fill,
        tearing=tearing,
        decomposable=decomposable,
    )

    # Total degree of each candidate across every constraint and the objective.
    var_degree = {name: 0 for name in candidate_names}
    for c in getattr(model, "_constraints", []):
        if isinstance(c, Constraint):
            names = _collect_var_names(c.body)
            if names:
                for nm in names & candidate_names:
                    var_degree[nm] += 1
    obj = getattr(model, "_objective", None)
    if obj is not None:
        names = _collect_var_names(obj.expression)
        if names:
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
        lin = linear_vars(c, candidate_names, pivot_tol=pivot_tol)
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
    "all_vars",
    "linear_vars",
]
