"""Tests for :mod:`discopt.aggregation.incidence`.

Validates the bipartite incidence construction against hand-built models with
known structure: candidate selection (scalar continuous only), edge sets, and
linear vs nonlinear edge labelling with correct constant coefficients.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.incidence import (
    all_vars,
    build_incidence,
    candidate_variables,
    linear_vars,
)
from discopt.aggregation.matching import maximum_matching

pytestmark = pytest.mark.unit


def _vars(m):
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    z = m.continuous("z", lb=-10, ub=10)
    return x, y, z


# --------------------------------------------------------------------------- #
# candidate selection                                                         #
# --------------------------------------------------------------------------- #
def test_candidates_are_scalar_continuous_only():
    m = dm.Model("t")
    x, y, _z = _vars(m)
    m.integer("k", lb=0, ub=5)
    m.binary("b")
    cands = candidate_variables(m)
    assert set(cands) == {"x", "y", "z"}
    assert cands["x"] is x


def test_array_variables_excluded_from_candidates():
    m = dm.Model("t")
    m.continuous("x", lb=-1, ub=1)
    m.continuous("a", shape=(3,), lb=-1, ub=1)  # non-scalar -> excluded
    cands = candidate_variables(m)
    assert set(cands) == {"x"}


# --------------------------------------------------------------------------- #
# all_vars / linear_vars                                                       #
# --------------------------------------------------------------------------- #
def test_all_and_linear_vars_affine_constraint():
    m = dm.Model("t")
    x, y, _z = _vars(m)
    m.subject_to(x + 2 * y == 1)
    c = m._constraints[0]
    names = {"x", "y", "z"}
    assert all_vars(c, names) == {"x", "y"}
    lv = linear_vars(c, names)
    assert lv == {"x": 1.0, "y": 2.0}


def test_linear_vars_excludes_nonlinear_occurrences():
    m = dm.Model("t")
    x, y, z = _vars(m)
    m.subject_to(x * z + y == 0)  # x,z nonlinear (product); y linear
    c = m._constraints[0]
    names = {"x", "y", "z"}
    assert all_vars(c, names) == {"x", "y", "z"}
    assert linear_vars(c, names) == {"y": 1.0}


def test_linear_vars_power_is_nonlinear():
    m = dm.Model("t")
    x, y, _z = _vars(m)
    m.subject_to(y + x**2 == 0)  # y linear, x under a power -> nonlinear
    c = m._constraints[0]
    names = {"x", "y", "z"}
    assert linear_vars(c, names) == {"y": 1.0}


# --------------------------------------------------------------------------- #
# build_incidence                                                             #
# --------------------------------------------------------------------------- #
def test_build_incidence_edges_and_labels():
    m = dm.Model("t")
    x, y, z = _vars(m)
    m.subject_to(x + 2 * y == 1)  # c0: x,y linear
    m.subject_to(x * z == 1)  # c1: x,z nonlinear
    m.subject_to(y + z**2 == 0)  # c2: y linear, z nonlinear
    m.subject_to(x + y <= 5)  # inequality -> ignored
    m.minimize(x + y + z)

    inc = build_incidence(m)

    # Only equality constraints become left nodes.
    assert set(inc.constraints) == {0, 1, 2}

    assert inc.constraint_vars(0) == {"x", "y"}
    assert inc.constraint_vars(1) == {"x", "z"}
    assert inc.constraint_vars(2) == {"y", "z"}

    # Linear labelling.
    assert inc.constraint_linear_vars(0) == {"x", "y"}
    assert inc.constraint_linear_vars(1) == set()
    assert inc.constraint_linear_vars(2) == {"y"}

    assert inc.is_linear_edge(0, "x") and inc.is_linear_edge(0, "y")
    assert not inc.is_linear_edge(1, "x")
    assert inc.is_linear_edge(2, "y") and not inc.is_linear_edge(2, "z")

    # Coefficients on the linear edges.
    assert inc.linear_coeff[(0, "x")] == 1.0
    assert inc.linear_coeff[(0, "y")] == 2.0
    assert inc.linear_coeff[(2, "y")] == 1.0


def test_incidence_ignores_inequalities_only_model():
    m = dm.Model("t")
    x, y, _z = _vars(m)
    m.subject_to(x + y <= 5)
    m.subject_to(x - y >= -3)
    inc = build_incidence(m)
    assert inc.constraints == {}
    assert inc.graph.edges == set()


def test_incidence_no_candidate_variables():
    m = dm.Model("t")
    a = m.continuous("a", shape=(2,), lb=-1, ub=1)
    m.subject_to(a[0] + a[1] == 1)  # references only a non-scalar variable
    inc = build_incidence(m)
    # No scalar candidates -> the constraint contributes no edges.
    assert inc.variables == {}
    assert inc.graph.edges == set()


# --------------------------------------------------------------------------- #
# integration with the matching kernel                                        #
# --------------------------------------------------------------------------- #
def test_linear_graph_feeds_maximum_matching():
    m = dm.Model("t")
    x, y, z = _vars(m)
    m.subject_to(x + y + z == 1)  # all linear
    m.subject_to(y + z == 2)  # linear
    inc = build_incidence(m)
    match = maximum_matching(inc.linear_graph)
    # Two equalities, three linear variables -> both constraints matched.
    assert len(match) == 2
    for cid, var in match.items():
        assert var in inc.constraint_linear_vars(cid)


def test_customcall_blocks_dangling_reference():
    """C3: a variable inside an opaque CustomCall is seen and substituted, not dropped.

    ``y`` is defined by ``y == x^2+1`` and also appears inside a CustomCall in
    another constraint.  Eliminating ``y`` must substitute its definition into
    the CustomCall (no dangling reference to the deleted variable).
    """
    import discopt.modeling.core as core

    from discopt.aggregation.api import aggregate
    from discopt.aggregation.exprwalk import collect_variable_names

    m = dm.Model("customcall")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.subject_to(y - (x**2 + 1.0) == 0.0)
    m.subject_to(core.CustomCall(lambda a, b: a * b, y, z) - 4.0 == 0.0)
    m.minimize(x**2 + z**2)
    agg = aggregate(m, method="lm")
    for c in agg.reduced_model._constraints:
        names, analyzable = collect_variable_names(c.body)
        assert analyzable
        assert "y" not in names, "eliminated variable left dangling inside CustomCall"


def test_unrecognised_node_refuses():
    """C3: a genuinely unrecognised node makes incidence refuse rather than guess."""
    import pytest

    from discopt.aggregation.incidence import build_incidence

    class _Weird:
        # not an Expression the walker knows; child_exprs -> None
        pass

    m = dm.Model("weird")
    x = m.continuous("x")
    y = m.continuous("y")
    m.subject_to(y - x == 0.0)
    m.minimize(x**2)
    # Inject an opaque body discopt's walker cannot analyse.
    m._constraints.append(type(m._constraints[0])(_Weird(), "=="))
    with pytest.raises(NotImplementedError):
        build_incidence(m)
