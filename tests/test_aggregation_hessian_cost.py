"""Tests for R2: expression-graph size / CSE diagnostics.

These pin the measurement primitives and the two structural facts behind the
Hessian-cost finding: a single substitution preserves sharing (so the DAG stays
small), but aggregating a deep nonlinear chain makes the composed expression's
tree grow super-linearly.  The discopt-lowering half of the finding (a shared
linear DAG still lowers to an exponential Hessian) is demonstrated in the worked
example, not asserted here, to keep the unit suite fast and solver-free.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.api import aggregate
from discopt.aggregation.hessian_cost import (
    expression_dag_size,
    expression_tree_size,
    model_expression_size,
    sharing_ratio,
)
from discopt.aggregation.transform import substitute_expression

pytestmark = pytest.mark.unit


def test_tree_and_dag_agree_on_a_tree():
    m = dm.Model("t")
    x = m.continuous("x")
    y = m.continuous("y")
    expr = x + y * 2.0  # no shared subexpression
    assert expression_tree_size(expr) == expression_dag_size(expr)
    assert sharing_ratio(expr) == pytest.approx(1.0)


def test_single_substitution_preserves_sharing():
    m = dm.Model("s")
    z = m.continuous("z")
    w = m.continuous("w")
    expr = z + 3.0 * z * z  # z referenced 3x (one object)
    d = (w + 1.0) * (w + 1.0)  # a chunky replacement
    result = substitute_expression(expr, {"z": d})
    # shared: dag(result) ~ dag(d) + a few wrapper nodes, NOT ~3 * dag(d)
    assert expression_dag_size(result) < expression_dag_size(d) + 8
    # but the tree counts the replacement several times over
    assert expression_tree_size(result) > expression_dag_size(result)


def _nonlinear_chain(depth: int):
    m = dm.Model(f"chain{depth}")
    x = m.continuous("x", lb=-2, ub=2)
    s = [m.continuous(f"s{i}", lb=-50, ub=50) for i in range(depth)]

    def phi(z):
        return 0.5 * (z + 0.05 * z * z)  # z appears 3x -> compounding

    m.subject_to(s[0] - phi(x) == 0.0)
    for i in range(1, depth):
        m.subject_to(s[i] - phi(s[i - 1]) == 0.0)
    m.minimize(s[depth - 1] ** 2)
    return m


def test_deep_chain_aggregation_grows_superlinearly():
    # gr substitutes the whole chain into the objective; the composed tree grows
    # much faster than the chain depth (the Hessian-cost pathology).
    sizes = []
    for depth in [2, 3, 4]:
        reduced = aggregate(_nonlinear_chain(depth), method="gr").reduced_model
        tree, _dag = model_expression_size(reduced)
        sizes.append(tree)
    # super-linear: each +1 depth more than doubles the composed tree size
    assert sizes[1] > 2 * sizes[0]
    assert sizes[2] > 2 * sizes[1]


def test_model_expression_size_on_plain_model():
    m = dm.Model("p")
    x = m.continuous("x")
    m.subject_to(x >= 0.0)
    m.minimize(x * x)
    tree, dag = model_expression_size(m)
    assert tree >= dag >= 1
