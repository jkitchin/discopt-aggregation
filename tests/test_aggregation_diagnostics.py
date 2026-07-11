"""Tests for S1: Dulmage-Mendelsohn structural diagnostics.

Cross-checks the coarse decomposition on constructed square / under-determined /
over-determined equality systems, confirms the block partition, that the square
block is exactly what aggregation eliminates, and the pure-graph core.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.api import aggregate
from discopt.aggregation.diagnostics import structural_diagnosis
from discopt.aggregation.matching import BipartiteGraph, dulmage_mendelsohn

pytestmark = pytest.mark.unit


def _square():
    m = dm.Model("sq")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(x + y == 1.0)
    m.subject_to(x - y == 0.0)
    m.minimize(x**2 + y**2)
    return m


def _under():
    m = dm.Model("un")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(x + y == 1.0)
    m.minimize(x**2 + y**2)
    return m


def _over():
    m = dm.Model("ov")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(x == 1.0)
    m.subject_to(y == 2.0)
    m.subject_to(x + y == 3.0)  # redundant -> structurally over-determined
    m.minimize(x**2 + y**2)
    return m


def test_square_system_is_well_posed():
    d = structural_diagnosis(_square())
    assert not d.is_structurally_singular
    assert d.warnings() == []
    assert d.free_variables == []
    assert set(d.dm.square[1]) == {"x", "y"}
    assert d.dm.underdetermined == ([], []) and d.dm.overdetermined == ([], [])


def test_underdetermined_system_has_free_variables():
    d = structural_diagnosis(_under())
    assert not d.is_structurally_singular
    assert set(d.dm.underdetermined[1]) == {"x", "y"}
    assert d.dm.overdetermined == ([], [])
    # one degree of freedom: 2 variables, 1 equality
    assert len(d.dm.underdetermined[1]) - len(d.dm.underdetermined[0]) == 1


def test_free_variables_reports_only_the_degrees_of_freedom():
    """RC8: free_variables returns the *unmatched* under-determined variables (the
    actual DOF), not every variable in the under-determined block.  ``x + y == 1``
    has one degree of freedom, so free_variables has length 1 (which specific
    variable is matching-dependent but deterministic), and degrees_of_freedom is 1.
    """
    d = structural_diagnosis(_under())
    assert len(d.free_variables) == 1
    assert set(d.free_variables) <= {"x", "y"}
    assert d.degrees_of_freedom == 1


def test_overdetermined_system_flags_singular():
    d = structural_diagnosis(_over())
    assert d.is_structurally_singular
    assert d.overdetermined_constraints == [0, 1, 2]
    assert len(d.warnings()) == 1
    assert "over-determined" in d.warnings()[0]


def test_square_block_matches_aggregation_targets():
    # A fixed-variable cascade (a = 3, b = 2a + 1) is uniquely determined -> both
    # in the square block, and aggregation eliminates exactly those.
    m = dm.Model("defn")
    a = m.continuous("a", lb=-9, ub=9)
    b = m.continuous("b", lb=-20, ub=20)
    m.subject_to(a == 3.0)
    m.subject_to(b - 2 * a == 1.0)
    m.minimize(a**2 + b**2)
    d = structural_diagnosis(m)
    assert set(d.dm.square[1]) == {"a", "b"}
    assert d.free_variables == []
    assert set(aggregate(m, method="d2").eliminated) == {"a", "b"}


def test_underdetermined_defining_equation_is_a_free_choice():
    # y = x + 1 with x free: 1 equality, 2 variables -> under-determined block
    # (which variable is "defined" is a matching choice), and aggregation picks one.
    m = dm.Model("free")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(y - (x + 1.0) == 0.0)
    m.minimize(x**2 + y**2)
    d = structural_diagnosis(m)
    assert set(d.dm.underdetermined[1]) == {"x", "y"}
    assert d.dm.square == ([], [])
    assert aggregate(m, method="gr").n_eliminated == 1  # one of the two


def test_blocks_partition_the_nodes():
    d = structural_diagnosis(_over())
    all_left = set(d.dm.underdetermined[0]) | set(d.dm.square[0]) | set(d.dm.overdetermined[0])
    all_right = set(d.dm.underdetermined[1]) | set(d.dm.square[1]) | set(d.dm.overdetermined[1])
    # blocks are disjoint
    assert len(d.dm.underdetermined[0]) + len(d.dm.square[0]) + len(d.dm.overdetermined[0]) == len(
        all_left
    )
    assert len(d.dm.underdetermined[1]) + len(d.dm.square[1]) + len(d.dm.overdetermined[1]) == len(
        all_right
    )


def test_pure_graph_decomposition():
    # square 2x2
    g = BipartiteGraph.from_edges([("c0", "x"), ("c0", "y"), ("c1", "x"), ("c1", "y")])
    assert dulmage_mendelsohn(g).is_square
    # underdetermined: 1 constraint, 2 variables
    g2 = BipartiteGraph.from_edges([("c0", "x"), ("c0", "y")])
    dm2 = dulmage_mendelsohn(g2)
    assert not dm2.is_square
    assert set(dm2.underdetermined[1]) == {"x", "y"}
