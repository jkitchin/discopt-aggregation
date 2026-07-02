"""Tests for K1: cyclic-block extraction into ImplicitBlock primitives.

``implicit_blocks(model)`` finds the irreducible (size > 1) diagonal blocks of
the linear-matching matching — the blocks the explicit strategies drop — and
returns them as :class:`ImplicitBlock` primitives.  ``eliminate_implicit`` is the
guarded reduced-model rewrite: it identifies the blocks but raises until
discopt#379's implicit-function node exists.  These tests pin: a nonsingular
2-cycle is extracted and solves to its closed form, a pure chain yields no
blocks, a degenerate (singular) cycle is extracted but does not solve, and the
guarded rewrite raises with the block count.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.implicit import eliminate_implicit, implicit_blocks

pytestmark = pytest.mark.unit


def _nonsingular_cycle():
    # x = 0.5y + 3, y = 0.5x + 2 : both constraints touch only {x, y}, so the
    # maximum matching is forced into the 2-cycle. Jacobian [[1,-0.5],[-0.5,1]]
    # has det 0.75 (nonsingular). Solution: x = 16/3, y = 14/3.
    m = dm.Model("cyc_ns")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(x - (0.5 * y + 3) == 0.0)
    m.subject_to(y - (0.5 * x + 2) == 0.0)
    m.minimize(x**2 + y**2)
    return m


def _singular_cycle():
    # x - y = 0 and x - y = 1 : a 2-cycle with singular Jacobian [[1,-1],[1,-1]]
    # that is also inconsistent, so Newton cannot converge (unlike x==y, whose
    # (0,0) is a valid root). Extracted as a block, but no implicit solve.
    m = dm.Model("cyc_s")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(x - y == 0.0)
    m.subject_to(x - y == 1.0)
    m.minimize(x**2 + y**2)
    return m


def _chain():
    m = dm.Model("chain")
    xs = [m.continuous(f"x{i}", lb=-10, ub=10) for i in range(5)]
    for i in range(1, 5):
        m.subject_to(xs[i] - (2 * xs[i - 1] + 1) == 0.0)
    m.minimize(sum(x**2 for x in xs))
    return m


def test_nonsingular_cycle_extracted_and_solves():
    blocks = implicit_blocks(_nonsingular_cycle())
    assert len(blocks) == 1
    block = blocks[0]
    assert set(block.variables) == {"x", "y"}
    assert len(block.equations) == 2
    assert block.is_nonsingular({}, {"x": 0.0, "y": 0.0})
    sol = block.solve({})
    assert sol is not None
    assert sol["x"] == pytest.approx(16 / 3, abs=1e-6)
    assert sol["y"] == pytest.approx(14 / 3, abs=1e-6)


def test_chain_has_no_cyclic_blocks():
    assert implicit_blocks(_chain()) == []


def test_singular_cycle_extracted_but_does_not_solve():
    blocks = implicit_blocks(_singular_cycle())
    assert len(blocks) == 1
    block = blocks[0]
    assert set(block.variables) == {"x", "y"}
    # extracted, but the singular Jacobian means no implicit solve
    assert not block.is_nonsingular({}, {"x": 0.0, "y": 0.0})
    assert block.solve({}) is None


def test_extraction_is_deterministic():
    a = implicit_blocks(_nonsingular_cycle())
    b = implicit_blocks(_nonsingular_cycle())
    assert [blk.variables for blk in a] == [blk.variables for blk in b]


def test_eliminate_implicit_eliminates_the_block():
    # now wired end-to-end (discopt#379): the nonsingular cycle is eliminated
    el = eliminate_implicit(_nonsingular_cycle())
    assert set(el.eliminated) == {"x", "y"}
    assert el.reduced_model is not None


def test_eliminate_implicit_raises_when_no_cyclic_block():
    with pytest.raises(ValueError, match="no eliminable"):
        eliminate_implicit(_chain())
