"""Tests for K2 (plugin side): KKT-space reducible-block indices for the Schur solve.

``kkt_schur_indices`` is pure analysis — it maps the linear-matching reducible
block (eliminated variables + their defining equalities) to the KKT-space indices
``pounce.Problem.set_kkt_schur_block`` consumes.  It does no solving and touches no
solver internals; handing the indices to pounce belongs in discopt's pounce
bridge.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.schur import kkt_schur_indices
from discopt.aggregation.testproblems import reaction_diffusion

pytestmark = pytest.mark.unit


def test_indices_are_valid_kkt_positions():
    m = reaction_diffusion(20)
    idx = kkt_schur_indices(m)
    # n = 2n+1-ish vars, m constraints; KKT dim = n + m. Indices in range, unique.
    assert idx == sorted(set(idx))
    assert all(i >= 0 for i in idx)
    assert len(idx) > 0


def test_indices_span_x_and_eq_dual_blocks():
    # the reducible block has both eliminated variables (x-block) and their
    # defining equalities (eq-dual block, indices >= n)
    from discopt.aggregation.scalarize import scalarize_model

    m = reaction_diffusion(20)
    sm, _ = scalarize_model(m)
    n = len(sm._variables)
    idx = kkt_schur_indices(m)
    assert any(i < n for i in idx)  # x-block entries
    assert any(i >= n for i in idx)  # eq-dual entries


def test_inequality_model_offsets_eq_dual_block():
    # a model with an inequality: the eq-dual index must include the n_ineq slack
    # offset (KKT layout [x, slack, eq-dual, ineq-dual]).
    m = dm.Model("mix")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-20, ub=20)
    z = m.continuous("z", lb=-5, ub=5)
    m.subject_to(y - (2 * x + 1) == 0.0)  # eq (cid 0), defines y; eq_rank 0
    m.subject_to(x + z >= 1.0)  # inequality (cid 1) -> n_ineq = 1
    m.minimize(y**2 + x + z)
    # n_x = 3, n_ineq = 1 -> eq-dual base = 4; y is x-block index 1 (order x,y,z)
    idx = kkt_schur_indices(m)
    assert idx == [1, 4]  # {y in x-block, its defining eq in the offset eq-dual block}


def test_surviving_array_shifts_flat_indices():
    """C5: an abstained array stays multi-scalar in the flat KKT layout.

    ``a`` (shape ``(3,)``) is used via a whole-array ``SumExpression`` so it does
    not scalarize; it therefore occupies flat x-indices ``0..2`` and shifts ``x``
    and ``y`` to ``3`` and ``4``.  The old code counted ``a`` as one variable
    (``n_x = 3``) and used positional indices, giving the wrong block
    ``[1, 2, 4, 5]``; the flat layout gives ``[3, 4, 6, 7]``.
    """
    from discopt.modeling.core import SumExpression

    m = dm.Model("arr")
    a = m.continuous("a", shape=(3,), lb=-1, ub=1)
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-20, ub=20)
    m.subject_to(SumExpression(a) + x - 10.0 == 0.0)  # a abstains -> stays (3,)
    m.subject_to(y - (2 * x + 1) == 0.0)
    m.subject_to(x <= 4.0)  # n_ineq = 1
    m.minimize(y**2 + x)

    idx, analyzed = kkt_schur_indices(m, return_model=True)
    n_x = sum(int(getattr(v, "size", 1)) for v in analyzed._variables)
    assert n_x == 5  # a(3) + x + y, not 3
    # x and y both eliminated: x-block {3,4} (past the array), eq-dual {6,7}
    assert idx == [3, 4, 6, 7]
    assert all(i >= 3 for i in idx if i < n_x)  # x-block past the array
    assert all(i >= n_x + 1 for i in idx if i >= n_x)  # eq-dual past n_x + n_ineq


def test_no_eliminable_block_gives_empty():
    # a model where lm eliminates nothing -> empty reducible block
    m = dm.Model("free")
    x = m.continuous("x", lb=-5, ub=5)
    m.minimize(x**2)
    assert kkt_schur_indices(m) == []


def test_deterministic():
    m = reaction_diffusion(16)
    assert kkt_schur_indices(m) == kkt_schur_indices(m)
