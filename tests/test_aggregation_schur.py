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
    m.subject_to(y - (2 * x + 1) == 0.0)  # eq (cid 0); x and y are both valid pivots
    m.subject_to(x + z >= 1.0)  # inequality (cid 1) -> n_ineq = 1
    m.minimize(y**2 + x + z)
    # n_x = 3, n_ineq = 1 -> eq-dual base = 4.  The defining equality (cid 0) has
    # two valid pivots (x, y); the deterministic matching (C6) picks x -> x-block
    # index 0.  Either choice is mathematically valid; what matters here is the
    # eq-dual offset lands at n_x + n_ineq = 4, not n_x = 3.
    idx = kkt_schur_indices(m)
    assert idx == [0, 4]  # {eliminated var in x-block, its defining eq at offset 4}


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


def test_equality_only_hand_checked_offsets():
    """T8: an all-equality model with hand-checked KKT indices.

    Two eliminable variables, no inequalities (``n_ineq = 0``), so the eq-dual
    block starts at ``n_x = 3``.  A wrong offset in ``kkt_schur_indices`` would
    fail this exact-index assertion rather than pass silently (the e2e objective
    check cannot catch a wrong-but-in-range block because pounce falls back).
    """
    m = dm.Model("eqonly")
    a = m.continuous("a", lb=-9, ub=9)
    b = m.continuous("b", lb=-9, ub=9)
    c = m.continuous("c", lb=-9, ub=9)
    m.subject_to(b - (a + 1) == 0.0)  # cid 0, eq_rank 0
    m.subject_to(c - (2 * b) == 0.0)  # cid 1, eq_rank 1
    m.minimize(a**2 + b**2 + c**2)
    # the deterministic matching (C6) eliminates a (via cid 0) and b (via cid 1):
    # x-block a->0, b->1; n_x = 3, n_ineq = 0, so eq-dual base = 3; cids 0,1 -> 3,4.
    assert kkt_schur_indices(m) == [0, 1, 3, 4]


@pytest.mark.skip(
    reason="pounce's NLPResult exposes no field indicating the Schur block was "
    "taken (it falls back transparently), so a wrong block cannot be detected at "
    "solve time — only via the hand-checked index fixtures above. Documented gap; "
    "would need an upstream pounce stats field (pounce#180 timing/Schur hooks)."
)
def test_schur_path_taken_indicator_unavailable():  # pragma: no cover - documentation
    pass


def test_no_eliminable_block_gives_empty():
    # a model where lm eliminates nothing -> empty reducible block
    m = dm.Model("free")
    x = m.continuous("x", lb=-5, ub=5)
    m.minimize(x**2)
    assert kkt_schur_indices(m) == []


def test_deterministic():
    m = reaction_diffusion(16)
    assert kkt_schur_indices(m) == kkt_schur_indices(m)
