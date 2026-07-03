"""T6: guard-interplay and edge-path coverage.

These exercise combinations and error/edge paths that individual feature tests
skipped: multiple guards at once, guards with exact tearing, integer arrays
through scalarization, and small error branches.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.aggregation.api import aggregate, solve

pytestmark = pytest.mark.unit


def _fan_model(n: int = 4):
    """A shallow fan: n independent intermediates y_i = a*x_i + b*x_i (2-var each),
    each also used in a shared target — exercises fill/guards without the deep
    nonlinear composition that stresses expression size."""
    m = dm.Model("fan")
    xs = [m.continuous(f"x{i}", lb=-5, ub=5) for i in range(n)]
    ys = [m.continuous(f"y{i}", lb=-50, ub=50) for i in range(n)]
    w = m.continuous("w", lb=-500, ub=500)
    for i in range(n):
        m.subject_to(ys[i] - (2.0 * xs[i] + 1.0) == 0.0)  # linear 2-var def
    m.subject_to(w - sum(ys[i] * xs[i] for i in range(n)) == 0.0)  # ys used nonlinearly
    m.minimize(w**2 + sum(x**2 for x in xs))
    return m


def test_max_fill_with_lm_and_recursive():
    """max_fill combined with lm + recursive (previously only tested with gr)."""
    base = aggregate(_fan_model(), method="lm", recursive=True)
    capped = aggregate(_fan_model(), method="lm", recursive=True, max_fill=0)
    assert capped.n_eliminated <= base.n_eliminated


def test_all_three_guards_together_stay_sound():
    """max_fill + max_def_nodes + max_condition applied at once stays sound."""
    agg = aggregate(
        _fan_model(4),
        method="gr",
        max_fill=1,
        max_def_nodes=50,
        max_condition=1e6,
    )
    res = solve(_fan_model(4), method="gr", max_fill=1, max_def_nodes=50, max_condition=1e6)
    assert "OPTIMAL" in str(res.status).upper() or res.status == "optimal"
    assert agg.n_eliminated >= 0


def test_exact_tearing_with_guards():
    """tearing='exact' composes with max_def_nodes without error."""
    m = dm.Model("cyc")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    t = m.continuous("t", lb=-9, ub=9)
    m.subject_to(x - (0.5 * y + 3.0) == 0.0)
    m.subject_to(y - (0.5 * x + 2.0) == 0.0)
    m.subject_to(t - (x + 1.0) == 0.0)
    m.minimize(x**2 + y**2 + t**2)
    agg = aggregate(m, method="lm", tearing="exact", max_def_nodes=100)
    assert agg.n_eliminated >= 1


def test_integer_array_survives_scalarization_and_solves():
    """An integer array is rebuilt un-scalarized; the model still aggregates+solves."""
    m = dm.Model("intarr")
    c = m.continuous("c", shape=(2,), lb=-5, ub=5)
    k = m.integer("k", shape=(2,), lb=0, ub=3)
    s = m.continuous("s", lb=-20, ub=20)
    m.subject_to(s - (c[0] + c[1]) == 0.0)  # eliminable continuous intermediate
    m.subject_to(k[0] + k[1] >= 1)
    m.minimize(c[0] ** 2 + c[1] ** 2 + s**2 + k[0] + k[1])
    res = solve(m, method="d2")
    assert "OPTIMAL" in str(res.status).upper() or res.status == "optimal"
    assert "k" in res.x  # integer array preserved (not scalarized)


def test_balanced_sum_empty_raises():
    from discopt.aggregation.profiling import balanced_sum

    with pytest.raises(ValueError, match="at least one term"):
        balanced_sum([])


def test_exact_hessian_nnz_none_when_evaluator_fails(monkeypatch):
    from discopt.aggregation import metrics

    m = dm.Model("h")
    x = m.continuous("x", lb=-5, ub=5)
    m.subject_to(x**2 == 1.0)
    m.minimize(x)

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("cannot build")

    monkeypatch.setattr("discopt._jax.nlp_evaluator.NLPEvaluator", _Boom)
    assert metrics.exact_hessian_nnz(m) is None  # degrades gracefully


def test_objectiveless_model_structure_and_aggregate():
    """A model with no objective still yields structure and aggregates."""
    from discopt.aggregation.metrics import model_structure

    m = dm.Model("noobj")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(y - (x + 1.0) == 0.0)
    s = model_structure(m)
    assert s.n_variables == 2 and s.nnz_hessian == 0
    agg = aggregate(m, method="ld1")
    assert "y" in agg.eliminated or agg.n_eliminated == 0


def test_fully_determined_sweep_instance():
    """sweep's fully-determined branch: every variable eliminated by aggregation."""
    from discopt.aggregation.recovery import evaluate_expression

    m = dm.Model("det")
    a = m.continuous("a", lb=-10, ub=10)
    b = m.continuous("b", lb=-10, ub=10)
    m.subject_to(a - 1.0 == 0.0)
    m.subject_to(b - (a + 1.0) == 0.0)
    m.minimize(a + b)
    res = solve(m, method="ld1")
    assert res.status == "optimal"
    assert res.reduced_result is None  # solved without a solver call
    assert float(evaluate_expression(m._objective.expression, res.x)) == pytest.approx(3.0)
    _ = np  # keep numpy import used
