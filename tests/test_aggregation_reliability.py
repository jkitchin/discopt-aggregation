"""RL2: the bounded search for a tuned-to-fail reliability instance — an honest
negative, pinned as regression tests.

The paper's headline is that aggregation converges *more* instances.  A bounded
search (scratch scripts, ~30 min compute) over ``gas_pipeline`` near
infeasibility, a badly-scaled ``cstr_dynamic``, and a closed recycle loop found
**no sound** demonstration of that headline on this stack:

* **Explicit** structure-preserving aggregation (``d2``/``lm``) never converges
  strictly more instances than the full space, and on the closed recycle it
  converges *fewer* — substituting the intermediates raises the nonlinearity of
  the surviving equation, so the reduced model needs *more* Newton iterations
  (12 vs 10 at the nominal recycle point), losing cells under a tight cap.  It is
  *sound* (POUNCE solves the bounded reduced model to the correct root).

* The **implicit** path (``eliminate_implicit`` on the closed recycle) does show
  a large *raw* convergence gap — the full space hits a tight iteration cap on
  most of a 7x7 grid while the inner Newton always solves the cycle — but that
  gap is **confounded**: the nonconvex conversion gives the recycle equation two
  roots, and the bound-unaware inner solve lands on the spurious one on ~43% of
  cells (recovering ``R = -3.86`` at the nominal point, violating ``R >= 0``,
  where the full space finds ``R = 9.33``).  It "converges" by reporting a
  *different, sometimes unphysical* solution, so it does not demonstrate the
  paper's claim.

Hence no ``HARD_INSTANCE`` is registered and ``recycle_cycle`` is kept out of the
``PROBLEMS`` reliability corpus.  These tests pin the honest-negative findings so
a future change that (accidentally or genuinely) alters them is caught.  The
quantitative grid sweep is ``slow``; the structural/soundness facts are fast.

See ``design/aggregation-stage2-results.md`` (Reliability results) and item RL2
in ``design/aggregation-gaps-and-integer-plan.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from discopt.aggregation.api import aggregate
from discopt.aggregation.benchmark import _local_solve
from discopt.aggregation.implicit import eliminate_implicit
from discopt.aggregation.recovery import evaluate_expression
from discopt.aggregation.testproblems import PROBLEMS, RECYCLE_CYCLE, recycle_cycle

pytestmark = pytest.mark.unit

RL2_CAP = 9  # full space needs ~10 Newton iters on this cycle; 9 caps the harder cells


def _max_constraint_residual(model, point: dict) -> float:
    """Largest |body| over the model's equality constraints at ``point``."""
    res = 0.0
    for c in model._constraints:
        val = float(np.asarray(evaluate_expression(c.body, point)).reshape(()))
        res = max(res, abs(val))
    return res


def _recovered_R(model) -> float:
    """R from the implicit inner Newton solve of the (0-variable) reduced cycle."""
    el = eliminate_implicit(model)
    assert not el.reduced_model._variables  # closed cycle: fully determined
    return float(el.recover({})["R"])


def _full_space_R(model) -> tuple[bool, float]:
    """(converged-uncapped, R) from the full-space local solve (obj = 1e-6 R^2)."""
    st, _it, _wt, obj = _local_solve(model)
    return st.endswith("OPTIMAL"), float(np.sqrt(max(obj, 0.0) / 1e-6))


@pytest.mark.unit
def test_recycle_cycle_is_not_in_the_reliability_corpus():
    """The RL2 probe is exported but deliberately excluded from ``PROBLEMS``.

    Its convergence gap is confounded (see below), so seeding it into the
    reliability corpus would advertise an unsound win.  ``RECYCLE_CYCLE`` is a
    documented ``TestProblem`` for study/repro only.
    """
    assert "recycle_cycle" not in PROBLEMS
    assert RECYCLE_CYCLE.name == "recycle_cycle"
    assert set(RECYCLE_CYCLE.sweep) == {"feed", "alpha"}


@pytest.mark.smoke
def test_implicit_recovery_on_closed_recycle_is_confounded():
    """The implicit inner Newton recovers a *spurious* root at the nominal point.

    The closed recycle's nonconvex conversion admits two roots.  The full space
    (bounded, ``R >= 0``) finds ``R ~ 9.33``; the bound-unaware inner Newton
    recovers ``R ~ -3.86`` — a valid root of the equations (tiny residual) but
    negative, violating the box.  This is why the implicit "convergence gap" is
    not a sound reliability demonstration.
    """
    m = recycle_cycle()  # nominal: feed=2, alpha=0.2, beta=1.9

    converged_full, r_full = _full_space_R(m)
    assert converged_full
    assert r_full == pytest.approx(9.3316, abs=1e-3)
    assert r_full >= 0.0  # full space respects the R >= 0 bound

    r_rec = _recovered_R(m)
    # the recovered point satisfies every defining equation (a genuine root) ...
    el = eliminate_implicit(m)
    assert _max_constraint_residual(m, el.recover({})) < 1e-6
    # ... yet it is the *other*, negative root — unphysical, and != the full-space
    # optimum, so counting it as "converged" would misreport the reliability win.
    assert r_rec == pytest.approx(-3.8579, abs=1e-3)
    assert r_rec < 0.0
    assert abs(r_rec - r_full) > 1.0


@pytest.mark.smoke
def test_explicit_d2_on_closed_recycle_is_sound_but_not_more_reliable():
    """Explicit ``d2`` is sound on the closed recycle but needs *more* iterations.

    ``d2`` eliminates the split/conversion intermediates (8 of 9 variables),
    leaving a 1-variable bounded reduced model that POUNCE solves to the *correct*
    root (``R ~ 9.33``, no spurious negative — bounds are enforced).  But the
    substitution raises the surviving equation's nonlinearity, so the reduced
    solve takes at least as many Newton iterations as the full space — the
    opposite of a reliability win.
    """
    m = recycle_cycle()
    res = aggregate(m, method="d2")
    assert res.n_eliminated == 8
    assert len(res.reduced_model._variables) == 1

    st_full, it_full, _wt, obj_full = _local_solve(m)
    st_red, it_red, _wt, obj_red = _local_solve(res.reduced_model)
    assert st_full.endswith("OPTIMAL") and st_red.endswith("OPTIMAL")
    # same (correct, bounded) optimum -> sound
    assert obj_red == pytest.approx(obj_full, abs=1e-6)
    assert np.sqrt(max(obj_red, 0.0) / 1e-6) == pytest.approx(9.3316, abs=1e-3)
    # but no iteration saving: the reduced model is at least as hard to solve
    assert it_red >= it_full


@pytest.mark.smoke
@pytest.mark.slow
def test_rl2_honest_negative_grid(_points: int = 7):
    """The full RL2 grid sweep: the only convergence gap is unsound (deterministic).

    On a 7x7 ``feed`` x ``alpha`` grid at ``max_iter = RL2_CAP``:

    * the full space fails on most cells (it needs ~10 iterations; the cap is 9),
    * explicit ``d2`` converges on **no more** cells than the full space
      (structure-preserving aggregation offers no reliability gain — a loss here),
    * the implicit inner Newton *recovers a root on every cell* (the raw gap),
    * but that recovery is **confounded** on a large fraction of cells (spurious
      root / bound violation), so it is not a sound win.

    Deterministic on this stack (rerun x3 gives identical counts — the C6
    determinism guarantee); ``slow`` and deselected from the default run.
    """
    grid = RECYCLE_CYCLE.sweep_grid(points=_points)
    n = len(grid)
    assert n == _points * _points

    orig_conv = d2_conv = implicit_recovered = confounded = 0
    for params in grid:
        m = recycle_cycle(**params)

        st_o, *_ = _local_solve(m, options={"max_iter": RL2_CAP})
        orig_conv += st_o.endswith("OPTIMAL")

        rm = aggregate(m, method="d2").reduced_model
        st_d, *_ = _local_solve(rm, options={"max_iter": RL2_CAP})
        d2_conv += st_d.endswith("OPTIMAL")

        el = eliminate_implicit(m)
        point = el.recover({})
        if _max_constraint_residual(m, point) < 1e-6:
            implicit_recovered += 1
        _ok, r_full = _full_space_R(m)
        if point["R"] < -1e-6 or abs(point["R"] - r_full) > 1e-3:
            confounded += 1

    # the full space genuinely fails under the cap (raw difficulty is real)
    assert 0 < orig_conv < n
    # explicit aggregation is NOT more reliable (no sound explicit win)
    assert d2_conv <= orig_conv
    # the implicit inner Newton always "solves" the cycle -> a raw convergence gap
    assert implicit_recovered == n
    # ... but that gap is confounded on a large fraction of cells -> unsound
    assert confounded >= n // 4
