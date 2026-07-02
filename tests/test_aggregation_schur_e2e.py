"""K2 end-to-end (smoke): the aggregation Schur block through discopt's passthrough.

Uses the *supported* API — `solve_nlp_from_model(model, x0, kkt_schur_block=...)`
(discopt#392) — fed by the plugin's `kkt_schur_indices`. No solver internals. The
Schur partition is correctness-safe (pounce falls back if unsuitable), so the
solution must be identical to a baseline solve.

Skips automatically when the running discopt lacks the passthrough, so it stays
green regardless of the discopt checkout and activates once it is present.
"""

from __future__ import annotations

import inspect

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.aggregation.scalarize import scalarize_model
from discopt.aggregation.schur import kkt_schur_indices
from discopt.aggregation.testproblems import reaction_diffusion

pytestmark = pytest.mark.smoke

try:
    from discopt.solvers.nlp_pounce import solve_nlp_from_model

    _HAS_PASSTHROUGH = "kkt_schur_block" in inspect.signature(solve_nlp_from_model).parameters
except ImportError:  # pragma: no cover - pounce backend unavailable
    solve_nlp_from_model = None  # type: ignore[assignment]
    _HAS_PASSTHROUGH = False


@pytest.mark.skipif(
    not _HAS_PASSTHROUGH,
    reason="discopt solve_nlp_from_model lacks kkt_schur_block (needs discopt#392)",
)
def test_schur_block_solves_identically():
    sm, _ = scalarize_model(reaction_diffusion(30))
    n = len(sm._variables)
    x0 = np.zeros(n)
    idx = kkt_schur_indices(sm)
    assert len(idx) > 0  # the reducible block is non-empty for this problem

    base = solve_nlp_from_model(sm, x0)
    schur = solve_nlp_from_model(sm, x0, kkt_schur_block=idx)

    assert str(base.status).endswith("OPTIMAL")
    assert str(schur.status).endswith("OPTIMAL")
    # correctness-safe: identical objective (Schur only changes fill/time)
    assert float(schur.objective) == pytest.approx(float(base.objective), abs=1e-4)


@pytest.mark.skipif(
    not _HAS_PASSTHROUGH,
    reason="discopt solve_nlp_from_model lacks kkt_schur_block (needs discopt#392)",
)
def test_schur_block_on_mixed_eq_ineq_model():
    # a model with inequalities exercises the n_ineq slack offset in the eq-dual
    # block; the Schur block must still solve to the same optimum.
    m = dm.Model("mix")
    c = m.continuous("c", shape=(31,), lb=-10, ub=10)
    res = m.continuous("res", shape=(29,), lb=-100, ub=100)
    m.subject_to(c[0] == 1.0)
    m.subject_to(c[30] == 0.0)
    for i in range(1, 30):
        m.subject_to(res[i - 1] - (c[i - 1] - 2 * c[i] + c[i + 1]) == 0.0)
    m.subject_to(c[1] <= 9.0)  # inequality -> n_ineq > 0
    m.subject_to(c[29] >= -9.0)
    m.minimize(sum(res[i] ** 2 for i in range(29)))

    sm, _ = scalarize_model(m)
    n = len(sm._variables)
    x0 = np.zeros(n)
    idx = kkt_schur_indices(sm)
    assert len(idx) > 0

    base = solve_nlp_from_model(sm, x0)
    schur = solve_nlp_from_model(sm, x0, kkt_schur_block=idx)
    assert str(base.status).endswith("OPTIMAL")
    assert str(schur.status).endswith("OPTIMAL")
    assert float(schur.objective) == pytest.approx(float(base.objective), abs=1e-4)
