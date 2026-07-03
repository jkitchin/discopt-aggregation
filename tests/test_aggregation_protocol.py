"""P2/P3: protocol-scale sweep + per-callback cost breakdown.

The protocol-scale reliability sweep (P2) is marked ``slow`` and deselected from
the default run (``addopts = -m 'not slow'``); run it with ``pytest -m slow``.
The callback-breakdown test (P3) is a fast per-evaluation cost measurement.
"""

from __future__ import annotations

import pytest

from discopt.aggregation.benchmark import callback_breakdown
from discopt.aggregation.testproblems import reaction_diffusion


def test_callback_breakdown_well_formed():
    """P3: the per-callback cost breakdown is well-formed for every method.

    The fractions sum to 1 and each is in [0, 1]; the Hessian/function ratio is a
    finite positive number.  (The paper's *trend* — Hessian cost dominating under
    aggressive aggregation — is real but only robust at DAE scale; at these tiny
    sizes per-evaluation timing is dominated by Python/dispatch noise, an honest
    limitation of a per-evaluation proxy for solver-internal time.)
    """
    for method in ("original", "d2", "gr"):
        b = callback_breakdown(reaction_diffusion(8), method=method)
        assert abs(b["func"] + b["grad"] + b["jac"] + b["hess"] - 1.0) < 1e-9
        assert all(0.0 <= b[k] <= 1.0 for k in ("func", "grad", "jac", "hess"))
        assert b["hess_over_func"] > 0.0


@pytest.mark.slow
def test_protocol_scale_reliability_sweep(tmp_path):
    """P2: a protocol-scale grid sweep, per method, with per-cell CSV + virtual best.

    Runs a 7x7 grid on the fast nonconvex ``gas_pipeline`` here so ``pytest -m
    slow`` completes in a few minutes; raise ``points`` to 11 (and add problems)
    for the paper's full 121-point protocol.  Structure-preserving methods only:
    gr/lm reduced models densify (and on a chain the substitution can blow the
    expression up during incidence analysis) — the paper's own cautionary
    finding, so they belong in a guarded, opt-in run, not the default protocol.
    """
    from discopt.aggregation.sweep import reliability_sweep, sweep_to_csv, virtual_best
    from discopt.aggregation.testproblems import GAS_PIPELINE

    points = 7
    methods = ("ld1", "d2")
    res = reliability_sweep(GAS_PIPELINE, methods=methods, points=points)
    for r in res.values():
        assert len(r.cells) == points * points
    vb = virtual_best(res)
    # headline direction: the virtual best is at least as good as any single method
    assert vb.n_converged >= max(r.n_converged for r in res.values())
    out = tmp_path / "gas_pipeline_sweep.csv"
    sweep_to_csv(res, str(out))
    # one header row + one row per (method incl. original) per cell
    assert out.exists()
    assert len(out.read_text().splitlines()) == (len(methods) + 1) * points * points + 1
