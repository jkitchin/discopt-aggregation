"""Parameterized test problems for the aggregation study (plan item S1).

Each problem is a builder producing a discopt model with (a) intermediate
variables defined by equality constraints (so aggregation has something to
eliminate), (b) array variables (so scalarization is exercised), and (c) two
sweepable parameters — the substrate for the structural comparison (Table 4),
runtime breakdown (S2), and reliability sweeps (S3).

Intermediates are given finite (non-binding) bounds: discopt's solver is far
faster and more robust on bounded variables than on effectively-free ones, and
the aggregation bound-pruning (interval propagation) removes the resulting
box->inequality constraints when they are redundant, so aggregation stays clean.

Three problems span the paper's structure classes and MINLP/NLP kinds:
``reaction_diffusion`` (a 1-D diffusion-reaction least-squares collocation —
convex QP / NLP), ``unit_selection`` (on/off units with quadratic cost — a small
MIQP / MINLP, whose binaries aggregation never touches), and ``gas_pipeline`` (a
Weymouth gas line — a small nonconvex NLP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import discopt.modeling as dm


def reaction_diffusion(n: int = 8, k: float = 1.0, c_in: float = 1.0) -> Any:
    """1-D steady diffusion-reaction on a grid, as a least-squares collocation QP.

    State ``c[0..n]`` with Dirichlet boundaries ``c[0] = c_in``, ``c[n] = 0``.
    Each interior residual ``res[i] = c[i-1] - 2 c[i] + c[i+1] - h^2 k c[i]`` is a
    defined intermediate; the objective drives the residuals to zero (a convex
    QP).  ``k`` (reaction rate) and ``c_in`` (inlet) are the sweep parameters;
    ``n`` is a size knob.
    """
    m = dm.Model(f"reaction_diffusion_{n}")
    c = m.continuous("c", shape=(n + 1,), lb=-10.0, ub=10.0)
    res = m.continuous("res", shape=(n - 1,), lb=-100.0, ub=100.0)
    h = 1.0 / n
    m.subject_to(c[0] == float(c_in))
    m.subject_to(c[n] == 0.0)
    for i in range(1, n):
        m.subject_to(res[i - 1] - (c[i - 1] - 2 * c[i] + c[i + 1] - h * h * k * c[i]) == 0.0)
    m.minimize(sum(res[i] ** 2 for i in range(n - 1)))
    return m


def unit_selection(n: int = 3, demand: float = 3.0, fixed_cost: float = 2.0) -> Any:
    """On/off unit selection with quadratic operating cost — a small **MIQP**.

    Each unit ``i`` produces ``x[i] >= 0`` only if switched on (``b[i]`` binary,
    ``x[i] <= cap*b[i]``); its operating cost ``cost[i] = x[i]^2`` is a defined
    intermediate (degree-2, eliminable).  Total output must meet ``demand``;
    minimise operating cost plus a per-unit ``fixed_cost``.  Sweep parameters:
    ``demand`` and ``fixed_cost``.  The binaries are never aggregated.
    """
    m = dm.Model(f"unit_selection_{n}")
    cap = 5.0
    x = [m.continuous(f"x{i}", lb=0.0, ub=cap) for i in range(n)]
    cost = [m.continuous(f"cost{i}", lb=0.0, ub=cap * cap) for i in range(n)]
    b = [m.binary(f"b{i}") for i in range(n)]
    for i in range(n):
        m.subject_to(cost[i] - x[i] ** 2 == 0.0)  # defining (degree-2)
        m.subject_to(x[i] <= cap * b[i])  # on/off linking
    m.subject_to(sum(x) >= float(demand))
    m.minimize(sum(cost) + float(fixed_cost) * sum(b))
    return m


def gas_pipeline(n: int = 3, p_supply: float = 10.0, demand: float = 2.0) -> Any:
    """Weymouth gas line — a small **nonconvex NLP**.

    Node pressures ``p[0..n-1]`` (``p[0] = p_supply`` fixed); squared-pressure
    intermediates ``sq[i] = p[i]^2`` are defined (degree-2, eliminable); edge
    flows obey the Weymouth relation ``f[i]^2 = C (sq[i] - sq[i+1])``.  The last
    node must deliver ``demand``; minimise total flow.  Sweep parameters:
    ``p_supply`` and ``demand``.  Eliminating ``sq`` recovers the pressure-form
    Weymouth constraints ``f[i]^2 = C (p[i]^2 - p[i+1]^2)``.
    """
    m = dm.Model(f"gas_pipeline_{n}")
    p = [m.continuous(f"p{i}", lb=1.0, ub=10.0) for i in range(n)]
    sq = [m.continuous(f"sq{i}", lb=0.0, ub=100.0) for i in range(n)]
    f = [m.continuous(f"f{i}", lb=0.0, ub=10.0) for i in range(n - 1)]
    m.subject_to(p[0] == float(p_supply))
    for i in range(n):
        m.subject_to(sq[i] - p[i] ** 2 == 0.0)  # defining (degree-2)
    for i in range(n - 1):
        m.subject_to(f[i] ** 2 - 0.5 * (sq[i] - sq[i + 1]) == 0.0)  # Weymouth
    m.subject_to(f[n - 2] >= float(demand))
    m.minimize(sum(f))
    return m


def recycle_loop(n: int = 4, feed: float = 2.0, alpha: float = 0.2, beta: float = 0.7) -> Any:
    """Steady-state recycle loop solved as a tear-residual minimization (plan R1).

    A feed plus recycle ``R`` is split across ``n`` parallel reactor units; each
    unit applies one mild nonlinear conversion ``out[i] = in[i] - alpha in[i]^2``
    with ``in[i] = (feed + R)/n``.  The recycle produced is ``rec = beta*sum(out)``
    and steady state is ``R == rec``, imposed by minimizing the tear residual
    ``(R - rec)^2`` — the equation-oriented form of a sequential-modular recycle
    convergence.  Every ``in``/``out``/``rec`` is an intermediate defined by a
    (linearly-appearing) equality, so aggregation eliminates the whole loop down to
    a single decision variable ``R``.  Sweep parameters: ``feed`` and ``alpha``.

    Design note (why *parallel*, not a chain).  A **closed** recycle loop is one
    irreducible algebraic cycle — an implicit-elimination (E5/K1) case, not an
    explicit one.  A **deep nonlinear cascade** is explicitly eliminable but its
    substitution compounds to a degree-``2^n`` expression (the S2 Hessian-cost
    pathology), so aggregation hurts rather than helps.  Parallel units keep each
    eliminated intermediate low-degree in ``R`` (no compounding), so aggregation
    reduces the model cheaply and cuts solver iterations (~2x here) without an
    expression blow-up.
    """
    m = dm.Model(f"recycle_loop_{n}")
    r = m.continuous("R", lb=0.0, ub=5.0)
    inp = m.continuous("inp", shape=(n,), lb=0.0, ub=20.0)
    out = m.continuous("out", shape=(n,), lb=-20.0, ub=20.0)
    rec = m.continuous("rec", lb=-50.0, ub=50.0)
    for i in range(n):
        m.subject_to(inp[i] - (feed + r) / n == 0.0)  # split (linear, eliminable)
        m.subject_to(out[i] - (inp[i] - alpha * inp[i] ** 2) == 0.0)  # one nonlinear stage
    m.subject_to(rec - beta * sum(out[i] for i in range(n)) == 0.0)
    m.minimize((r - rec) ** 2)
    return m


def cstr_dynamic(n: int = 8, k: float = 1.0, c_in: float = 1.0) -> Any:
    """Time-discretized CSTR optimal control — a discretized-DAE problem.

    A dynamic continuous stirred-tank reactor with product-inhibited kinetics is
    discretized in time (backward Euler over ``n`` steps) — the structure class of
    the paper's distillation/moving-bed *dynamic optimization* problems (DAEs with
    rational nonlinearities).  The differential state is the concentration ``c[t]``;
    the control is a feed rate ``u[t]``.  The reaction rate
    ``rate[t] = k c[t] / (1 + 0.5 c[t])`` is an *algebraic* variable defined by a
    two-variable degree-2 equation (``rate`` linear, ``c`` rational), so degree-2
    aggregation eliminates the algebraic rates — exactly the reduced-space move of
    keeping differential states and substituting algebraics.  The four-variable
    dynamics ``c[t] = c[t-1] + dt(u[t] - rate[t])`` are *not* degree-2 eliminable,
    so the states do not chain-compound (no expression blow-up).  The objective
    tracks a setpoint with a control-effort penalty.  Sweep parameters: ``k`` and
    the feed level ``c_in``; ``n`` is the horizon length.
    """
    m = dm.Model(f"cstr_dynamic_{n}")
    c = m.continuous("c", shape=(n,), lb=0.0, ub=10.0)
    u = m.continuous("u", shape=(n,), lb=-5.0, ub=5.0)
    rate = m.continuous("rate", shape=(n,), lb=-50.0, ub=50.0)
    dt = 1.0 / n
    m.subject_to(c[0] == float(c_in))  # initial condition
    for t in range(n):
        m.subject_to(rate[t] - (k * c[t] / (1.0 + 0.5 * c[t])) == 0.0)  # algebraic (degree-2)
    for t in range(1, n):
        m.subject_to(c[t] - (c[t - 1] + dt * (u[t] - rate[t])) == 0.0)  # dynamics (4 vars)
    target = 0.3
    m.minimize(sum((c[t] - target) ** 2 + 0.1 * u[t] ** 2 for t in range(n)))
    return m


@dataclass
class TestProblem:
    """A parameterized test problem.

    Attributes:
        name: identifier.
        builder: ``builder(n=..., **params) -> Model``.
        nominal: nominal values of the sweep parameters.
        sweep: ``{param: (lo, hi)}`` ranges for the reliability sweep (S3).
        default_size: default ``n`` for the size knob.
    """

    name: str
    builder: Callable[..., Any]
    nominal: dict[str, float]
    sweep: dict[str, tuple[float, float]]
    default_size: int = 8

    def build(self, n: int | None = None, **overrides: float) -> Any:
        """Build the model at ``nominal`` parameters, with optional overrides."""
        params = {**self.nominal, **overrides}
        return self.builder(n=self.default_size if n is None else n, **params)

    def sweep_grid(self, points: int = 11) -> list[dict[str, float]]:
        """Cartesian grid of parameter dicts over the sweep ranges (S3 substrate)."""
        axes = {}
        for name, (lo, hi) in self.sweep.items():
            step = (hi - lo) / (points - 1) if points > 1 else 0.0
            axes[name] = [lo + step * i for i in range(points)]
        grid: list[dict[str, float]] = [{}]
        for name, values in axes.items():
            grid = [{**g, name: v} for g in grid for v in values]
        return grid


REACTION_DIFFUSION = TestProblem(
    name="reaction_diffusion",
    builder=reaction_diffusion,
    nominal={"k": 1.0, "c_in": 1.0},
    sweep={"k": (0.1, 5.0), "c_in": (0.0, 2.0)},
)

UNIT_SELECTION = TestProblem(
    name="unit_selection",
    builder=unit_selection,
    nominal={"demand": 3.0, "fixed_cost": 2.0},
    sweep={"demand": (1.0, 5.0), "fixed_cost": (0.5, 4.0)},
    default_size=3,
)

GAS_PIPELINE = TestProblem(
    name="gas_pipeline",
    builder=gas_pipeline,
    nominal={"p_supply": 10.0, "demand": 2.0},
    sweep={"p_supply": (6.0, 10.0), "demand": (0.5, 4.0)},
    default_size=3,
)

RECYCLE_LOOP = TestProblem(
    name="recycle_loop",
    builder=recycle_loop,
    nominal={"feed": 2.0, "alpha": 0.2},
    sweep={"feed": (1.0, 3.0), "alpha": (0.1, 0.3)},
    default_size=4,
)

CSTR_DYNAMIC = TestProblem(
    name="cstr_dynamic",
    builder=cstr_dynamic,
    nominal={"k": 1.0, "c_in": 1.0},
    sweep={"k": (0.2, 3.0), "c_in": (0.5, 2.0)},
    default_size=8,
)

PROBLEMS: dict[str, TestProblem] = {
    p.name: p
    for p in (REACTION_DIFFUSION, UNIT_SELECTION, GAS_PIPELINE, RECYCLE_LOOP, CSTR_DYNAMIC)
}


__all__ = [
    "TestProblem",
    "reaction_diffusion",
    "unit_selection",
    "gas_pipeline",
    "recycle_loop",
    "cstr_dynamic",
    "REACTION_DIFFUSION",
    "UNIT_SELECTION",
    "GAS_PIPELINE",
    "RECYCLE_LOOP",
    "CSTR_DYNAMIC",
    "PROBLEMS",
]
