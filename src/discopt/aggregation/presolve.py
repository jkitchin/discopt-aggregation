"""Interaction with discopt's native presolve, and composition safety (plan item S3).

discopt already ships a variable-elimination presolve (Rust,
``discopt-core/src/presolve``).  Because this plugin runs as a *pre-pass* — it
produces a reduced model that discopt then solves (and presolves natively) — it is
worth documenting how the two overlap and confirming they compose safely.

**Overlap with discopt's native passes** (see :func:`native_presolve_overlap`):

* ``eliminate_variables`` (M10) — fixes a *singleton*-equality continuous variable
  (``coeff·v + const == rhs``, ``v`` the only variable): the plugin's ``ld1``.
* ``aggregate_variables`` (C1) — affine substitution of ``x`` in
  ``c_x·x + c_y·y + c0 == rhs`` where ``x`` appears *only in this equality*
  (write-only): a restricted ``ld2``/``ecd2``.
* ``factorable_eliminate`` (C2) — frees ``v`` determined by
  ``coeff·v + g(others) == rhs`` (``g`` factorable): the multi-variable case, but
  it only *drops the equation and frees v* (recovered post-solve), it does not
  substitute ``v`` into the rest of the model.

The unifying restriction: discopt's native passes act on variables that appear
**only in their defining equation** (or are pinned to a constant).  The plugin
instead **substitutes the defining expression into every occurrence** (objective
and other constraints), so it eliminates the *used* intermediates the native
passes leave — the two are **complementary, not redundant**.

**Recommended ordering:** run the plugin aggregation **first** (a pre-pass), then
let discopt solve; its native presolve mops up any remaining singleton/write-only
variables on the already-reduced model.  There is no double-elimination conflict:
a variable the plugin removed is simply absent for the native pass.  This is
verified empirically by :func:`verify_composition` (aggregate-then-solve matches a
direct solve).

**Implied integrality (scoped out, documented).**  Per
van der Hulst & Walter (2025), a presolve should avoid hiding a variable whose
value is implied-integer.  For this plugin that guard would essentially never
fire: aggregation *never eliminates integer/binary variables* (only scalar
continuous candidates — enforced by ``candidate_variables``), and eliminating a
continuous equality-defined intermediate is exact / equivalence-preserving (same
feasible set and optimum).  So the integrality a solver would branch on lives on
variables the plugin does not touch.  We keep the sound invariant (never remove an
integrality-carrying variable) rather than build a guard that cannot trigger —
the same call made for G4 (convexity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PresolveOverlap:
    """How one native discopt presolve pass relates to the plugin."""

    native_pass: str
    native_does: str
    plugin_analog: str
    relationship: str  # "subset" (native ⊂ plugin) or "complementary"


_OVERLAP = [
    PresolveOverlap(
        native_pass="eliminate_variables (M10)",
        native_does="fix a singleton-equality continuous variable (only variable in the eq)",
        plugin_analog="ld1 (fixed variable)",
        relationship="subset",
    ),
    PresolveOverlap(
        native_pass="aggregate_variables (C1)",
        native_does="affine-substitute a write-only variable (appears only in one equality)",
        plugin_analog="ld2 / ecd2 (restricted to write-only)",
        relationship="subset",
    ),
    PresolveOverlap(
        native_pass="factorable_eliminate (C2)",
        native_does="free a variable determined by a factorable equation; drop the equation only",
        plugin_analog="gr / lm (but plugin substitutes into all occurrences)",
        relationship="complementary",
    ),
]


def native_presolve_overlap() -> list[PresolveOverlap]:
    """The documented overlap between the plugin and discopt's native presolve."""
    return list(_OVERLAP)


def recommended_ordering() -> str:
    """The recommended composition order, with rationale."""
    return (
        "Run plugin aggregation first (pre-pass), then discopt.solve(). The plugin "
        "substitutes *used* intermediates the native passes cannot; discopt's native "
        "presolve then removes any remaining singleton/write-only variables on the "
        "reduced model. No double-elimination conflict — a plugin-removed variable is "
        "absent for the native pass. Composition is exact (verify_composition)."
    )


@dataclass
class CompositionCheck:
    """Result of comparing a direct solve with an aggregate-then-solve."""

    method: str
    direct_objective: float
    aggregated_objective: float
    matches: bool
    n_eliminated: int


def verify_composition(model: Any, method: str = "gr", atol: float = 1e-5) -> CompositionCheck:
    """Check plugin-aggregate-then-solve matches a direct discopt solve.

    Confirms the plugin composes safely with discopt's native presolve (which runs
    during the reduced-model solve): same objective, no conflict from overlapping
    eliminations.  Neither ``model.solve()`` nor ``aggregate`` mutates ``model``.
    """
    from discopt.aggregation.api import solve as aggregated_solve

    direct = model.solve()
    result = aggregated_solve(model, method=method)
    do = float(direct.objective)
    ao = float(result.objective)
    return CompositionCheck(
        method=method,
        direct_objective=do,
        aggregated_objective=ao,
        matches=abs(do - ao) <= atol,
        n_eliminated=result.aggregation.n_eliminated,
    )


__all__ = [
    "PresolveOverlap",
    "native_presolve_overlap",
    "recommended_ordering",
    "CompositionCheck",
    "verify_composition",
]
