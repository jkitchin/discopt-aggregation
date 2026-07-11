"""Structural model diagnostics via Dulmage-Mendelsohn decomposition (plan item S1).

The coarse Dulmage-Mendelsohn decomposition of a model's *equality* incidence
(variables x equality constraints) reveals structural well-posedness — the same
analysis Parker, Nicholson, Siirola & Biegler use to debug nonlinear optimization
problems (arXiv / Comput. Chem. Eng. 2023).  It partitions the equality subsystem
into:

* **under-determined** — variables with no defining equality (the model's degrees
  of freedom / decision variables); normal for optimization.
* **square** — variables matched one-to-one with defining equalities; exactly the
  variables aggregation can eliminate explicitly.
* **over-determined** — more equality constraints than variables in a block: a
  structurally singular / redundant / inconsistent subsystem.  This is a *warning*
  sign — such a subsystem cannot be a valid square defining set for elimination.

Pure structural analysis (no solve, no discopt internals): it uses only the
incidence graph and a maximum matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from discopt.aggregation.incidence import build_incidence
from discopt.aggregation.matching import (
    DulmageMendelsohn,
    dulmage_mendelsohn,
    maximum_matching,
)


@dataclass
class StructuralDiagnosis:
    """Structural well-posedness of a model's equality subsystem.

    Attributes:
        dm: the coarse Dulmage-Mendelsohn decomposition (left = equality
            constraint ids, right = variable names).
        n_equality_constraints: number of equality constraints analysed.
        n_variables: number of candidate (scalar continuous) variables.
    """

    dm: DulmageMendelsohn
    n_equality_constraints: int
    n_variables: int
    matching: dict = field(default_factory=dict)

    @property
    def free_variables(self) -> list[str]:
        """The model's degrees of freedom: under-determined variables with no
        defining equality — i.e. the *unmatched* variables of the
        under-determined block, NOT every variable in it.

        The under-determined block also contains matched variables (each pinned by
        a defining equality *inside* the block); only the unmatched excess are
        genuine degrees of freedom.  For ``x + y == 1`` (one equality, two
        variables) this returns a single free variable, matching the one degree of
        freedom, rather than both variables.  The matching is deterministic
        (sorted iteration), so the returned set is stable across hash seeds.
        """
        matched = set(self.matching.values())
        return [v for v in self.dm.underdetermined[1] if v not in matched]

    @property
    def degrees_of_freedom(self) -> int:
        """Number of degrees of freedom: the under-determined block's excess of
        variables over equality constraints (well-defined regardless of which
        maximum matching was chosen)."""
        return len(self.dm.underdetermined[1]) - len(self.dm.underdetermined[0])

    @property
    def overdetermined_constraints(self) -> list[int]:
        """Equality-constraint ids in the structurally over-determined block."""
        return list(self.dm.overdetermined[0])

    @property
    def is_structurally_singular(self) -> bool:
        """True iff any equality constraint is structurally over-determined."""
        return bool(self.dm.overdetermined[0])

    def warnings(self) -> list[str]:
        """Human-readable structural warnings (empty if well-posed)."""
        out: list[str] = []
        over_c, over_v = self.dm.overdetermined
        if over_c:
            out.append(
                f"{len(over_c)} equality constraint(s) are structurally "
                f"over-determined (redundant or inconsistent): "
                f"{len(over_c)} constraints vs {len(over_v)} variables — "
                f"cids {over_c}"
            )
        return out


def structural_diagnosis(model: Any, pivot_tol: float = 1e-6) -> StructuralDiagnosis:
    """Dulmage-Mendelsohn structural diagnosis of ``model``'s equality subsystem.

    Builds the incidence graph, computes a maximum matching, and returns the
    coarse decomposition plus convenience accessors and warnings.  The *square*
    block's variables are exactly those aggregation can eliminate; the
    *over-determined* block flags a structurally singular subsystem.
    """
    inc = build_incidence(model, pivot_tol=pivot_tol)
    matching = maximum_matching(inc.graph)
    dm = dulmage_mendelsohn(inc.graph, matching)
    return StructuralDiagnosis(
        dm=dm,
        n_equality_constraints=len(inc.constraints),
        n_variables=len(inc.variables),
        matching=matching,
    )


__all__ = ["StructuralDiagnosis", "structural_diagnosis"]
