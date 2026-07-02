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

from dataclasses import dataclass
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

    @property
    def free_variables(self) -> list[str]:
        """Variables with no defining equality (the model's degrees of freedom)."""
        return list(self.dm.underdetermined[1])

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
    )


__all__ = ["StructuralDiagnosis", "structural_diagnosis"]
