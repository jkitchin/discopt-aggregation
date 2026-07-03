"""Aggregation-set selection strategies (paper Section 3, Algorithms 1-6).

Each strategy inspects a model's incidence graph and returns an
:class:`AggregationSet`: an *ordered* list of variable eliminations whose
defining equations form a strictly lower-triangular system (Lemma 1), so the
eliminated variables can be substituted explicitly and recovered post-solve
without any iterative solve.

Six strategies are provided, from most conservative to most aggressive:

===== ============================ ================================ ============
key   name                         defining constraints eliminated   paper alg.
===== ============================ ================================ ============
ld1   fixed-variable (degree-1)    ``y = a``                        Alg. 3
ecd2  equal-coefficient degree-2   ``y = x + a`` (``|coef|`` equal)  Alg. 6
ld2   linear degree-2              ``y = a*x + b``                   Alg. 5
d2    degree-2                     ``y = f(x)`` (<= 2 variables)     Alg. 4
gr    greedy                       ``y = f(w, x, ...)`` (linear y)   Alg. 1
lm    linear-matching              ``y = f(w, x, ...)`` (linear y)   Alg. 2
===== ============================ ================================ ============

``gr`` and ``lm`` are the approximate-maximum strategies; ``ld1``/``ecd2``/
``ld2``/``d2`` are structure-preserving (each eliminating a variable using a
constraint of at most two variables, so no remaining constraint gains
variables).  The paper recommends ``d2`` as the best balance of convergence
benefit vs. Hessian cost.

The lower-triangular guarantee is enforced structurally: ``gr`` builds the order
incrementally (Alg. 1), ``lm`` block-triangularizes and applies ``gr`` to any
cyclic block (Alg. 2), and :func:`_finalize` re-derives the topological order and
drops any residual cyclic block so the returned set is *always* valid.

References
----------
Naik, Biegler, Bent, Parker, arXiv:2502.13869, Section 3 and Table 2.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from discopt._jax.dependent_vars import _collect_var_names, _isolated_affine_coeff
from discopt.modeling.core import Constraint

from discopt.aggregation.incidence import IncidenceModel, build_incidence
from discopt.aggregation.matching import (
    BipartiteGraph,
    block_triangularize,
    maximum_matching,
    topological_order,
)

# Blocks with more than this many variables fall back to greedy tearing
# (the exact maximum-acyclic-subset search is exponential in the block size).
_EXACT_TEAR_CAP = 13


@dataclass(frozen=True)
class Elimination:
    """A single variable elimination.

    Attributes:
        var: name of the eliminated (defined) variable.
        cid: index of the defining equality constraint in ``model._constraints``.
        coeff: constant coefficient of ``var`` in the constraint body, so the
            equality ``coeff*var + rest(others) == 0`` solves to
            ``var = -rest(others) / coeff``.
    """

    var: str
    cid: int
    coeff: float


@dataclass
class AggregationSet:
    """An ordered set of variable eliminations for one model.

    ``eliminations`` are in topological (inputs-first) order: the defining
    equation of ``eliminations[i]`` references, among *eliminated* variables,
    only ``eliminations[:i]`` — this is exactly the strict lower-triangularity of
    Lemma 1.
    """

    eliminations: list[Elimination]
    incidence: IncidenceModel
    method: str
    n_dropped: int = 0  # matched pairs discarded because they formed a cycle
    stats: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.eliminations)

    def matching(self) -> dict[int, str]:
        """The selected aggregation as a ``{cid: var}`` matching."""
        return {e.cid: e.var for e in self.eliminations}

    def variables(self) -> list[str]:
        """Names of the eliminated variables, in elimination order."""
        return [e.var for e in self.eliminations]


# --------------------------------------------------------------------------- #
# core routines                                                                #
# --------------------------------------------------------------------------- #
def _greedy(
    inc: IncidenceModel, cids: list[int], restrict: set[str] | None = None
) -> dict[int, str]:
    """Greedy aggregation (paper Algorithm 1).

    Walk the constraints in the given order; for each, eliminate the first
    linear variable that has not yet appeared in any previously-chosen
    constraint.  Accumulating *all* of a chosen constraint's variables into the
    used-set ``S`` keeps the induced incidence lower triangular by construction.

    When ``restrict`` is given (Algorithm 2 step 6b: greedy *on a block*), only
    variables in that set are eligible and only they enter ``S`` — so a block's
    tear cannot "steal" a variable matched in another block, which would collapse
    to a duplicate and corrupt the Theorem-2 bookkeeping.

    Returns a ``{cid: var}`` matching.
    """
    matching: dict[int, str] = {}
    used: set[str] = set()
    for cid in cids:
        lin = inc.constraint_linear_vars(cid)
        allv = inc.constraint_vars(cid)
        if restrict is not None:
            lin = lin & restrict
            allv = allv & restrict
        for x in sorted(lin):  # deterministic
            if x not in used:
                matching[cid] = x
                used |= allv  # every (in-scope) var in this constraint
                break
    return matching


def _finalize(matching: dict[int, str], inc: IncidenceModel, method: str) -> AggregationSet:
    """Order a matching topologically and drop any residual cyclic block.

    Runs :func:`block_triangularize` over the *full* incidence (linear and
    nonlinear edges), so a nonlinear occurrence of one eliminated variable in
    another's defining equation is also treated as a dependency.  Singleton
    blocks become eliminations in topological order; any block of size > 1 is a
    cycle that cannot be explicitly eliminated and is dropped.

    When ``inc.max_fill`` is set, singleton eliminations whose estimated fill
    (:meth:`IncidenceModel.fill_estimate`) exceeds the cap are also dropped.
    Dropping any subset of a lower-triangular set leaves it lower triangular
    (the remaining eliminations then depend only on earlier or kept variables),
    so the guard is always sound.
    """
    blocks = block_triangularize(matching, inc.graph.adj_left)
    eliminations: list[Elimination] = []
    n_dropped = 0
    for block in blocks:
        if len(block) == 1:
            cid, var = block[0]
            coeff = inc.linear_coeff.get((cid, var))
            if coeff is None:  # matched on a non-linear edge -> cannot substitute
                n_dropped += 1
                continue
            if inc.max_fill is not None and inc.fill_estimate(cid, var) > inc.max_fill:
                n_dropped += 1
                continue
            eliminations.append(Elimination(var=var, cid=cid, coeff=coeff))
        else:
            n_dropped += len(block)
    return AggregationSet(
        eliminations=eliminations, incidence=inc, method=method, n_dropped=n_dropped
    )


def _constraint_is_linear(constraint: Constraint) -> bool:
    """Whether every variable in ``constraint`` participates linearly (affine body)."""
    names = _collect_var_names(constraint.body)
    if names is None:  # opaque body -> not provably linear
        return False
    return all(_isolated_affine_coeff(constraint.body, nm) is not None for nm in names)


def _defining_is_nonlinear(inc: IncidenceModel, cid: int, var: str) -> bool:
    """Whether ``var``'s defining expression from constraint ``cid`` is nonlinear.

    True when some *other* variable of the defining constraint appears
    nonlinearly there, so substituting ``var`` introduces a nonlinear term.
    """
    return len(inc.constraint_vars(cid) - inc.constraint_linear_vars(cid)) > 0


def _var_in_other_linear_constraint(inc: IncidenceModel, var: str, exclude_cid: int) -> bool:
    """Whether ``var`` appears in a currently-linear constraint other than ``exclude_cid``."""
    for i, c in enumerate(inc.model._constraints):
        if i == exclude_cid or not isinstance(c, Constraint):
            continue
        names = _collect_var_names(c.body)
        if names and var in names and _constraint_is_linear(c):
            return True
    return False


def drop_linearity_breaking(agg: AggregationSet) -> AggregationSet:
    """Remove eliminations that would turn a linear target constraint nonlinear.

    An elimination ``(v, cid)`` is dropped when ``v``'s defining expression is
    nonlinear *and* ``v`` also appears in some other constraint that is currently
    linear — substituting the nonlinear expression there would make that
    constraint nonlinear (the degradation Amarger et al. warn about).  Dropping a
    subset of a lower-triangular set leaves it lower triangular, so the result is
    re-finalized and remains valid.
    """
    inc = agg.incidence
    kept: dict[int, str] = {}
    for e in agg.eliminations:
        if _defining_is_nonlinear(inc, e.cid, e.var) and _var_in_other_linear_constraint(
            inc, e.var, e.cid
        ):
            continue  # would break a linear target
        kept[e.cid] = e.var
    if len(kept) == agg.n:
        return agg  # nothing dropped
    return _finalize(kept, inc, agg.method)


def _induced_acyclic(subset: set[str], deps: dict[str, set[str]]) -> bool:
    """Whether the dependency subgraph induced by ``subset`` is acyclic.

    ``deps[v]`` are the variables ``v`` depends on; an edge ``d -> v`` exists for
    each ``d`` in ``deps[v]``.  Uses :func:`topological_order` (raises on a cycle).
    """
    succ: dict[str, list[str]] = {v: [] for v in subset}
    for v in subset:
        for d in deps[v]:
            if d in subset:
                succ[d].append(v)
    try:
        topological_order(subset, lambda n: succ[n])
        return True
    except ValueError:
        return False


def _max_acyclic_subset(nodes: list[str], deps: dict[str, set[str]]) -> set[str]:
    """Largest subset of ``nodes`` whose induced dependency subgraph is acyclic.

    Exact (brute force over subsets, largest first).  Equivalently the complement
    of a minimum feedback vertex set — the maximum set of variables that can be
    eliminated from a cyclic block while remaining strictly lower triangular.
    """
    for size in range(len(nodes), 0, -1):
        for subset in itertools.combinations(nodes, size):
            s = set(subset)
            if _induced_acyclic(s, deps):
                return s
    return set()


def _exact_tear(inc: IncidenceModel, block: list[tuple[int, str]]) -> dict[int, str]:
    """Maximum lower-triangular sub-matching of a cyclic block (exact tearing).

    Blocks up to :data:`_EXACT_TEAR_CAP` variables use the brute-force maximum
    acyclic subset; larger blocks use the lazy-cycle MILP
    (:func:`~discopt.aggregation.tearing.max_acyclic_subset_milp`, S2), which is
    also exact but scales.  Any MILP failure falls back to the greedy tear, so the
    result is never worse than greedy.
    """
    block_vars = [var for _cid, var in block]
    cid_of = {var: cid for cid, var in block}
    var_set = set(block_vars)
    deps = {var: (inc.constraint_vars(cid_of[var]) & var_set) - {var} for var in block_vars}
    if len(block_vars) <= _EXACT_TEAR_CAP:
        keep = _max_acyclic_subset(block_vars, deps)
    else:
        try:
            from discopt.aggregation.tearing import max_acyclic_subset_milp

            keep = max_acyclic_subset_milp(block_vars, deps)
        except Exception:  # non-optimal MILP or solver error: never degrade
            return _greedy(inc, [cid for cid, _ in block], restrict=var_set)
    return {cid_of[v]: v for v in keep}


def _n_block(matching: dict[int, str], inc: IncidenceModel) -> int:
    return len(block_triangularize(matching, inc.graph.adj_left))


def _decomposable_matching(
    inc: IncidenceModel, matching: dict[int, str], max_iters: int = 200
) -> dict[int, str]:
    """Local-search a maximum matching for a more decomposable block structure.

    Starting from a maximum matching, repeatedly apply cardinality-preserving
    moves — rematch a constraint to a currently-free linear variable, or 2-swap
    two constraints' matched variables — accepting a move only when it strictly
    increases the number of blocks (SCCs) of the induced projection.  ``n_block``
    is Theorem 2's lower bound on the number of eliminations, so a more
    decomposable matching can only help.  Deterministic (sorted iteration),
    monotone in ``n_block``, and matching cardinality is invariant.
    """
    best = dict(matching)
    best_nb = _n_block(best, inc)
    for _ in range(max_iters):
        matched_by = {v: c for c, v in best.items()}
        improved = False
        for c in sorted(best):
            v = best[c]
            for vp in sorted(inc.constraint_linear_vars(c)):
                if vp == v:
                    continue
                cp = matched_by.get(vp)
                cand = dict(best)
                if cp is None:  # rematch c to a free variable
                    cand[c] = vp
                elif cp != c and v in inc.constraint_linear_vars(cp):  # 2-swap
                    cand[c] = vp
                    cand[cp] = v
                else:
                    continue
                if _n_block(cand, inc) > best_nb:
                    best, best_nb = cand, _n_block(cand, inc)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def linear_matching_aggregation_set(
    inc: IncidenceModel, cids: list[int] | None = None, method: str = "lm"
) -> AggregationSet:
    """Linear-matching aggregation (paper Algorithm 2).

    Compute a maximum matching on the linear incidence graph, block-
    triangularize it over the full incidence, keep singleton blocks directly and
    tear each larger (cyclic) block.  The tear is greedy by default; setting
    ``inc.tearing == "exact"`` uses the exact maximum-acyclic-subset tear, which
    eliminates at least as many variables per block.  When ``inc.decomposable``
    is set, the maximum matching is first local-searched for a more decomposable
    (higher ``n_block``) structure.  Optionally restrict to a subset of
    constraint ids (used by the filtered strategies).
    """
    if cids is None:
        cids = list(inc.constraints)
    cids = sorted(cids)

    linear_edges = [(cid, v) for cid in cids for v in inc.constraint_linear_vars(cid)]
    linear_graph = BipartiteGraph.from_edges(linear_edges)
    m_linear = maximum_matching(linear_graph)
    if getattr(inc, "decomposable", False):
        m_linear = _decomposable_matching(inc, m_linear)

    blocks = block_triangularize(m_linear, inc.graph.adj_left)
    m_reduced: dict[int, str] = {}
    for block in blocks:
        if len(block) == 1:
            cid, var = block[0]
            m_reduced[cid] = var
        elif getattr(inc, "tearing", "greedy") == "exact":
            m_reduced.update(_exact_tear(inc, block))
        else:
            # Algorithm 2 step 6b: greedy on the block's induced subgraph, so it
            # cannot match a variable belonging to another block.
            block_vars = {v for _cid, v in block}
            m_reduced.update(_greedy(inc, [cid for cid, _ in block], restrict=block_vars))

    result = _finalize(m_reduced, inc, method)
    result.stats = {
        "n_match": len(m_linear),
        "n_block": len(blocks),
        "n_agg": result.n,
    }
    # Theorem 2 chain n_block <= n_agg <= n_match holds for the algorithmic
    # matching (before any max_fill / guard drops in _finalize reduce n_agg).
    assert result.stats["n_block"] <= len(m_reduced) <= result.stats["n_match"], (
        f"Theorem-2 bound violated: {result.stats}"
    )
    return result


def greedy_aggregation_set(inc: IncidenceModel) -> AggregationSet:
    """Greedy aggregation over the full linear system (paper Algorithm 1)."""
    matching = _greedy(inc, sorted(inc.constraints))
    return _finalize(matching, inc, "gr")


# --------------------------------------------------------------------------- #
# constraint filters (paper Table 2) + filtered strategies                     #
# --------------------------------------------------------------------------- #
def _filter_fixed_variable(inc: IncidenceModel) -> list[int]:
    """``y = a``: exactly one variable *total*, appearing linearly.

    The total count includes non-candidate variables (integers, arrays), so a
    constraint like ``y = a + i`` (integer ``i``) is not treated as fixed-variable.
    """
    return [
        cid
        for cid in inc.constraints
        if inc.constraint_total_degree(cid) == 1 and len(inc.constraint_linear_vars(cid)) == 1
    ]


def _filter_degree_k(inc: IncidenceModel, k: int) -> list[int]:
    """At most ``k`` variables *total*, at least one of them linear (``y = f(x, ...)``).

    Counts every variable in the constraint (candidates and non-candidates), so a
    non-candidate passenger (an integer or array variable) is not silently ignored
    — matching the paper's degree semantics and the structure-preserving claim
    that no remaining constraint gains variables.
    """
    return [
        cid
        for cid in inc.constraints
        if inc.constraint_total_degree(cid) <= k and len(inc.constraint_linear_vars(cid)) >= 1
    ]


def _filter_degree2(inc: IncidenceModel) -> list[int]:
    """At most two variables, at least one of them linear (``y = f(x)``)."""
    return _filter_degree_k(inc, 2)


def _filter_linear_degree2(inc: IncidenceModel) -> list[int]:
    """Exactly two variables *total*, both linear (``y = a*x + b``).

    Requiring the total variable count to be 2 (not merely two *linear*
    variables) excludes constraints that also carry nonlinear terms in other
    variables, so the defining constraint is genuinely linear.
    """
    return [
        cid
        for cid in inc.constraints
        if inc.constraint_total_degree(cid) == 2 and len(inc.constraint_linear_vars(cid)) == 2
    ]


def _filter_equal_coefficient(inc: IncidenceModel) -> list[int]:
    """Two linear variables (and no others) with equal-magnitude coefficients.

    Matches ``y = x + a`` (coefficients of equal magnitude), the most
    restrictive structure-preserving form.
    """
    out: list[int] = []
    for cid in inc.constraints:
        lin = inc.constraint_linear_vars(cid)
        if len(lin) != 2 or inc.constraint_total_degree(cid) != 2:
            continue
        c1, c2 = (inc.linear_coeff[(cid, v)] for v in lin)
        if abs(abs(c1) - abs(c2)) <= 1e-12 * max(1.0, abs(c1), abs(c2)):
            out.append(cid)
    return out


def fixed_variable_aggregation_set(inc: IncidenceModel) -> AggregationSet:
    """LD1 — linear degree-1 / fixed-variable aggregation (paper Algorithm 3)."""
    return linear_matching_aggregation_set(inc, _filter_fixed_variable(inc), "ld1")


def degree_two_aggregation_set(inc: IncidenceModel) -> AggregationSet:
    """D2 — degree-2 aggregation (paper Algorithm 4)."""
    return linear_matching_aggregation_set(inc, _filter_degree2(inc), "d2")


def degree_k_aggregation_set(inc: IncidenceModel, k: int) -> AggregationSet:
    """Dk — degree-k aggregation: eliminate using constraints with <= k variables.

    Generalises D2 (``k=2``) to a caller-chosen bound on the number of variables
    in a defining constraint, the intermediate between D2 and the unbounded
    greedy/linear-matching strategies suggested by the paper.  Larger ``k``
    reaches more constraints but permits denser substitutions.
    """
    if k < 2:
        raise ValueError(f"degree-k aggregation requires k >= 2, got {k}")
    return linear_matching_aggregation_set(inc, _filter_degree_k(inc, k), f"d{k}")


def linear_degree_two_aggregation_set(inc: IncidenceModel) -> AggregationSet:
    """LD2 — linear degree-2 aggregation (paper Algorithm 5)."""
    return linear_matching_aggregation_set(inc, _filter_linear_degree2(inc), "ld2")


def equal_coefficient_aggregation_set(inc: IncidenceModel) -> AggregationSet:
    """ECD2 — equal-coefficient degree-2 aggregation (paper Algorithm 6)."""
    return linear_matching_aggregation_set(inc, _filter_equal_coefficient(inc), "ecd2")


def linear_matching_full(inc: IncidenceModel) -> AggregationSet:
    """LM — linear-matching aggregation over the full system (paper Algorithm 2)."""
    return linear_matching_aggregation_set(inc, None, "lm")


STRATEGIES: dict[str, Callable[[IncidenceModel], AggregationSet]] = {
    "ld1": fixed_variable_aggregation_set,
    "ecd2": equal_coefficient_aggregation_set,
    "ld2": linear_degree_two_aggregation_set,
    "d2": degree_two_aggregation_set,
    "gr": greedy_aggregation_set,
    "lm": linear_matching_full,
}

# Degree-k methods: ``d3``, ``d4``, ... generalise ``d2`` for any k >= 2.
_DEGREE_K_RE = re.compile(r"^d(\d+)$")


def resolve_strategy(method: str) -> Callable[[IncidenceModel], AggregationSet]:
    """Return the strategy callable for a method name.

    Accepts the named strategies in :data:`STRATEGIES` plus the degree-k family
    ``d<k>`` for any integer ``k >= 2`` (``d3``, ``d4``, ...).  Raises
    :class:`ValueError` for anything else.
    """
    key = method.lower()
    if key in STRATEGIES:
        return STRATEGIES[key]
    m = _DEGREE_K_RE.match(key)
    if m:
        k = int(m.group(1))
        if k >= 2:
            return lambda inc: degree_k_aggregation_set(inc, k)
    raise ValueError(
        f"unknown aggregation method {method!r}; choose from {sorted(STRATEGIES)} or 'd<k>' (k>=2)"
    )


def aggregation_set(
    model_or_incidence: Any,
    method: str = "d2",
    pivot_tol: float = 1e-6,
    max_fill: float | None = None,
) -> AggregationSet:
    """Select an aggregation set for a model using the named ``method``.

    Args:
        model_or_incidence: a :class:`discopt.modeling.core.Model` or an
            already-built :class:`~discopt.aggregation.incidence.IncidenceModel`.
        method: one of :data:`STRATEGIES` (``ld1``, ``ecd2``, ``ld2``, ``d2``,
            ``gr``, ``lm``) or a degree-k method ``d<k>`` (``d3``, ``d4``, ...).
            Defaults to ``d2`` (the paper's recommendation).
        pivot_tol: relative pivot tolerance for the linear graph (used only when
            building incidence from a model; ignored when an
            :class:`IncidenceModel` is passed, which already carries its own).
        max_fill: optional per-elimination fill cap (see
            :meth:`IncidenceModel.fill_estimate`); ignored when an incidence is
            passed.
    """
    strategy = resolve_strategy(method)
    inc = (
        model_or_incidence
        if isinstance(model_or_incidence, IncidenceModel)
        else build_incidence(model_or_incidence, pivot_tol=pivot_tol, max_fill=max_fill)
    )
    return strategy(inc)


__all__ = [
    "Elimination",
    "AggregationSet",
    "aggregation_set",
    "greedy_aggregation_set",
    "linear_matching_full",
    "linear_matching_aggregation_set",
    "fixed_variable_aggregation_set",
    "degree_two_aggregation_set",
    "degree_k_aggregation_set",
    "linear_degree_two_aggregation_set",
    "equal_coefficient_aggregation_set",
    "drop_linearity_breaking",
    "resolve_strategy",
    "STRATEGIES",
]
