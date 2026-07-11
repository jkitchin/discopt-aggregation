"""Implicit elimination of cyclic blocks (plan item E5).

A cyclic (irreducible) block of defining equations ``g_B(u, v_B) = 0`` cannot be
eliminated by explicit substitution — that is why the strategies drop such blocks
(their variables stay in the reduced model and are solved directly).  When the
block Jacobian ``∂g_B/∂v_B`` is nonsingular, the implicit function theorem gives
``v_B = φ(u)`` locally, with exact derivatives ``dφ/du = -(∂g_B/∂v_B)^{-1}
∂g_B/∂u``.  This module provides that implicit solve and its derivatives as a
primitive (Newton with exact symbolic Jacobians from
:func:`~discopt.aggregation.derivatives.differentiate_expression`).

This module provides both the *primitive* and the *end-to-end elimination*:

* :class:`ImplicitBlock` — the mathematically-correct primitive: solve a
  nonsingular block for its variables given the kept variables, with exact
  implicit-function-theorem derivatives (analysis / recovery).
* :func:`eliminate_implicit` — the end-to-end reduced-model elimination (K1),
  now unblocked by **discopt#379**: it rebuilds the model with each nonsingular
  cyclic block replaced by discopt's ``Model.implicit`` node (a differentiable
  inner solve via ``jax.lax.custom_root``), so the block is transparent to the
  NLP solver, and :meth:`ImplicitElimination.recover` reconstructs the block
  variables post-solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from discopt.aggregation.derivatives import differentiate_expression
from discopt.aggregation.recovery import evaluate_expression

_INF = 9.9e19  # below discopt's +-9.999e19 sentinel; real bounds < 9.9e19 are kept

_SINGULAR_TOL = 1e-12


@dataclass
class ImplicitBlock:
    """A cyclic block ``g_B(u, v_B) = 0`` solvable for ``v_B`` by an implicit solve.

    Attributes:
        variables: the block variables ``v_B`` (names).
        equations: the defining expression bodies (each ``== 0``), functions of
            the block variables and the kept variables ``u``.
    """

    variables: list[str]
    equations: list[Any]

    def _residual(self, env: dict[str, Any]) -> np.ndarray:
        return np.array(
            [float(np.asarray(evaluate_expression(e, env)).reshape(())) for e in self.equations],
            dtype=float,
        )

    def _jacobian(self, env: dict[str, Any], wrt: list[str]) -> np.ndarray | None:
        """Jacobian ``∂equations/∂wrt`` at ``env``; ``None`` if not differentiable."""
        jac = np.zeros((len(self.equations), len(wrt)), dtype=float)
        for i, e in enumerate(self.equations):
            for j, var in enumerate(wrt):
                d = differentiate_expression(e, var)
                if d is None:
                    return None
                jac[i, j] = float(np.asarray(evaluate_expression(d, env)).reshape(()))
        return jac

    def is_nonsingular(self, kept_values: dict[str, Any], v_values: dict[str, Any]) -> bool:
        """Whether ``∂g_B/∂v_B`` is nonsingular at ``(kept_values, v_values)``."""
        env = {**kept_values, **v_values}
        jac = self._jacobian(env, self.variables)
        if jac is None or jac.shape[0] != jac.shape[1]:
            return False
        return abs(float(np.linalg.det(jac))) > _SINGULAR_TOL

    def solve(
        self,
        kept_values: dict[str, Any],
        x0: dict[str, float] | None = None,
        tol: float = 1e-10,
        maxiter: int = 50,
    ) -> dict[str, float] | None:
        """Solve ``g_B(u, v_B) = 0`` for ``v_B`` by Newton's method.

        Returns ``{var: value}`` on convergence, or ``None`` if the Jacobian is
        singular / not differentiable or Newton fails to converge (the caller then
        falls back — never a wrong answer).
        """
        v = {name: float(x0.get(name, 0.0)) if x0 else 0.0 for name in self.variables}
        for _ in range(maxiter):
            env = {**kept_values, **v}
            residual = self._residual(env)
            if np.linalg.norm(residual, ord=np.inf) < tol:
                return dict(v)
            jac = self._jacobian(env, self.variables)
            if jac is None or jac.shape[0] != jac.shape[1]:
                return None
            if abs(float(np.linalg.det(jac))) <= _SINGULAR_TOL:
                return None
            try:
                step = np.linalg.solve(jac, -residual)
            except np.linalg.LinAlgError:
                return None
            for k, name in enumerate(self.variables):
                v[name] += float(step[k])
        env = {**kept_values, **v}
        if np.linalg.norm(self._residual(env), ord=np.inf) < tol * 100:
            return dict(v)
        return None

    def implicit_derivative(
        self, kept_var: str, kept_values: dict[str, Any], v_values: dict[str, Any]
    ) -> dict[str, float] | None:
        """``dv_B/d(kept_var)`` via the implicit function theorem, at the solution.

        ``dφ/du = -(∂g_B/∂v_B)^{-1} ∂g_B/∂u`` — returns ``{block_var: dvar/dkept}``
        or ``None`` if the block Jacobian is singular / not differentiable.
        """
        env = {**kept_values, **v_values}
        jac_v = self._jacobian(env, self.variables)
        jac_u = self._jacobian(env, [kept_var])
        if jac_v is None or jac_u is None or jac_v.shape[0] != jac_v.shape[1]:
            return None
        if abs(float(np.linalg.det(jac_v))) <= _SINGULAR_TOL:
            return None
        try:
            dvdu = np.linalg.solve(jac_v, -jac_u[:, 0])
        except np.linalg.LinAlgError:
            return None
        return {name: float(dvdu[k]) for k, name in enumerate(self.variables)}


_IMPLICIT_NODE_ISSUE = "discopt#379"


def implicit_blocks(model: Any, *, pivot_tol: float = 1e-6) -> list[ImplicitBlock]:
    """Extract the cyclic (irreducible, size > 1) blocks of ``model`` as primitives.

    Builds the linear-matching maximum matching, block-triangularizes it, and
    returns one :class:`ImplicitBlock` per diagonal block of size > 1 — exactly
    the blocks the explicit strategies drop because they cannot be substituted
    directly.  Each block's ``variables`` are the matched block variables and its
    ``equations`` are the defining constraint bodies ``g_B(u, v_B) == 0``.

    This is the *analysis* half of K1: it identifies what could be eliminated
    implicitly.  Whether a given block is actually eliminable depends on its
    Jacobian being nonsingular at the solution point
    (:meth:`ImplicitBlock.is_nonsingular`); the block is returned regardless so
    the caller can test that at a point.  The reduced-model rewrite that would
    substitute the block is :func:`eliminate_implicit` (not yet available).
    """
    from discopt.aggregation.incidence import build_incidence
    from discopt.aggregation.matching import (
        BipartiteGraph,
        block_triangularize,
        maximum_matching,
    )

    inc = build_incidence(model, pivot_tol=pivot_tol)
    cids = sorted(inc.constraints)
    linear_edges = [(cid, v) for cid in cids for v in inc.constraint_linear_vars(cid)]
    matching = maximum_matching(BipartiteGraph.from_edges(linear_edges))
    blocks = block_triangularize(matching, inc.graph.adj_left)

    result: list[ImplicitBlock] = []
    for block in blocks:
        if len(block) <= 1:
            continue
        variables = [var for _cid, var in block]
        equations = [inc.constraints[cid].body for cid, _var in block]
        result.append(ImplicitBlock(variables=variables, equations=equations))
    return result


# --------------------------------------------------------------------------- #
# End-to-end implicit elimination (K1) — now unblocked by discopt#379          #
# --------------------------------------------------------------------------- #
# discopt's ``Model.implicit(residual, u_inputs, n_unknowns, x0=...)`` node
# (issue #379) compiles a variable vector ``v`` defined by ``residual(u, v) = 0``
# to a differentiable inner solve.  That lets us substitute a nonsingular cyclic
# block *into* the reduced model — the piece E5/K1 was missing.


def _jax_eval_expr(expr: Any, env: dict[str, Any], jnp: Any) -> Any:
    """JAX-traceable evaluation of a modeling expression (mirrors evaluate_expression).

    Unlike :func:`~discopt.aggregation.recovery.evaluate_expression`, this never
    calls ``np.asarray`` (which breaks tracing): variable/constant leaves pass
    through and every op dispatches to ``jnp``.  Raises :class:`NotImplementedError`
    on an unsupported node so callers can skip a block rather than build a wrong
    residual.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    jfuncs = {
        "sin": jnp.sin, "cos": jnp.cos, "tan": jnp.tan, "exp": jnp.exp,
        "log": jnp.log, "sqrt": jnp.sqrt, "tanh": jnp.tanh, "abs": jnp.abs,
        "sinh": jnp.sinh, "cosh": jnp.cosh, "arctan": jnp.arctan, "atan": jnp.arctan,
    }  # fmt: skip

    if isinstance(expr, Variable):
        return env[expr.name]
    if isinstance(expr, (Constant, Parameter)):
        return expr.value
    if isinstance(expr, UnaryOp):
        operand = _jax_eval_expr(expr.operand, env, jnp)
        if expr.op in ("-", "neg"):
            return -operand
        if expr.op in ("+", "pos"):
            return operand
        if expr.op in jfuncs:
            return jfuncs[expr.op](operand)
        raise NotImplementedError(f"unary op {expr.op!r}")
    if isinstance(expr, BinaryOp):
        left = _jax_eval_expr(expr.left, env, jnp)
        right = _jax_eval_expr(expr.right, env, jnp)
        op = expr.op
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op in ("**", "^"):
            return left**right
        raise NotImplementedError(f"binary op {op!r}")
    if isinstance(expr, FunctionCall):
        if expr.func_name not in jfuncs:
            raise NotImplementedError(f"function {expr.func_name!r}")
        return jfuncs[expr.func_name](*[_jax_eval_expr(a, env, jnp) for a in expr.args])
    if isinstance(expr, IndexExpression):
        return _jax_eval_expr(expr.base, env, jnp)[expr.index]
    if isinstance(expr, SumOverExpression):
        terms = [_jax_eval_expr(t, env, jnp) for t in expr.terms]
        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return total
    if isinstance(expr, SumExpression):
        return jnp.sum(_jax_eval_expr(expr.operand, env, jnp))
    if isinstance(expr, (int, float)):
        return expr
    raise NotImplementedError(f"cannot JAX-evaluate node {type(expr).__name__}")


def _block_var_names(expr: Any) -> set[str]:
    """Variable-leaf names in ``expr`` (iterative)."""
    from discopt.modeling.core import Variable

    from discopt.aggregation.hessian_cost import _children

    names: set[str] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Variable):
            names.add(node.name)
        stack.extend(_children(node))
    return names


@dataclass
class ImplicitElimination:
    """A model whose nonsingular cyclic blocks are eliminated via implicit nodes.

    Attributes:
        reduced_model: the rebuilt model — each block's variables replaced by a
            ``Model.implicit`` node, its defining equations dropped.  Solve it on
            the local NLP path.
        blocks: the eliminated :class:`ImplicitBlock` primitives, in order.
        block_inputs: per block, the kept-variable input names (aligned with the
            implicit node's ``u_inputs``).
        eliminated: all eliminated (block) variable names.
        x0_value: the scalar initial guess used for each block's inner solve.
        n_bound_inequalities: number of box->inequality constraints emitted on the
            implicit-node components for bounded block variables (C4).
    """

    reduced_model: Any
    blocks: list[ImplicitBlock]
    block_inputs: list[list[str]]
    eliminated: list[str]
    x0_value: float
    n_bound_inequalities: int = 0

    def recover(self, kept_values: dict[str, Any]) -> dict[str, Any]:
        """Full-space point: kept solution + block variables from the inner solves."""
        full = dict(kept_values)
        for block, inputs in zip(self.blocks, self.block_inputs):
            # Read inputs from the accumulated point, not just kept_values, so a
            # block whose input is another block's recovered variable still works.
            u = {name: full[name] for name in inputs}
            sol = block.solve(u, x0=dict.fromkeys(block.variables, self.x0_value))
            if sol is None:  # pragma: no cover - nonsingular blocks were pre-filtered
                raise RuntimeError(f"implicit block {block.variables} failed to solve")
            full.update(sol)
        return full


def eliminate_implicit(
    model: Any, *, pivot_tol: float = 1e-6, x0_value: float = 1.0
) -> ImplicitElimination:
    """Eliminate ``model``'s nonsingular cyclic blocks as implicit-function nodes (K1).

    Each irreducible (size > 1) diagonal block of the linear-matching is replaced
    by a ``Model.implicit`` node (discopt#379): the block's defining equations
    ``g_B(u, v_B) = 0`` become a differentiable inner solve for ``v_B`` given the
    kept inputs ``u``, and every other use of ``v_B`` is substituted by the node's
    components.  Recover the block variables post-solve with
    :meth:`ImplicitElimination.recover`.

    Scope: blocks whose equations use only scalar variables and JAX-supported ops
    (a block with an unsupported construct is left in place, its variables kept).
    Raises :class:`ValueError` if the model has no eliminable cyclic block.
    """
    import discopt.modeling as dm
    import jax.numpy as jnp
    from discopt.modeling.core import ObjectiveSense, VarType

    from discopt.aggregation.incidence import build_incidence
    from discopt.aggregation.matching import (
        BipartiteGraph,
        block_triangularize,
        maximum_matching,
    )
    from discopt.aggregation.transform import _make_constraint, substitute_expression

    inc = build_incidence(model, pivot_tol=pivot_tol)
    cids = sorted(inc.constraints)
    linear_edges = [(cid, v) for cid in cids for v in inc.constraint_linear_vars(cid)]
    matching = maximum_matching(BipartiteGraph.from_edges(linear_edges))
    blocks = block_triangularize(matching, inc.graph.adj_left)

    # Collect eliminable cyclic-block contexts (skip anything unsupported).
    contexts: list[tuple[list[str], list[int], list[Any], list[str]]] = []
    for block in blocks:
        if len(block) <= 1:
            continue
        block_vars = [var for _cid, var in block]
        block_cids = [cid for cid, _var in block]
        equations = [inc.constraints[cid].body for cid in block_cids]
        if any(_contains_index(e) for e in equations):
            continue  # array/index constructs out of scope
        names: set[str] = set()
        for e in equations:
            names |= _block_var_names(e)
        kept_inputs = sorted(names - set(block_vars))
        imp = ImplicitBlock(variables=block_vars, equations=equations)
        if not imp.is_nonsingular(
            dict.fromkeys(kept_inputs, x0_value), dict.fromkeys(block_vars, x0_value)
        ):
            continue  # only eliminate provably-nonsingular blocks
        contexts.append((block_vars, block_cids, equations, kept_inputs))

    # A block whose equations reference a variable eliminated by ANOTHER block is
    # coupled to it.  Chaining block substitutions (topological block order) is
    # out of scope for K1, which eliminates independent blocks; a coupled block's
    # input would be absent from the reduced model and crash the node build.  So
    # leave such a block in place — decline it — matching the documented "a block
    # with an unsupported construct is left in place" contract.  Conservative
    # single pass: drop any block referencing another candidate block's variable.
    all_block_vars = {v for ctx in contexts for v in ctx[0]}
    contexts = [
        ctx
        for ctx in contexts
        if not (set(ctx[3]) & (all_block_vars - set(ctx[0])))  # ctx[3]=inputs, ctx[0]=vars
    ]

    if not contexts:
        raise ValueError("no eliminable nonsingular cyclic block found")

    eliminated = {v for ctx in contexts for v in ctx[0]}
    elim_cids = {cid for ctx in contexts for cid in ctx[1]}

    reduced = dm.Model(f"{getattr(model, 'name', 'model')}_implicit")
    orig_by_name = {v.name: v for v in model._variables}
    newvar_map: dict[str, Any] = {}
    for v in model._variables:
        if v.name in eliminated:
            continue
        if v.var_type == VarType.CONTINUOUS:
            nv = reduced.continuous(v.name, shape=v.shape, lb=v.lb, ub=v.ub)
        elif v.var_type == VarType.INTEGER:
            nv = reduced.integer(v.name, shape=v.shape, lb=v.lb, ub=v.ub)
        elif v.var_type == VarType.BINARY:
            nv = reduced.binary(v.name, shape=v.shape)
            # binary() takes no bounds; a tightened binary must keep them, as an
            # ndarray shaped like the variable (the form the solver expects).
            nv.lb = np.asarray(v.lb, dtype=float).reshape(v.shape)
            nv.ub = np.asarray(v.ub, dtype=float).reshape(v.shape)
        else:  # pragma: no cover - defensive
            raise NotImplementedError(f"unsupported variable type {v.var_type!r}")
        newvar_map[v.name] = nv

    # Build an implicit node per block; map each block variable to a node component.
    replacement: dict[str, Any] = {}
    out_blocks: list[ImplicitBlock] = []
    block_inputs: list[list[str]] = []
    for block_vars, _block_cids, equations, kept_inputs in contexts:
        u_exprs = [newvar_map[name] for name in kept_inputs]

        def residual(
            u_vec: Any,
            v_vec: Any,
            eqs: Any = equations,
            ks: Any = kept_inputs,
            bs: Any = block_vars,
        ) -> Any:
            env = {name: u_vec[j] for j, name in enumerate(ks)}
            env.update({name: v_vec[i] for i, name in enumerate(bs)})
            return jnp.array([_jax_eval_expr(e, env, jnp) for e in eqs])

        node = reduced.implicit(
            residual, u_exprs, n_unknowns=len(block_vars), x0=jnp.full(len(block_vars), x0_value)
        )
        for i, name in enumerate(block_vars):
            replacement[name] = node[i]
        out_blocks.append(ImplicitBlock(variables=block_vars, equations=equations))
        block_inputs.append(kept_inputs)

    combined: dict[str, Any] = {**newvar_map, **replacement}

    # Re-express each eliminated block variable's box as inequalities on its
    # implicit-node component.  Dropping these (as this rewrite previously did)
    # lets the reduced solve leave the block variables outside their bounds
    # (C4).  No interval pruning here: an implicit-function component's range is
    # not cheaply enclosable, so every finite bound is emitted (sound; at worst
    # redundant).
    n_bound_inequalities = 0
    for name in sorted(eliminated):
        v = orig_by_name[name]
        lb = float(np.min(v.lb))
        ub = float(np.max(v.ub))
        comp = replacement[name]
        # IA3 (symmetry with transform.apply_aggregation): an eliminated
        # integer/binary block variable only takes integer values, so tighten its
        # box to the integer grid before emitting the bound inequalities.  In
        # practice integer variables are out of scope for implicit elimination —
        # they never enter a nonsingular cyclic block — so this branch is a
        # defensive equivalence rather than a hot path (a test asserts integers are
        # skipped, not eliminated).
        if v.var_type in (VarType.INTEGER, VarType.BINARY):  # pragma: no cover - defensive
            if lb > -_INF:
                lb = math.ceil(lb - 1e-9)
            if ub < _INF:
                ub = math.floor(ub + 1e-9)
        if lb > -_INF:
            reduced.subject_to(_make_constraint(comp - lb, ">="))
            n_bound_inequalities += 1
        if ub < _INF:
            reduced.subject_to(_make_constraint(comp - ub, "<="))
            n_bound_inequalities += 1

    for cid, c in enumerate(model._constraints):
        if cid in elim_cids:
            continue
        new_body = substitute_expression(c.body, combined)
        reduced.subject_to(_make_constraint(new_body, c.sense))

    obj = model._objective
    if obj is not None:
        new_obj = substitute_expression(obj.expression, combined)
        if obj.sense == ObjectiveSense.MINIMIZE:
            reduced.minimize(new_obj)
        else:
            reduced.maximize(new_obj)

    return ImplicitElimination(
        reduced_model=reduced,
        blocks=out_blocks,
        block_inputs=block_inputs,
        eliminated=sorted(eliminated),
        x0_value=x0_value,
        n_bound_inequalities=n_bound_inequalities,
    )


def _contains_index(expr: Any) -> bool:
    """Whether ``expr`` contains an :class:`IndexExpression` (array element)."""
    from discopt.modeling.core import IndexExpression

    from discopt.aggregation.hessian_cost import _children

    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, IndexExpression):
            return True
        stack.extend(_children(node))
    return False


__all__ = [
    "ImplicitBlock",
    "implicit_blocks",
    "eliminate_implicit",
    "ImplicitElimination",
]
