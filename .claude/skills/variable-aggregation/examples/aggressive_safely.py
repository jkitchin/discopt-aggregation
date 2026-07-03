"""Aggressive (gr/lm) aggregation done safely: fill-aware order + size cap.

Shows the substitution blow-up and how order='min_fill' + max_def_nodes tame it,
without changing the eliminations or the optimum. Run: python aggressive_safely.py
"""

from discopt.aggregation import aggregate, model_expression_size
from discopt.aggregation.benchmark import _local_solve
from discopt.aggregation.testproblems import reaction_diffusion


def report(tag, agg):
    tree, dag = model_expression_size(agg.reduced_model)
    status, iters, wall, obj = _local_solve(agg.reduced_model)  # LOCAL solve (paper setting)
    print(
        f"{tag:24s} elim={agg.n_eliminated:2d} objExprTree={tree:6d} dag={dag:5d}"
        f"  solve={status.split('.')[-1]} iters={iters} {wall:.2f}s"
    )


m = reaction_diffusion(n=8)

# Structure-preserving: small and fast, modest reduction.
report("d2 (structure-preserving)", aggregate(m, method="d2"))

# Aggressive, naive: same eliminations but a huge substituted expression.
report("gr (index order)", aggregate(m, method="gr"))

# Aggressive, safe: fill-aware order collapses the expression ~40x.
report("gr (min_fill order)", aggregate(m, method="gr", order="min_fill"))

# Belt-and-suspenders: also cap the expression size.
report("gr (min_fill + cap)", aggregate(m, method="gr", order="min_fill", max_def_nodes=300))

print(
    "\nTakeaway: gr/lm eliminate the most, but naive substitution blows up the\n"
    "expression graph (Hessian cost). order='min_fill' (and max_def_nodes) keep\n"
    "the same eliminations while shrinking the reduced model ~40x."
)
