"""Quickstart: aggregate -> solve -> recover, and inspect what happened.

Run: python quickstart.py
"""

import discopt.modeling as dm

from discopt.aggregation import aggregate, compare_methods, solve


def build():
    # y and w are DEFINED by equalities -> aggregation targets.
    m = dm.Model("quickstart")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-20, ub=20)
    w = m.continuous("w", lb=-50, ub=50)
    m.subject_to(y - (2 * x + 1) == 0.0)  # y = 2x + 1        (linear degree-2)
    m.subject_to(w - (y**2) == 0.0)  # w = y^2           (degree-2)
    m.minimize(w + x**2)
    return m


# 1. One-call solve with the recommended default method.
res = solve(build(), method="d2")
print("status   :", res.status)
print("objective:", round(float(res.objective), 6))
print("full x   :", {k: round(float(v), 4) for k, v in res.x.items()})

# 2. Inspect the reduction without solving.
agg = aggregate(build(), method="d2")
print("\neliminated:", agg.eliminated, "in", agg.n_passes, "pass(es)")
print("reduced vars:", [v.name for v in agg.reduced_model._variables])

# 3. Compare methods structurally (paper Table 4 shape).
print("\nmethod  elim  vars  nnz/con")
for row in compare_methods(build(), methods=["ld1", "d2", "gr", "lm"]):
    print(
        f"  {row['method']:6s} {row['n_eliminated']:4d} {row['n_variables']:5d}"
        f"  {row['avg_nnz_per_con']:.2f}"
    )
