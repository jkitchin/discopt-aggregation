"""C6: aggregation results are reproducible across PYTHONHASHSEED.

The graph kernel iterates set-valued adjacency; without a fixed order the
maximum matching (and everything downstream — eliminations, Schur indices, block
orderings) depends on the process hash seed.  These tests run the pipeline in
fresh subprocesses under different seeds and assert byte-identical output.
"""

from __future__ import annotations

import subprocess
import sys

_SNIPPET = r"""
import discopt.modeling as dm
from discopt.aggregation.api import aggregate
from discopt.aggregation.schur import kkt_schur_indices
from discopt.aggregation.ordering import block_triangular_ordering

def model():
    m = dm.Model("d")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-20, ub=20)
    z = m.continuous("z", lb=-5, ub=5)
    w = m.continuous("w", lb=-5, ub=5)
    m.subject_to(y - (2 * x + 1) == 0.0)
    m.subject_to(w - (x + z) == 0.0)
    m.subject_to(x + z >= 1.0)
    m.minimize(y**2 + w**2 + x + z)
    return m

for method in ("lm", "gr", "d2"):
    agg = aggregate(model(), method=method)
    print(method, sorted(agg.eliminated))
print("schur", kkt_schur_indices(model()))
print("order", block_triangular_ordering(model()).permutation)
"""


def _run(seed: str) -> str:
    import os

    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return out.stdout


def test_pipeline_output_stable_across_hash_seeds():
    outputs = {seed: _run(seed) for seed in ("0", "5", "7")}
    baseline = outputs["0"]
    assert baseline.strip()  # non-empty
    for seed, out in outputs.items():
        assert out == baseline, f"nondeterministic output at PYTHONHASHSEED={seed}"
