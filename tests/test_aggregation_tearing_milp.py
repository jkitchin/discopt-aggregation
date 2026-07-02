"""Tests for S2: scalable exact tearing via lazy-cycle MILP.

Validates the MILP maximum-acyclic-subset against brute force on small cyclic
graphs, that its result is genuinely acyclic, that it scales past the ~13-var
brute-force cap, and that ``tearing="exact"`` still round-trips end-to-end.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import pytest

from discopt.aggregation.api import aggregate
from discopt.aggregation.tearing import find_cycle, max_acyclic_subset_milp

pytestmark = pytest.mark.unit


def _induced_succ(subset, deps):
    succ = {v: [] for v in subset}
    for v in subset:
        for d in deps[v]:
            if d in subset:
                succ[d].append(v)
    return succ


def _brute_max_acyclic(nodes, deps):
    for size in range(len(nodes), 0, -1):
        for sub in itertools.combinations(nodes, size):
            s = set(sub)
            if find_cycle(list(s), _induced_succ(s, deps)) is None:
                return s
    return set()


def _ring(n):
    nodes = [f"v{i}" for i in range(n)]
    deps = {f"v{i}": {f"v{(i - 1) % n}"} for i in range(n)}
    return nodes, deps


def _double_cycle(n):
    nodes = [f"v{i}" for i in range(n)]
    deps = {f"v{i}": {f"v{(i - 1) % n}", f"v{(i - 2) % n}"} for i in range(n)}
    return nodes, deps


@pytest.mark.parametrize("build", [lambda: _ring(6), lambda: _double_cycle(6), lambda: _ring(9)])
def test_milp_matches_brute_force(build):
    nodes, deps = build()
    brute = _brute_max_acyclic(nodes, deps)
    milp = max_acyclic_subset_milp(nodes, deps)
    assert len(milp) == len(brute)  # both optimal; cardinality is unique


def test_milp_result_is_acyclic():
    nodes, deps = _double_cycle(8)
    keep = max_acyclic_subset_milp(nodes, deps)
    assert find_cycle(list(keep), _induced_succ(keep, deps)) is None


def test_milp_scales_past_brute_cap():
    # 20 > _EXACT_TEAR_CAP (13); brute force would be 2^20.
    nodes, deps = _double_cycle(20)
    keep = max_acyclic_subset_milp(nodes, deps)
    assert find_cycle(list(keep), _induced_succ(keep, deps)) is None
    assert len(keep) >= 18  # remove only a couple to break the interleaved cycles


def test_find_cycle_detects_and_clears():
    assert find_cycle(["a", "b"], {"a": ["b"], "b": ["a"]}) is not None
    assert find_cycle(["a", "b", "c"], {"a": ["b"], "b": ["c"], "c": []}) is None


def test_exact_tearing_round_trips_end_to_end():
    # mutually-dependent defining constraints (a cyclic block); tearing="exact"
    # must still produce a correct reduced model with >= greedy eliminations.
    def build():
        m = dm.Model("cyc")
        v = [m.continuous(f"x{i}", lb=-9, ub=9) for i in range(4)]
        m.subject_to(v[0] - v[1] == 0.0)
        m.subject_to(v[1] - (v[0] + v[2]) == 0.0)
        m.subject_to(v[2] - v[3] == 0.0)
        m.subject_to(v[3] - v[2] == 0.0)
        m.minimize(sum((x - 1.0) ** 2 for x in v))
        return m

    greedy = aggregate(build(), method="lm", tearing="greedy")
    exact = aggregate(build(), method="lm", tearing="exact")
    assert exact.n_eliminated >= greedy.n_eliminated
