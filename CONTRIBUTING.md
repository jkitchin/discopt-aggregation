# Contributing to `discopt-aggregation`

Thanks for contributing. This is a **standalone plugin** that installs into the
`discopt.aggregation` namespace (PEP 420) on top of `discopt`. A few conventions
keep it that way.

## Development setup

Uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv venv
uv pip install -e ".[dev]"     # discopt, numpy, scipy + pytest, ruff, mypy
```

`discopt` must be importable (its `__init__.py` extends `__path__` via
`pkgutil.extend_path`, which is what lets this namespace plugin merge in). The
solve-based tests additionally need `pounce` (the local Rust NLP solver).

## The gate (run before every PR)

CI runs exactly these; a change lands only when all pass:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/discopt/aggregation
uv run pytest -q
```

- **ruff** — lint + import sorting (`extend-select = ["I"]`) and formatting,
  line length 100.
- **mypy** — `src/discopt/aggregation` type-clean.
- **pytest** — markers: `unit` (pure logic, no solver), `smoke` (one solve per
  path), `slow` (skipped by default). Keep `unit` tests solver-free and fast.

## Conventions specific to this package

1. **Keep `discopt` / `pounce` / `feral` untouched.** The only change to the base
   package is the one-line namespace shim in discopt's `__init__.py`. Anything
   that needs a core change is **filed upstream** (an issue), not vendored or
   patched — and the plugin must not reach into another package's *private*
   internals to work around a missing API (see `schur.py`'s integration-boundary
   note for an example). Import public APIs; file for the rest.

2. **New behavior is opt-in and default-off.** Every guard/option/knob defaults to
   the prior behavior. `d2` stays the default method; `tearing="greedy"` the
   default tear. A change must not alter existing results unless that *is* the
   change (with tests proving the old path unchanged).

3. **Every feature ships a worked example.** Add an `example_*` in
   `src/discopt/aggregation/examples.py`, wire it into `run_all()`, and cover it
   with a `test_*_example_runs` in `tests/test_aggregation_examples.py`. Examples
   are executable documentation (`python -m discopt.aggregation.examples`).

4. **Tests must be hash-seed robust.** `maximum_matching` returns *a* maximum
   matching, not a unique one — string-hash randomization changes which. Do not
   assert a specific matching/ordering/variable identity; assert invariants
   (valid permutation, block counts, inherent cycles). Check with
   `PYTHONHASHSEED=1 uv run pytest ...`.

5. **Soundness first.** Eliminations must preserve the strict lower-triangular
   (Lemma 1) structure; dropping any subset stays sound. When a transform can't
   guarantee a correct result (unsupported node, singular block), it declines or
   raises — never returns a wrong answer.

6. **Cite as you go.** Related work lives in `design/references.bib` (verified
   entries only). The design docs under `design/` cite it via org-cite
   (`[cite:@key]`). Verify DOIs/authors before adding an entry.

## Where things go

```
src/discopt/aggregation/    # the plugin (namespace subpackage)
  matching · incidence · strategies · transform · recovery · api     # core
  scalarize · metrics · diagnostics · ordering · schur               # structure
  guards · hessian_cost · duals · derivatives · implicit · tearing   # numerics
  testproblems · benchmark · sweep · profiling · presolve · examples # study
tests/                      # test_aggregation_*.py
design/                     # roadmap · parity plan · references.bib
```

## Versioning & changelog

Semantic versioning. Bump `version` in `pyproject.toml` **and**
`discopt.aggregation.__version__` together (a test enforces consistency), and add
a `[x.y.z]` section to `CHANGELOG.md`.

## Filing upstream

Features that need a core change (a modeling node, a solver hook, a compiler
optimization) are filed as issues on the relevant repo (`discopt` / `pounce` /
`feral`) with a minimal reproduction and, where possible, the plugin-side input
ready to feed it. Reference the issue number from the plan doc and the code.
