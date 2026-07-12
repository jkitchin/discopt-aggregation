# discopt-aggregation — developer & release guide

Variable-aggregation (reduced-space presolve) plugin for the `discopt` modeling
language. This distribution packages **only** the `discopt.aggregation` PEP 420
namespace subpackage (never a top-level `discopt`), so it merges into the base
`discopt` namespace at import time. Requires `discopt >= 0.6`.

## Development

Everything runs through **uv** (the lockfile `uv.lock` is committed). The build
backend is **setuptools**; there is no Makefile — the commands below are the
whole toolchain, and CI (`ci.yml`) runs exactly the gate ones.

| Task | Command |
| --- | --- |
| Create env + install (with dev extras) | `uv venv && uv pip install -e ".[dev]"` |
| Run tests (fast set) | `uv run pytest -q` |
| Run tests incl. slow | `uv run pytest tests/ -m "slow or not slow"` |
| Lint | `uv run ruff check src/ tests/` |
| Format check | `uv run ruff format --check src/ tests/` |
| Type check | `uv run mypy src/discopt/aggregation` |
| Build the docs | `uv pip install -e ".[docs]" && uv run mkdocs build --strict` |
| Build artifacts | `rm -rf dist && uv build` |
| Check artifacts | `uvx twine check dist/*` |

Notes:
- The default pytest `addopts` is `-m 'not slow'`, so **slow tests are skipped
  unless you ask for them**. Markers are `unit` (pure logic, no solver), `smoke`
  (one solve per path), `slow`. A release must run the slow set.
- `discopt` must be importable — its `__init__.py` extends `__path__` via
  `pkgutil.extend_path`, which is what lets this namespace plugin merge in. The
  `smoke`/solve tests additionally need **`pounce`** (the local Rust NLP solver),
  which is not a declared dependency; without it those tests are skipped/fail, so
  install it locally before a release run.
- Lint/format is **ruff** only (line-length 100, target py310, import sorting via
  `extend-select = ["I"]`). Types are **mypy** (`python_version = "3.12"`).
- Docs are **mkdocs-material** (`mkdocs build --strict`, warnings-as-errors),
  deployed to GitHub Pages by `docs.yml` on pushes that touch `docs/**`,
  `mkdocs.yml`, or `CHANGELOG.md`.
- See `CONTRIBUTING.md` for the package-specific conventions (keep
  discopt/pounce/feral untouched, new behavior opt-in and default-off, every
  feature ships a worked example, tests hash-seed robust, soundness first).

## Release checklist

Work top to bottom. The version lives in **three** places and must match:

1. `version` in `pyproject.toml` (line 7).
2. `__version__` in `src/discopt/aggregation/__init__.py` (line ~36).
3. `version` in `.claude-plugin/plugin.json`.

`tests/test_packaging.py` enforces that (1) == (2) and that the version has a
`[x.y.z]` section in `CHANGELOG.md`, but **nothing enforces (3)** — sync
`plugin.json` by hand or a stale plugin ships. That same test also pins
`len(__all__) == 83`, so a new/removed public export is a loud failure until the
count and docs are updated.

### 0. Prerequisite gate — dependencies must be PyPI-resolvable ⛔

PyPI rejects any distribution that depends on a git/URL source, so every runtime
dependency must resolve from a package index.

- [ ] `discopt >= 0.6` (and numpy/scipy) are published on PyPI at the required
      versions. `grep -rn "git+" pyproject.toml uv.lock` returns nothing for
      runtime dependencies.
- [ ] No `[tool.uv.sources]` git/URL overrides for runtime deps remain in
      `pyproject.toml`.
- [ ] `discopt >= 0.6` is importable in the release environment; `pounce` is
      installed so `smoke`/solve tests actually run (they silently skip or fail
      otherwise — see Development notes).

### 1. Pre-flight — code health

- [ ] On `main`, clean working tree, synced with remote (`git status`, `git pull`).
- [ ] Latest CI run on `main` is green (`gh run list --branch main --limit 1`).
- [ ] Full test suite passes locally **including slow tests**:
      `uv run pytest tests/ -m "slow or not slow"`.
- [ ] Hash-seed robustness holds (`maximum_matching` returns *a* matching, not a
      unique one): `PYTHONHASHSEED=1 uv run pytest -q`.
- [ ] Lint + format clean: `uv run ruff check src/ tests/` and
      `uv run ruff format --check src/ tests/`.
- [ ] Types clean: `uv run mypy src/discopt/aggregation`.
- [ ] Examples run end to end: `python -m discopt.aggregation.examples` exits 0.
- [ ] Docs build clean: `uv run mkdocs build --strict`.

### 2. Version & metadata

- [ ] Bump the version in all **three** locations above per [semver](https://semver.org).
- [ ] Add a `## [x.y.z] - YYYY-MM-DD` section to `CHANGELOG.md` describing the
      changes (this project adds a dated section per release; there is no
      `[Unreleased]` holding area). The packaging test requires the current
      version to appear here.
- [ ] Re-check `README.md` (install steps, the `discopt >= 0.6` requirement, and
      `discopt-aggregation-skill install`).
- [ ] Re-check `pyproject.toml` metadata: `description`, `license`, `authors`,
      `requires-python`, `classifiers`, `[project.urls]`.

### 3. Build & verify artifacts

- [ ] `rm -rf dist && uv build` — wipe first; a stale gitignored `dist/` may
      exist. Produces an sdist + a wheel.
- [ ] `uvx twine check dist/*` passes.
- [ ] Inspect the wheel: `python -m zipfile -l dist/discopt_aggregation-*.whl` —
      it must contain `discopt/aggregation/**` **only** (no top-level
      `discopt/__init__.py`, which would shadow the base package) and must include
      the skill assets (`discopt/aggregation/skill/**`). The slow test
      `tests/test_aggregation_wheel.py` builds a real wheel and checks exactly
      this — make sure it ran in step 1.
- [ ] Smoke test in a clean venv:
      `python -m venv /tmp/da && /tmp/da/bin/pip install dist/discopt_aggregation-*.whl`
      (with `discopt` available), then
      `/tmp/da/bin/python -c "import discopt.aggregation; print(discopt.aggregation.__version__)"`
      and run `discopt-aggregation-skill install` (writes to `./.claude/skills/`;
      `--user` for `~/.claude/skills/`).
- [ ] (Optional but recommended) TestPyPI dry run: Actions → **publish** →
      *Run workflow* with `target = testpypi`, then
      `pip install -i https://test.pypi.org/simple/ discopt-aggregation` in a
      clean env.

### 4. Tag & release

- [ ] Commit the version bumps + CHANGELOG, push to `main` (or via PR).
- [ ] Annotated tag: `git tag -a vX.Y.Z -m "vX.Y.Z"` then `git push origin vX.Y.Z`.
- [ ] Create the GitHub release — this **triggers `publish.yml`**, which builds,
      re-checks the wheel layout, and uploads to PyPI via OIDC:
      `gh release create vX.Y.Z --title vX.Y.Z --notes-file <notes.md>`.
      (Always use `--notes-file`; never inline `--notes` with prose.)

### 5. Post-release verification

- [ ] The **publish** workflow succeeded (Actions tab) and PyPI shows the new
      version.
- [ ] `pip install "discopt-aggregation==X.Y.Z"` works in a fresh venv (with
      `discopt >= 0.6` available).
- [ ] Pushing to `main` triggered `docs.yml`; GitHub Pages shows the new
      CHANGELOG / docs.
- [ ] (Optional) open a follow-up to start the next dev cycle.

## PyPI trusted-publisher setup (one-time)

`publish.yml` uses **OIDC trusted publishing** — no API tokens are stored. A
GitHub release publishes to PyPI; the `workflow_dispatch → target=testpypi` path
is a dry run. The identity the workflow presents must match the trusted-publisher
config on each index exactly:

- **Owner:** `jkitchin`
- **Repository name:** `discopt-aggregation`
- **Workflow name:** `publish.yml` (filename only, not a path)
- **Environment name:** `pypi` (real index) / `testpypi` (dry run)

### On PyPI / TestPyPI

At https://pypi.org/manage/account/publishing/ (and, for the dry run, the
separate https://test.pypi.org/manage/account/publishing/ — TestPyPI is its own
site with its own account), the trusted publisher for **PyPI Project Name**
`discopt-aggregation` must use the four values above, with environment `pypi` on
PyPI and `testpypi` on TestPyPI. Since you've already set this up on PyPI, just
confirm the environment name registered there is `pypi` (that's what the workflow
declares); add the TestPyPI publisher too if you want the dry-run path.

### On GitHub (the environment side)

`publish.yml` references `environment: pypi` / `testpypi`, so both must exist
(**Settings → Environments → New environment**). For `pypi`, adding yourself
under **Required reviewers** makes a real publish pause for a one-click approval
before uploading. No secrets to add — OIDC handles auth.
