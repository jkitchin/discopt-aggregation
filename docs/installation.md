# Installation

`discopt-aggregation` is a **namespace plugin** (PEP 420) that merges into the
`discopt` package at import time. It requires `discopt >= 0.6`.

```bash
pip install discopt-aggregation          # from PyPI
```

For development, install from a checkout instead:

```bash
uv pip install -e ".[dev]"               # or: pip install -e ".[dev]"
```

The plugin packages only `discopt.aggregation` — never `discopt` — so it never
shadows the base package. For the namespace to merge, `discopt/__init__.py` must
extend its search path:

```python
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
```

discopt 0.6+ ships this line. (If you use an older discopt, add it once.)

Solve-based use additionally needs `pounce` (`pounce-solver`), the local Rust NLP
solver discopt drives.

## Verify

```python
import discopt.aggregation as agg
print(agg.__version__)
```

```bash
python -m discopt.aggregation.examples    # runs every feature end-to-end
```

## The Claude skill

The package ships a [Claude Code](https://code.claude.com/docs/en/skills) skill
that documents method choice, every option, recovery, and the study harnesses.
Claude Code does not auto-discover skills from installed Python packages, so after
`pip install` run the bundled installer once:

```bash
discopt-aggregation-skill install            # -> ./.claude/skills/  (this project)
discopt-aggregation-skill install --user     # -> ~/.claude/skills/  (every project)
discopt-aggregation-skill path               # where the packaged skill lives
```

## Development

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[dev]"     # discopt, numpy, scipy + pytest, ruff, mypy
```

The CI gate:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/discopt/aggregation
uv run pytest -q
```

`pytest -m unit` runs the solver-free unit set; `-m smoke` adds one solve per
code path; `slow` tests are opt-in.
