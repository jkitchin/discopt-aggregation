"""The bundled Claude skill ships in the package and installs discoverably.

Claude Code does not auto-discover skills from site-packages, so the skill is
shipped inside the wheel and installed into a ``.claude/skills`` directory by the
``discopt-aggregation-skill`` console script.  These tests pin that the skill is
present, well-formed, and installable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from discopt.aggregation import _skill_cli

pytestmark = pytest.mark.unit

_SKILL_NAME = "variable-aggregation"


def test_repo_skill_symlinks_are_correctly_named():
    """RD1: the two in-repo skill entries (``.claude/skills`` and ``skills``) must
    be symlinks named exactly ``variable-aggregation`` resolving to the packaged
    skill source — and no Finder/Dropbox duplicate (a ``... 2`` sibling) may creep
    back in, which would register the skill under the wrong name (this repo lives
    in Dropbox, which produces such duplicates)."""
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "pyproject.toml").is_file():  # pragma: no cover - installed tree
        pytest.skip("not running from the repo checkout")
    target = (repo / "src" / "discopt" / "aggregation" / "skill").resolve()
    duplicate = re.compile(r".* \d+$")  # e.g. "variable-aggregation 2"
    for parent in (repo / ".claude" / "skills", repo / "skills"):
        assert parent.is_dir(), f"{parent} missing"
        link = parent / _SKILL_NAME
        assert link.is_symlink(), f"{link} is not a symlink"
        assert link.resolve() == target, f"{link} resolves to {link.resolve()}, want {target}"
        dups = [p.name for p in parent.iterdir() if duplicate.match(p.name)]
        assert not dups, f"Finder-duplicate skill entries in {parent}: {dups}"


def test_packaged_skill_is_present_and_well_formed():
    root = _skill_cli._packaged_skill_root()
    skill_md = root / "SKILL.md"
    assert skill_md.is_file(), f"SKILL.md missing at {root}"
    text = skill_md.read_text()
    # YAML frontmatter with name + description (name must match the directory)
    assert text.startswith("---")
    front = text.split("---", 2)[1]
    assert f"name: {_SKILL_NAME}" in front
    assert "description:" in front
    # description + when_to_use must fit Claude Code's 1536-char listing cap
    desc_line = next(line for line in front.splitlines() if line.startswith("description:"))
    assert len(desc_line) < 1536
    # supporting files referenced by the routing section exist
    for ref in ("methods", "options", "troubleshooting", "recovery-and-implicit", "study-tools"):
        assert (root / "references" / f"{ref}.md").is_file()
    for ex in ("quickstart.py", "aggressive_safely.py"):
        assert (root / "examples" / ex).is_file()


def test_skill_docs_do_not_contradict_integer_aggregation():
    """RD2: the skill must not claim integers are *never* eliminated — that
    contradicts the shipped ``integer_aggregation`` feature.  Guard against the
    stale wording creeping back into any skill file."""
    root = _skill_cli._packaged_skill_root()
    offenders = []
    for md in root.rglob("*.md"):
        text = md.read_text().lower()
        if "never eliminated" in text or "never aggregated" in text:
            offenders.append(md.name)
    assert not offenders, f"stale 'never eliminated/aggregated' wording in {offenders}"


def test_skill_study_docs_list_every_test_problem():
    """RD2: the study-tools reference must list every problem in the PROBLEMS
    registry (it silently omitted ``inventory_chain``)."""
    from discopt.aggregation.testproblems import PROBLEMS

    root = _skill_cli._packaged_skill_root()
    study = (root / "references" / "study-tools.md").read_text()
    missing = [name for name in PROBLEMS if name not in study]
    assert not missing, f"PROBLEMS not documented in study-tools.md: {missing}"


def test_installer_copies_skill_into_target(tmp_path: Path):
    base = tmp_path / ".claude" / "skills"
    dest = _skill_cli.install(base, force=False)
    assert dest == base / _SKILL_NAME
    assert (dest / "SKILL.md").is_file()
    assert (dest / "references" / "methods.md").is_file()

    # idempotency: a second install without --force refuses; with --force it overwrites
    with pytest.raises(FileExistsError):
        _skill_cli.install(base, force=False)
    dest2 = _skill_cli.install(base, force=True)
    assert dest2 == dest and (dest2 / "SKILL.md").is_file()


def test_install_force_over_symlink_destination(tmp_path: Path):
    """RD4: --force onto a symlinked destination (the in-repo skill layout) must
    replace it, not crash — shutil.rmtree raises OSError on a symlink."""
    base = tmp_path / ".claude" / "skills"
    base.mkdir(parents=True)
    # a pre-existing symlink where the skill would go
    (base / _SKILL_NAME).symlink_to(_skill_cli._packaged_skill_root())
    dest = _skill_cli.install(base, force=True)  # must not raise
    assert dest == base / _SKILL_NAME
    assert not dest.is_symlink()  # replaced by a real copied directory
    assert (dest / "SKILL.md").is_file()


def test_cli_install_user_scope(tmp_path: Path, monkeypatch):
    """RD4: `install --user` targets ~/.claude/skills."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert _skill_cli.main(["install", "--user"]) == 0
    assert (tmp_path / ".claude" / "skills" / _SKILL_NAME / "SKILL.md").is_file()


def test_cli_path_and_install_commands(tmp_path: Path, capsys, monkeypatch):
    # `path` prints the packaged location
    assert _skill_cli.main(["path"]) == 0
    assert "skill" in capsys.readouterr().out

    # `install --project` targets ./.claude/skills relative to cwd
    monkeypatch.chdir(tmp_path)
    assert _skill_cli.main(["install", "--project"]) == 0
    assert (tmp_path / ".claude" / "skills" / _SKILL_NAME / "SKILL.md").is_file()
