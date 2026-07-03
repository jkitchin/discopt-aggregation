"""The bundled Claude skill ships in the package and installs discoverably.

Claude Code does not auto-discover skills from site-packages, so the skill is
shipped inside the wheel and installed into a ``.claude/skills`` directory by the
``discopt-aggregation-skill`` console script.  These tests pin that the skill is
present, well-formed, and installable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from discopt.aggregation import _skill_cli

pytestmark = pytest.mark.unit

_SKILL_NAME = "variable-aggregation"


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


def test_cli_path_and_install_commands(tmp_path: Path, capsys, monkeypatch):
    # `path` prints the packaged location
    assert _skill_cli.main(["path"]) == 0
    assert "skill" in capsys.readouterr().out

    # `install --project` targets ./.claude/skills relative to cwd
    monkeypatch.chdir(tmp_path)
    assert _skill_cli.main(["install", "--project"]) == 0
    assert (tmp_path / ".claude" / "skills" / _SKILL_NAME / "SKILL.md").is_file()
