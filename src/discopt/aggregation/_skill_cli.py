"""Install the bundled Claude skill into a ``.claude/skills`` directory.

Claude Code discovers skills only from ``~/.claude/skills`` (personal),
``<project>/.claude/skills`` (project), and installed plugins — it does **not**
scan pip/site-packages.  So when this package is pip-installed, its skill is not
automatically visible to Claude.  This console script (``discopt-aggregation-skill``)
copies the skill — shipped inside the wheel under ``discopt/aggregation/skill`` —
into one of those discovered locations.

    discopt-aggregation-skill install            # -> ./.claude/skills/ (project)
    discopt-aggregation-skill install --user     # -> ~/.claude/skills/ (personal)
    discopt-aggregation-skill path               # print the packaged skill path
"""

from __future__ import annotations

import argparse
import shutil
import sys
from importlib import resources
from pathlib import Path

_SKILL_NAME = "variable-aggregation"


def _packaged_skill_root() -> Path:
    """Filesystem path to the skill bundled in the installed package."""
    # importlib.resources.files works for regular packages (this subpackage has
    # an __init__.py); ``as_file`` is unnecessary since wheels unpack to disk.
    return Path(str(resources.files("discopt.aggregation") / "skill"))


def install(target_dir: Path, *, force: bool) -> Path:
    """Copy the bundled skill into ``target_dir/<skill-name>``; return that path."""
    src = _packaged_skill_root()
    if not (src / "SKILL.md").is_file():
        raise FileNotFoundError(
            f"packaged skill not found at {src} (SKILL.md missing). "
            "Reinstall discopt-aggregation, or run from a wheel that includes the skill."
        )
    dest = target_dir / _SKILL_NAME
    if dest.exists():
        if not force:
            raise FileExistsError(
                f"{dest} already exists; pass --force to overwrite, or remove it first."
            )
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # copy only the skill content (SKILL.md, README, references/, examples/)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="discopt-aggregation-skill",
        description="Install the discopt.aggregation Claude skill into a .claude/skills directory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser("install", help="copy the skill into a .claude/skills directory")
    scope = p_install.add_mutually_exclusive_group()
    scope.add_argument(
        "--user",
        action="store_const",
        dest="base",
        const=Path.home() / ".claude" / "skills",
        help="install into ~/.claude/skills (available in every project)",
    )
    scope.add_argument(
        "--project",
        action="store_const",
        dest="base",
        const=Path.cwd() / ".claude" / "skills",
        help="install into ./.claude/skills (this project only) [default]",
    )
    p_install.add_argument("--force", action="store_true", help="overwrite an existing copy")

    sub.add_parser("path", help="print the path to the skill bundled in the package")

    args = parser.parse_args(argv)

    if args.command == "path":
        print(_packaged_skill_root())
        return 0

    base = args.base or (Path.cwd() / ".claude" / "skills")
    try:
        dest = install(base, force=args.force)
    except (FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"installed skill -> {dest}")
    print("Claude Code will discover it on its next scan of that .claude/skills directory.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
