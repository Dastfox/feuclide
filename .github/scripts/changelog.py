#!/usr/bin/env python3
"""CHANGELOG.md: every commit adds a line, and a release moves the new lines under its heading.

    changelog.py check-staged [--amend]   the commit being made adds a line (pre-commit hook)
    changelog.py check-range BASE HEAD    every commit in BASE..HEAD added one (CI)
    changelog.py release                  move Unreleased under today's heading (pre-push hook)

A line is an entry, `- ` at the start. A commit adds one when the file has more entries after
it than before, so moving entries under a release heading neither adds nor loses any.
`release` does nothing when Unreleased is empty or no crate's version is new to crates.io.
"""

import datetime
import pathlib
import subprocess
import sys

PATH = "CHANGELOG.md"
UNRELEASED = "## Unreleased"
SCRIPTS = pathlib.Path(__file__).resolve().parent


def git(*args, check=True):
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result


def entries(text):
    return sum(1 for line in text.splitlines() if line.startswith("- "))


def entries_at(revision):
    """Entries in the changelog at a revision (`:` for the index), 0 where there is none."""
    shown = git("show", f"{revision}{PATH}", check=False)
    return entries(shown.stdout) if shown.returncode == 0 else 0


def check_staged(amend):
    parent = "HEAD^" if amend else "HEAD"
    before = entries_at(f"{parent}:") if git("rev-parse", "-q", "--verify", parent, check=False).returncode == 0 else 0
    if entries_at(":") <= before:
        sys.exit(
            f"Every commit adds a line to {PATH}, under {UNRELEASED!r}: `- what changed`.\n"
            f"Stage it with the rest, or commit with --no-verify for a commit CI will refuse."
        )


def check_range(base, head):
    if git("cat-file", "-e", f"{base}^{{commit}}", check=False).returncode != 0:
        # A new branch or a forced push: check what is not on main yet.
        base = git("merge-base", "origin/main", head).stdout.strip()
    missing = []
    for commit in git("rev-list", "--no-merges", "--reverse", f"{base}..{head}").stdout.split():
        if entries_at(f"{commit}:") <= entries_at(f"{commit}^:"):
            missing.append(git("log", "-1", "--format=%h %s", commit).stdout.strip())
    if missing:
        print(f"These commits add no line to {PATH}:", *missing, sep="\n  ")
        sys.exit(1)


def release():
    text = pathlib.Path(PATH).read_text()
    lines = text.splitlines()
    start = lines.index(UNRELEASED) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    moved = [line for line in lines[start:end] if line.strip()]
    if entries("\n".join(moved)) == 0:
        return
    listed = subprocess.run(
        [sys.executable, SCRIPTS / "publish-unreleased.py", "--list"], capture_output=True, text=True
    )
    if listed.returncode != 0:
        print(f"changelog: could not ask crates.io what this push releases, {PATH} left as is:\n"
              f"{listed.stderr.strip()}", file=sys.stderr)
        return
    crates = [line for line in listed.stdout.splitlines() if line.strip()]
    if not crates:
        return
    heading = [f"## {datetime.date.today().isoformat()}", "", ", ".join(crates), "", *moved, ""]
    lines[start:end] = ["", *heading]
    pathlib.Path(PATH).write_text("\n".join(lines) + "\n")


def main():
    match sys.argv[1:]:
        case ["check-staged"]:
            check_staged(amend=False)
        case ["check-staged", "--amend"]:
            check_staged(amend=True)
        case ["check-range", base, head]:
            check_range(base, head)
        case ["release"]:
            release()
        case _:
            sys.exit(__doc__)


if __name__ == "__main__":
    main()
