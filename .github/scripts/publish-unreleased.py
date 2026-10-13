#!/usr/bin/env python3
"""Publish every workspace crate whose version is not on crates.io yet, dependencies first.

Releasing a crate is bumping its `version`; a crate whose version is already published is
skipped, so a push that bumps nothing publishes nothing. `cargo publish` waits until a crate
is in the index before returning, so the crates depending on it resolve against it next.

    publish-unreleased.py [--dry-run]

Needs CARGO_REGISTRY_TOKEN unless --dry-run.
"""

import json
import subprocess
import sys
import urllib.error
import urllib.request


def published(name, version):
    # The API, not the sparse index: the index sits behind a CDN cache and can lag a publish.
    request = urllib.request.Request(
        f"https://crates.io/api/v1/crates/{name}/{version}",
        headers={"User-Agent": "feuclide-ci (https://github.com/Dastfox/feuclide)"},
    )
    try:
        with urllib.request.urlopen(request):
            return True
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise


def dependency_order(crates):
    # Dev-dependencies count too: crates.io refuses a manifest naming a crate it does not have.
    order, done = [], set()

    def visit(name, path):
        if name in done:
            return
        if name in path:
            sys.exit(f"dependency cycle: {' -> '.join(path + [name])}")
        for dep in crates[name]["dependencies"]:
            if dep["name"] in crates and dep["name"] != name:
                visit(dep["name"], path + [name])
        done.add(name)
        order.append(name)

    for name in sorted(crates):
        visit(name, [])
    return order


def main():
    dry_run = "--dry-run" in sys.argv[1:]
    metadata = json.loads(
        subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--no-deps"])
    )
    # `publish = false` reads as an empty list, an unrestricted crate as null.
    crates = {p["name"]: p for p in metadata["packages"] if p["publish"] is None}

    for name in dependency_order(crates):
        version = crates[name]["version"]
        if published(name, version):
            print(f"{name} {version}: already on crates.io")
            continue
        print(f"{name} {version}: publishing", flush=True)
        if not dry_run:
            subprocess.run(["cargo", "publish", "--locked", "-p", name], check=True)


if __name__ == "__main__":
    main()
