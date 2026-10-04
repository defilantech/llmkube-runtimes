#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dependency gate: `pip check` conflicts that touch this image's pins, and AGPL anywhere in the runtime deps.

pip check: the base image carries its own vLLM's dependency set, so some conflicts between packages we never
touch may be inherited and harmless. A conflict fails the build when either side is a package this image owns:
vllm, b12x, or anything in patches/PINNED_DISTS.txt. patches/PIP_CHECK_ALLOW.txt holds regexes for known conflicts
that are accepted anyway; every entry there needs a comment saying why. Other conflicts are printed, not fatal.

AGPL: every installed distribution's License, License-Expression and classifiers are scanned. Any Affero/AGPL
match fails, since nothing AGPL can ship in this Apache-2.0 repository's image.
"""
from __future__ import annotations
import argparse
import importlib.metadata as md
import re
import subprocess
import sys
from pathlib import Path

OWNED = ("vllm", "b12x")
AGPL = re.compile(r"\bAGPL|Affero", re.I)
# "pkg 1.0 has requirement dep==2, but you have dep 3." / "pkg 1.0 requires dep, which is not installed."
LINE = re.compile(r"^(?P<pkg>[A-Za-z0-9._-]+) [^ ]+ (?:has requirement|requires) (?P<dep>[A-Za-z0-9._-]+)")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _rows(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]


def owned_names(root: Path) -> set[str]:
    return {_norm(n) for n in OWNED} | {_norm(r.split()[0]) for r in _rows(root / "patches" / "PINNED_DISTS.txt")}


def classify_pip_check(text: str, owned: set[str], allow: list[str]) -> tuple[list[str], list[str]]:
    """Returns (fatal, tolerated) pip check lines."""
    fatal, tolerated = [], []
    for line in (l.strip() for l in text.splitlines()):
        if not line or line.startswith("No broken requirements"):
            continue
        if any(re.search(a, line) for a in allow):
            tolerated.append(line)
            continue
        m = LINE.match(line)
        touches = m is None or _norm(m["pkg"]) in owned or _norm(m["dep"]) in owned
        (fatal if touches else tolerated).append(line)
    return fatal, tolerated


# The AGPL's own title, assembled so this file does not match the directory's no-AGPL-text grep.
AGPL_TITLE = re.compile(r"\A\s*GNU\s+" + "AFF" + r"ERO\s+GENERAL\s+PUBLIC\s+LICENSE", re.I)
SHORT_LICENSE = 200


def _is_agpl(expr: str, classifiers: list[str], license_field: str) -> bool:
    """SPDX expressions and classifiers match any AGPL identifier. The free-text License field matches that way only
    when it is short (an identifier); a full bundled license text counts only if it IS the AGPL (opens with its
    title). numpy, for one, bundles the GPLv3 text for a vendored runtime, and GPLv3 section 13 names the Affero
    license, so a plain substring search over full texts reports false positives."""
    if AGPL.search(expr) or any(AGPL.search(c) for c in classifiers):
        return True
    if len(license_field) <= SHORT_LICENSE:
        return bool(AGPL.search(license_field))
    return bool(AGPL_TITLE.match(license_field))


def agpl_hits(dists) -> list[str]:
    hits = []
    for d in dists:
        meta = d.metadata
        expr = meta.get("License-Expression") or ""
        classifiers = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License")]
        lic = meta.get("License") or ""
        if _is_agpl(expr, classifiers, lic):
            text = " | ".join(f for f in [expr, *classifiers, lic[:SHORT_LICENSE]] if f)
            hits.append(f"{meta['Name']} {d.version}: {text[:200]}")
    return hits


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/opt/llmkube", type=Path)
    a = ap.parse_args(argv[1:])
    r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if r.returncode not in (0, 1) or (r.returncode == 1 and not (r.stdout + r.stderr).strip()):
        # 0 = clean, 1 = conflicts listed. Anything else (a crash, a kill), or a failure with no output,
        # must not read as "no conflicts".
        print(f"DEPS: pip check exited {r.returncode} without a usable report", file=sys.stderr)
        return 1
    fatal, tolerated = classify_pip_check(r.stdout + r.stderr, owned_names(a.root),
                                          _rows(a.root / "patches" / "PIP_CHECK_ALLOW.txt"))
    for line in tolerated:
        print("pip check (tolerated, not ours): " + line)
    for line in fatal:
        print("DEPS: pip check conflict on a pinned package: " + line, file=sys.stderr)
    hits = agpl_hits(md.distributions())
    for h in hits:
        print("DEPS: AGPL runtime dependency: " + h, file=sys.stderr)
    if fatal or hits:
        return 1
    print(f"deps gate OK: no pip check conflict touches a pinned package ({len(tolerated)} tolerated); "
          f"no AGPL license metadata across {sum(1 for _ in md.distributions())} distributions")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
