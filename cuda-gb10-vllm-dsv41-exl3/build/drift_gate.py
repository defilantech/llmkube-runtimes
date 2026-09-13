#!/usr/bin/env python3
"""Drift gate for the vLLM patch overlay (spec 5.1).

Refuses to let the build overwrite a vLLM file whose content is not the one the vendored patch set was
written against. The baseline file holds the MD5 prefix of every file the build will overwrite as
found in the pinned base image, plus ABSENT entries for files the patch set adds.
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path


def md5_prefix(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def load_baseline(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        want, rel = line.split()
        out[rel] = want
    return out


def check(site: Path, baseline: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for rel, want in baseline.items():
        p = site / rel
        if want == "ABSENT":
            if p.exists():
                errors.append(f"{rel}: expected ABSENT in the base image, but it exists")
            continue
        if not p.exists():
            errors.append(f"{rel}: missing from the base image")
            continue
        got = md5_prefix(p)
        if got != want:
            errors.append(f"{rel}: base has {got}, baseline expects {want} (base image drifted; re-derive the patch set)")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: drift_gate.py <vllm-site-dir> <baseline-file>", file=sys.stderr)
        return 2
    errors = check(Path(argv[1]), load_baseline(Path(argv[2])))
    for e in errors:
        print("DRIFT: " + e, file=sys.stderr)
    if errors:
        return 1
    print("drift gate OK: every overwrite target matches the recorded base")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
