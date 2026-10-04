#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pins gate: the image is exactly what patches/ says it is, or the build fails.

Run at build (after every install) and again by the Tier-1 gate against the shipped image. Checks:

  UPSTREAM_COMMITS.txt  "name repo commit": /src/<name> is a clean checkout of that repo at that commit, the
                        matching build ARG (VLLM_FORK_COMMIT, B12X_COMMIT) agrees when set, and every git-tracked
                        <name>/**/*.py is byte-identical in dist-packages (so the installed package IS that commit).
  PINNED_DISTS.txt      "dist version [sha256 url]": exactly one distribution of that name is installed, at that
                        version. The sha256/url column is consumed by the Dockerfile, which verifies before install.
  vllm-install.txt      "mode sha256 source" written by the vLLM install step: mode is source (compiled here) or
                        wheel (VLLM_WHEEL_URL, hash-verified); a set VLLM_WHEEL_SHA256 must match it. The single
                        installed vllm's version must carry the fork commit (setuptools-scm's +g<hash>).
  MD5SUMS.txt           "md5-prefix file" for vendored files in patches/; an empty list is valid.
  nccl-build.txt        native pins (NCCL) have no Python tree to compare. Their checkout and carried patches are checked
                        as above, each patch marker must be in the patched checkout, and the build record written by the
                        Dockerfile's nccl-build stage ("commit <sha>", "patches <sha256,...|none>", "lib <sha256>") must
                        name the pinned commit and exactly the carried patches, and its library sha256 must be the
                        installed nvidia/nccl/lib/libnccl.so.2.
  <name>/*.patch        an open upstream PR carried on top of the pin (patches/<name>/APPLIED.md records each one:
                        file, upstream, head SHA, author, scope, form, marker, sha256). The checkout may then differ
                        from the pin only by exactly those patches, applied in filename order (new files `git add -N`):
                        its changed and new files must be the patches' files, reversing the patches must leave a
                        clean checkout at the pin, every file a patch touches under <name>/ is byte-identical in
                        dist-packages, and each patch's marker string appears in one of those installed files.

Both pin files must name what this image is (vllm, b12x and nccl commits; torch, flashinfer-python and b12x versions): an
emptied or truncated file fails rather than checking nothing. Origins must be exactly the pinned GitHub repo.

Every failure names the pin it broke. Exit 0 clean, 1 on any failure, 2 on usage.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as md
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

COMMIT_ENV = {"vllm": "VLLM_FORK_COMMIT", "b12x": "B12X_COMMIT", "nccl": "NCCL_COMMIT"}
# An emptied or truncated pin file must not pass by checking nothing.
REQUIRED_COMMITS = ("vllm", "b12x", "nccl")
# Pins built into a native library rather than installed as a Python package: name -> path under site-packages.
NATIVE = {"nccl": "nvidia/nccl/lib/libnccl.so.2"}
REQUIRED_DISTS = ("torch", "flashinfer-python", "b12x")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
DIFF_GIT = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)


def _rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def origin_matches(origin: str, repo: str) -> bool:
    """Exact GitHub repo match over https or ssh remotes, so local-inference-lab/vllm-evil is not vllm."""
    o = origin.strip().rstrip("/")
    o = o[:-4] if o.endswith(".git") else o
    for prefix in ("https://github.com/", "git@github.com:", "ssh://git@github.com/"):
        if o.startswith(prefix):
            return o[len(prefix):].lower() == repo.lower()
    return False


def installed(site: Path, dist: str) -> list[str]:
    return [d.version for d in md.distributions(path=[str(site)]) if _norm(d.metadata["Name"] or "") == _norm(dist)]


def _applied_rows(path: Path) -> dict[str, dict[str, str]]:
    """APPLIED.md table rows keyed by patch file name (cells: file upstream head author scope form marker sha256)."""
    out: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 8 or not cells[0].endswith(".patch"):
            continue
        out[cells[0]] = {"upstream": cells[1], "marker": cells[6], "sha256": cells[7]}
    return out


def _status_paths(repo: Path) -> set[str]:
    """Paths that differ from HEAD in the index or the work tree, including untracked files."""
    r = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "-z", "--untracked-files=all"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git status: {r.stderr.strip()}")
    paths: set[str] = set()
    entries = iter(r.stdout.split("\0"))
    for entry in entries:
        if len(entry) > 3:
            paths.add(entry[3:])
            if entry[0] in "RC":  # a rename or copy record is followed by its source path
                next(entries, None)
    return paths


def check_patches(name: str, pdir: Path, checkout: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    """Verify the checkout is the pin plus exactly the carried patches; returns errors, touched files, markers and the
    patches' sha256 in filename order."""
    import shutil
    import tempfile
    errors: list[str] = []
    patches = sorted(pdir.glob("*.patch"))
    rows = _applied_rows(pdir / "APPLIED.md")
    touched: list[str] = []
    source: dict[str, str] = {}
    markers: list[str] = []
    shas: list[str] = []
    for p in patches:
        row = rows.get(p.name)
        if row is None:
            errors.append(f"{name}: {p.name} has no row in {pdir.name}/APPLIED.md")
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        shas.append(got)
        if got != row["sha256"]:
            errors.append(f"{name}: {p.name} sha256 {got}, APPLIED.md records {row['sha256']}")
        markers.append(row["marker"])
        for _, b in DIFF_GIT.findall(p.read_text(errors="replace")):
            if b not in touched:
                touched.append(b)
                source[b] = p.name
    changed = _status_paths(checkout)
    for path in sorted(changed - set(touched)):
        errors.append(f"{name}: {checkout} changes {path}, which no carried patch in {pdir} touches")
    for path in sorted(set(touched) - changed):
        errors.append(f"{name}: {checkout} does not carry {path}, which {source.get(path, 'a recorded patch')} changes")
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / name
        shutil.copytree(checkout, copy, symlinks=True)
        try:
            _git(copy, "reset", "-q")
            for p in reversed(patches):
                _git(copy, "apply", "-R", "--whitespace=nowarn", str(p))
            left = _status_paths(copy)
        except RuntimeError as e:
            errors.append(f"{name}: reversing the carried patches does not restore the pin ({e})")
        else:
            if left:
                errors.append(f"{name}: after reversing the carried patches, {' '.join(sorted(left))} still differ from the pin")
    return errors, touched, markers, shas


def check_native_build(root: Path, site: Path, name: str, commit: str, shas: list[str]) -> list[str]:
    """The build record names the pinned commit and exactly the carried patches, and its library is the installed one."""
    rec = root / f"{name}-build.txt"
    if not rec.is_file():
        return [f"{name}: {rec} missing; the {name}-build stage did not record what it built"]
    fields = dict(line.split(None, 1) for line in rec.read_text().splitlines() if len(line.split(None, 1)) == 2)
    errors: list[str] = []
    if fields.get("commit") != commit:
        errors.append(f"{name}: {rec} records commit {fields.get('commit')}, UPSTREAM_COMMITS pins {commit}")
    want = ",".join(shas) or "none"
    if fields.get("patches") != want:
        errors.append(f"{name}: {rec} records patches {fields.get('patches')}, patches/{name} carries {want}")
    lib_sha = fields.get("lib", "")
    if not HEX64.fullmatch(lib_sha):
        errors.append(f"{name}: {rec} has no lib sha256")
        return errors
    lib = site / NATIVE[name]
    if not lib.is_file():
        errors.append(f"{name}: {lib} is not installed")
    elif hashlib.sha256(lib.read_bytes()).hexdigest() != lib_sha:
        errors.append(f"{name}: installed {NATIVE[name]} is not the library the {name}-build stage recorded ({lib_sha[:12]})")
    return errors


def check_commits(root: Path, src: Path, site: Path) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    pins: dict[str, str] = {}
    for row in _rows(root / "patches" / "UPSTREAM_COMMITS.txt"):
        if len(row) != 3 or not HEX40.fullmatch(row[2]):
            errors.append(f"{row[0]}: malformed UPSTREAM_COMMITS line {' '.join(row)!r} (want: name repo <40-hex commit>)")
            continue
        name, repo, commit = row
        pins[name] = commit
        env = COMMIT_ENV.get(name)
        if env and os.environ.get(env) and os.environ[env] != commit:
            errors.append(f"{name}: build ARG {env}={os.environ[env]} disagrees with UPSTREAM_COMMITS {commit}")
        checkout = src / name
        try:
            head = _git(checkout, "rev-parse", "HEAD")
            origin = _git(checkout, "remote", "get-url", "origin")
            dirty = _git(checkout, "status", "--porcelain", "--untracked-files=no")
            tracked = _git(checkout, "ls-files", "-z", "--", f"{name}/*.py").split("\0")
        except (RuntimeError, FileNotFoundError) as e:
            errors.append(f"{name}: {checkout} is not a usable git checkout ({e})")
            continue
        if head != commit:
            errors.append(f"{name}: {checkout} is at {head}, UPSTREAM_COMMITS pins {commit} ({repo})")
            continue
        if not origin_matches(origin, repo):
            errors.append(f"{name}: {checkout} origin is {origin}, UPSTREAM_COMMITS pins repo {repo}")
        pdir = root / "patches" / name
        patched: list[str] = []
        markers: list[str] = []
        shas: list[str] = []
        if pdir.is_dir() and any(pdir.glob("*.patch")):
            perrors, patched, markers, shas = check_patches(name, pdir, checkout)
            errors += perrors
        elif dirty:
            errors.append(f"{name}: {checkout} has modified tracked files: {' '.join(dirty.split())}")
        if name in NATIVE:
            for marker in markers:
                if not any((checkout / f).is_file() and marker in (checkout / f).read_text(errors="replace") for f in patched):
                    errors.append(f"{name}: carried patch marker {marker!r} is in none of the patched source files")
            errors += check_native_build(root, site, name, commit, shas)
            continue
        tracked = sorted({t for t in tracked if t} | {f for f in patched if f.startswith(name + "/")})
        if not tracked:
            errors.append(f"{name}: no tracked {name}/*.py in {checkout}; the package layout changed")
        for rel in tracked:
            dst = site / rel
            if not (checkout / rel).is_file():
                continue  # a carried patch's file missing from the checkout is reported by check_patches
            if not dst.is_file():
                errors.append(f"{name}: {rel} is tracked at {commit[:12]} but not installed in {site}")
            elif dst.read_bytes() != (checkout / rel).read_bytes():
                errors.append(f"{name}: installed {rel} differs from {commit[:12]}" + (" + carried patches" if patched else ""))
        for marker in markers:
            hits = [f for f in patched if (site / f).is_file() and marker in (site / f).read_text(errors="replace")]
            if not hits:
                errors.append(f"{name}: carried patch marker {marker!r} is in none of the installed patched files")
    for name in REQUIRED_COMMITS:
        if name not in pins:
            errors.append(f"{name}: no entry in UPSTREAM_COMMITS.txt; the image cannot be checked against a pin")
    return errors, pins


def check_dists(root: Path, site: Path) -> list[str]:
    errors: list[str] = []
    rows = _rows(root / "patches" / "PINNED_DISTS.txt")
    names = {_norm(r[0]) for r in rows}
    for dist in REQUIRED_DISTS:
        if _norm(dist) not in names:
            errors.append(f"{dist}: no entry in PINNED_DISTS.txt; the image cannot be checked against a pin")
    for row in rows:
        if len(row) not in (2, 4) or (len(row) == 4 and not HEX64.fullmatch(row[2])):
            errors.append(f"{row[0]}: malformed PINNED_DISTS line {' '.join(row)!r} (want: dist version [sha256 url])")
            continue
        dist, want = row[0], row[1]
        got = installed(site, dist)
        if len(got) != 1:
            errors.append(f"{dist}: pinned {want}, {len(got)} installed {got}")
        elif got[0] != want:
            errors.append(f"{dist}: pinned {want}, installed {got[0]}")
    return errors


def check_vllm_install(root: Path, src: Path, site: Path, commit: str) -> list[str]:
    errors: list[str] = []
    rec = root / "vllm-install.txt"
    if not rec.is_file():
        return [f"vllm: {rec} missing; the vLLM install step did not record how vllm got here"]
    fields = rec.read_text().split()
    if len(fields) != 3 or fields[0] not in ("source", "wheel") or not HEX64.fullmatch(fields[1]):
        return [f"vllm: malformed {rec}: {' '.join(fields)!r} (want: source|wheel <sha256> <origin>)"]
    mode, sha, origin = fields
    want_url, want_sha = os.environ.get("VLLM_WHEEL_URL", ""), os.environ.get("VLLM_WHEEL_SHA256", "")
    if (want_url or want_sha) and mode != "wheel":
        errors.append(f"vllm: VLLM_WHEEL_URL/VLLM_WHEEL_SHA256 are set but {rec} records a {mode} build")
    if mode == "wheel" and want_sha and sha != want_sha:
        errors.append(f"vllm: installed wheel sha256 {sha}, VLLM_WHEEL_SHA256 pins {want_sha}")
    if mode == "wheel" and want_url and origin != want_url:
        errors.append(f"vllm: installed wheel came from {origin}, VLLM_WHEEL_URL pins {want_url}")
    versions = installed(site, "vllm")
    if len(versions) != 1:
        errors.append(f"vllm: {len(versions)} installed {versions}; the base image's vllm must be gone")
    elif f"g{commit[:7]}" not in versions[0]:
        try:
            tagged = bool(_git(src / "vllm", "describe", "--tags", "--exact-match", "HEAD"))
        except RuntimeError:
            tagged = False
        if not tagged:
            errors.append(f"vllm: installed version {versions[0]} does not carry fork commit g{commit[:7]}")
    return errors


def check_md5(root: Path) -> tuple[list[str], int]:
    rows = _rows(root / "patches" / "MD5SUMS.txt")
    errors: list[str] = []
    for row in rows:
        if len(row) != 2:
            errors.append(f"MD5SUMS: malformed line {' '.join(row)!r}")
            continue
        want, name = row
        p = root / "patches" / name
        if not p.is_file():
            errors.append(f"{name}: listed in MD5SUMS.txt but missing from patches/")
            continue
        got = hashlib.md5(p.read_bytes()).hexdigest()
        if not got.startswith(want):
            errors.append(f"{name}: md5 {got[:len(want)]}, MD5SUMS.txt pins {want}")
    return errors, len(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/opt/llmkube", type=Path, help="holds patches/ and vllm-install.txt")
    ap.add_argument("--src", default="/src", type=Path, help="holds one checkout per UPSTREAM_COMMITS name")
    ap.add_argument("--site", default=sysconfig.get_paths()["purelib"], type=Path, help="dist-packages")
    a = ap.parse_args(argv[1:])
    errors, pins = check_commits(a.root, a.src, a.site)
    errors += check_dists(a.root, a.site)
    if "vllm" in pins:
        errors += check_vllm_install(a.root, a.src, a.site, pins["vllm"])
    md5_errors, vendored = check_md5(a.root)
    errors += md5_errors
    for e in errors:
        print("PIN: " + e, file=sys.stderr)
    if errors:
        return 1
    print(f"pins gate OK: {', '.join(f'{n}@{c[:12]}' for n, c in pins.items())}; "
          f"{len(_rows(a.root / 'patches' / 'PINNED_DISTS.txt'))} dists at pinned versions; {vendored} vendored files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
