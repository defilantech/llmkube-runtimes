#!/usr/bin/env python3
"""Copy the vendored patch files to their destinations and verify them (spec 5.4).

mounts.txt (tonyd2wild's format) lists "<file> <dest>" where dest is relative to the vllm package
("models/…", "model_executor/…", "v1/…") or an absolute dist-packages path for the two cuda_exl3
overlays. Destinations must already exist as directories in the base: this script never creates
package directories, so a renamed upstream tree fails here instead of producing an orphan file.
"""
from __future__ import annotations
import hashlib
import shutil
import sys
from pathlib import Path

EXL3_ABS_PREFIX = "/usr/local/lib/python3.12/dist-packages/cuda_exl3/"


def md5_prefix(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def resolve(dest: str, site: Path, exl3_site: Path) -> Path:
    if dest.startswith(EXL3_ABS_PREFIX):
        return exl3_site / dest[len(EXL3_ABS_PREFIX):]
    if dest.startswith("/"):
        raise ValueError(f"unexpected absolute destination outside cuda_exl3: {dest}")
    return site / dest


def _mounts(patch_dir: Path) -> list[tuple[str, str]]:
    out = []
    for line in (patch_dir / "mounts.txt").read_text().splitlines():
        if line.strip():
            src, dest = line.split()
            out.append((src, dest))
    return out


def apply(patch_dir: Path, site: Path, exl3_site: Path, extra: dict[str, tuple[str, str]]) -> list[tuple[Path, Path]]:
    plan: list[tuple[Path, Path]] = []
    for src, dest in _mounts(patch_dir):
        plan.append((patch_dir / src, resolve(dest, site, exl3_site)))
    for src, (rel, _want) in extra.items():
        plan.append((patch_dir / src, site / rel))
    for src, dst in plan:
        if not dst.parent.is_dir():
            raise FileNotFoundError(f"destination directory missing in the base image: {dst.parent}")
        shutil.copyfile(src, dst)
    return plan


def verify(patch_dir: Path, copies: list[tuple[Path, Path]], extra: dict[str, tuple[str, str]]) -> list[str]:
    want: dict[str, str] = {}
    for line in (patch_dir / "MD5SUMS.txt").read_text().splitlines():
        if line.strip():
            prefix, name = line.split()
            want[name] = prefix
    for src, (_rel, prefix) in extra.items():
        want[src] = prefix
    errors: list[str] = []
    for src, dst in copies:
        expected = want.get(src.name)
        if expected is None:
            errors.append(f"{src.name}: no expected MD5 prefix recorded")
            continue
        got = md5_prefix(dst)
        if got != expected:
            errors.append(f"{src.name} -> {dst}: has {got}, expected {expected}")
    return errors


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--extra")]
    extras = [a.split("=", 1)[1] for a in argv[1:] if a.startswith("--extra=")]
    if len(args) != 3:
        print("usage: apply_patches.py <patch-dir> <vllm-site> <cuda-exl3-site> [--extra=<file>=<relpath>:<8hex> ...]", file=sys.stderr)
        return 2
    extra: dict[str, tuple[str, str]] = {}
    for e in extras:
        src, rest = e.split("=", 1)
        rel, prefix = rest.rsplit(":", 1)
        extra[src] = (rel, prefix)
    patch_dir, site, exl3_site = (Path(a) for a in args)
    copies = apply(patch_dir, site, exl3_site, extra)
    errors = verify(patch_dir, copies, extra)
    for err in errors:
        print("PATCH: " + err, file=sys.stderr)
    if errors:
        return 1
    for src, dst in copies:
        print(f"patched {dst} <- {src.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
