# SPDX-License-Identifier: Apache-2.0
"""pins_gate.py against a synthetic image: real git checkouts, a fake dist-packages, one wrong pin at a time.

Runs on the host (git and python only) and again inside the image, where it proves the gate itself still
behaves before the gate's verdict on the real tree is trusted.
"""
from __future__ import annotations
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE.parent
GATE = IMAGE_DIR / "build" / "pins_gate.py"
PIN_ENV = ("VLLM_FORK_COMMIT", "B12X_COMMIT", "NCCL_COMMIT", "VLLM_WHEEL_URL", "VLLM_WHEEL_SHA256")


def _git(cwd: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def _checkout(src: Path, name: str, repo: str, files: dict[str, str]) -> str:
    d = src / name
    for rel, body in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    _git(d.parent, "init", "-q", name)
    _git(d, "remote", "add", "origin", f"https://github.com/{repo}.git")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "pinned")
    return _git(d, "rev-parse", "HEAD")


def _dist(site: Path, name: str, version: str) -> None:
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


VLLM_FILES = {"vllm/__init__.py": "V = 1\n", "vllm/models/deepseek_v4_1/attention.py": "import b12x\n",
              "requirements/cuda.txt": "torch==2.13.0\n", "tests/test_x.py": "def test_x(): pass\n"}
B12X_FILES = {"b12x/__init__.py": "", "b12x/attention/__init__.py": "", "b12x/attention/mla.py": "K = 2\n"}
NCCL_FILES = {"src/transport/net_ib/p2p.cc": "int isend() {\n  return 0;\n}\n", "makefiles/version.mk": "NCCL_PATCH := 7\n"}
NCCL_LIB = "nvidia/nccl/lib/libnccl.so.2"


def _nccl_record(image, lib_bytes: bytes | None = None, patches: str = "none", commit: str | None = None) -> None:
    lib = image["site"] / NCCL_LIB
    if lib_bytes is not None:
        lib.parent.mkdir(parents=True, exist_ok=True)
        lib.write_bytes(lib_bytes)
    sha = hashlib.sha256(lib.read_bytes()).hexdigest()
    (image["root"] / "nccl-build.txt").write_text(
        f"commit {commit or image['nccl']}\npatches {patches}\nlib {sha}\ngencode sm_120 compute_120\n")


@pytest.fixture()
def image(tmp_path):
    """A consistent synthetic image. Tests break exactly one thing and expect the gate to name it."""
    src, site, root = tmp_path / "src", tmp_path / "site", tmp_path / "opt"
    vllm_sha = _checkout(src, "vllm", "local-inference-lab/vllm", VLLM_FILES)
    b12x_sha = _checkout(src, "b12x", "local-inference-lab/b12x", B12X_FILES)
    nccl_sha = _checkout(src, "nccl", "NVIDIA/nccl", NCCL_FILES)
    for name, files in (("vllm", VLLM_FILES), ("b12x", B12X_FILES)):
        for rel, body in files.items():
            if rel.startswith(name + "/"):
                p = site / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(body)
    _dist(site, "vllm", f"0.1.dev1+g{vllm_sha[:9]}.cu130")
    _dist(site, "torch", "2.13.0+cu130")
    _dist(site, "flashinfer-python", "0.6.18.post1")
    _dist(site, "b12x", "1.3.0")
    (root / "patches").mkdir(parents=True)
    (root / "patches" / "UPSTREAM_COMMITS.txt").write_text(
        f"# name repo commit\nvllm local-inference-lab/vllm {vllm_sha}\nb12x local-inference-lab/b12x {b12x_sha}\n"
        f"nccl NVIDIA/nccl {nccl_sha}\n")
    (root / "patches" / "PINNED_DISTS.txt").write_text(
        "# dist version [sha256 url]\ntorch 2.13.0+cu130\nb12x 1.3.0\n"
        f"flashinfer-python 0.6.18.post1 {'a' * 64} https://example.invalid/flashinfer_python-0.6.18.post1-py3-none-any.whl\n")
    (root / "patches" / "MD5SUMS.txt").write_text("# md5-prefix file (vendored files only; none today)\n")
    (root / "vllm-install.txt").write_text(f"source {'b' * 64} /src/vllm\n")
    img = {"src": src, "site": site, "root": root, "vllm": vllm_sha, "b12x": b12x_sha, "nccl": nccl_sha}
    _nccl_record(img, b"\x7fELF fenced libnccl 2.30.7")
    return img


def run_gate(image, **env):
    base = {k: v for k, v in os.environ.items() if k not in PIN_ENV}
    base.update(env)
    return subprocess.run([sys.executable, str(GATE), "--root", str(image["root"]), "--src", str(image["src"]),
                           "--site", str(image["site"])], env=base, capture_output=True, text=True)


def test_consistent_image_passes(image):
    r = run_gate(image)
    assert r.returncode == 0, r.stderr
    assert "pins gate OK" in r.stdout


def test_wrong_commit_pin_fails_and_names_it(image):
    wrong = "0" * 40
    f = image["root"] / "patches" / "UPSTREAM_COMMITS.txt"
    f.write_text(f.read_text().replace(image["b12x"], wrong))
    r = run_gate(image)
    assert r.returncode == 1
    assert "b12x" in r.stderr and wrong in r.stderr and image["b12x"] in r.stderr
    assert "vllm:" not in r.stderr, "only the broken pin should be reported"


def test_wrong_dist_version_fails_and_names_it(image):
    shutil.rmtree(next(image["site"].glob("flashinfer_python-*.dist-info")))
    _dist(image["site"], "flashinfer-python", "0.6.18")
    r = run_gate(image)
    assert r.returncode == 1
    assert "flashinfer-python" in r.stderr and "0.6.18.post1" in r.stderr


def test_modified_installed_file_fails(image):
    (image["site"] / "b12x" / "attention" / "mla.py").write_text("K = 3\n")
    r = run_gate(image)
    assert r.returncode == 1 and "b12x/attention/mla.py" in r.stderr


def test_missing_installed_file_fails(image):
    (image["site"] / "vllm" / "models" / "deepseek_v4_1" / "attention.py").unlink()
    r = run_gate(image)
    assert r.returncode == 1 and "vllm/models/deepseek_v4_1/attention.py" in r.stderr


def test_base_vllm_left_installed_fails(image):
    _dist(image["site"], "vllm", "0.20.0")
    r = run_gate(image)
    assert r.returncode == 1 and "vllm" in r.stderr and "2 installed" in r.stderr


def test_vllm_version_without_the_fork_commit_fails(image):
    shutil.rmtree(next(image["site"].glob("vllm-*.dist-info")))
    _dist(image["site"], "vllm", "0.20.0+cu130")
    r = run_gate(image)
    assert r.returncode == 1 and "0.20.0+cu130" in r.stderr


def test_dirty_checkout_fails(image):
    (image["src"] / "vllm" / "requirements" / "cuda.txt").write_text("")
    r = run_gate(image)
    assert r.returncode == 1 and "requirements/cuda.txt" in r.stderr


def test_build_arg_that_disagrees_with_the_pin_file_fails(image):
    r = run_gate(image, B12X_COMMIT="1" * 40)
    assert r.returncode == 1 and "B12X_COMMIT" in r.stderr
    assert run_gate(image, B12X_COMMIT=image["b12x"], VLLM_FORK_COMMIT=image["vllm"]).returncode == 0


def test_wheel_mode_must_carry_the_pinned_wheel_hash(image):
    good, bad = "c" * 64, "d" * 64
    (image["root"] / "vllm-install.txt").write_text(f"wheel {good} https://example.invalid/vllm.whl\n")
    ok = run_gate(image, VLLM_WHEEL_URL="https://example.invalid/vllm.whl", VLLM_WHEEL_SHA256=good)
    assert ok.returncode == 0, ok.stderr
    r = run_gate(image, VLLM_WHEEL_URL="https://example.invalid/vllm.whl", VLLM_WHEEL_SHA256=bad)
    assert r.returncode == 1 and bad in r.stderr


def test_wheel_arg_set_but_source_build_recorded_fails(image):
    r = run_gate(image, VLLM_WHEEL_URL="https://example.invalid/vllm.whl", VLLM_WHEEL_SHA256="c" * 64)
    assert r.returncode == 1 and "source" in r.stderr


def test_missing_install_record_fails(image):
    (image["root"] / "vllm-install.txt").unlink()
    r = run_gate(image)
    assert r.returncode == 1 and "vllm-install.txt" in r.stderr


def test_vendored_file_md5_checked_and_mismatch_named(image):
    vend = image["root"] / "patches" / "entry.sh"
    vend.write_text("#!/bin/sh\n")
    md5 = hashlib.md5(vend.read_bytes()).hexdigest()[:8]
    (image["root"] / "patches" / "MD5SUMS.txt").write_text(f"{md5} entry.sh\n")
    assert run_gate(image).returncode == 0
    vend.write_text("#!/bin/sh\necho changed\n")
    r = run_gate(image)
    assert r.returncode == 1 and "entry.sh" in r.stderr and md5 in r.stderr


def test_malformed_pin_line_fails(image):
    f = image["root"] / "patches" / "UPSTREAM_COMMITS.txt"
    f.write_text(f.read_text() + "tilelang 0.1.12\n")
    r = run_gate(image)
    assert r.returncode == 1 and "tilelang" in r.stderr


def test_empty_commit_pin_file_fails(image):
    (image["root"] / "patches" / "UPSTREAM_COMMITS.txt").write_text("# name repo commit\n")
    r = run_gate(image)
    assert r.returncode == 1
    for name in ("vllm", "b12x", "nccl"):
        assert f"{name}: no entry in UPSTREAM_COMMITS.txt" in r.stderr, r.stderr


def test_commit_pin_file_missing_one_entry_fails(image):
    f = image["root"] / "patches" / "UPSTREAM_COMMITS.txt"
    f.write_text("\n".join(l for l in f.read_text().splitlines() if not l.startswith("b12x ")) + "\n")
    r = run_gate(image)
    assert r.returncode == 1 and "b12x: no entry in UPSTREAM_COMMITS.txt" in r.stderr and "vllm:" not in r.stderr


def test_empty_dist_pin_file_fails(image):
    (image["root"] / "patches" / "PINNED_DISTS.txt").write_text("# dist version [sha256 url]\n")
    r = run_gate(image)
    assert r.returncode == 1
    for dist in ("torch", "flashinfer-python", "b12x"):
        assert f"{dist}: no entry in PINNED_DISTS.txt" in r.stderr, r.stderr


def test_lookalike_origin_fails(image):
    _git(image["src"] / "b12x", "remote", "set-url", "origin", "https://github.com/local-inference-lab/b12x-evil.git")
    r = run_gate(image)
    assert r.returncode == 1 and "b12x-evil" in r.stderr


def test_origin_match_is_exact_across_url_forms():
    spec = importlib.util.spec_from_file_location("pins_gate", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    repo = "local-inference-lab/vllm"
    for ok in ("https://github.com/local-inference-lab/vllm.git", "https://github.com/local-inference-lab/vllm",
               "git@github.com:local-inference-lab/vllm.git", "https://github.com/Local-Inference-Lab/vllm/"):
        assert gate.origin_matches(ok, repo), ok
    for bad in ("https://github.com/local-inference-lab/vllm-evil.git", "https://github.com/x/local-inference-lab/vllm",
                "https://gitlab.com/local-inference-lab/vllm.git", "https://github.com/local-inference-lab/vllm.git.evil"):
        assert not gate.origin_matches(bad, repo), bad


# ---- the real pin files in this directory ----

def _pins() -> dict[str, tuple[str, str]]:
    out = {}
    for line in (IMAGE_DIR / "patches" / "UPSTREAM_COMMITS.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, repo, commit = line.split()
            out[name] = (repo, commit)
    return out


def test_dockerfile_arg_defaults_match_upstream_commits():
    if not (IMAGE_DIR / "Dockerfile").is_file():
        pytest.skip("source-tree check; the image carries patches/ but not the Dockerfile")
    dockerfile = (IMAGE_DIR / "Dockerfile").read_text()
    args = dict(re.findall(r"^ARG (VLLM_FORK_COMMIT|B12X_COMMIT|NCCL_COMMIT)=([0-9a-f]{40})$", dockerfile, re.M))
    pins = _pins()
    assert args == {"VLLM_FORK_COMMIT": pins["vllm"][1], "B12X_COMMIT": pins["b12x"][1], "NCCL_COMMIT": pins["nccl"][1]}
    assert pins["vllm"][0] == "local-inference-lab/vllm" and pins["b12x"][0] == "local-inference-lab/b12x"
    assert pins["nccl"][0] == "NVIDIA/nccl"


def test_pinned_wheels_are_hashed_https_urls():
    rows = [l.split() for l in (IMAGE_DIR / "patches" / "PINNED_DISTS.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    wheels = [r for r in rows if len(r) == 4]
    assert {r[0] for r in wheels} == {"flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"}
    for dist, version, sha, url in wheels:
        assert re.fullmatch(r"[0-9a-f]{64}", sha), dist
        assert url.startswith("https://") and url.endswith(".whl"), dist
        assert version.split("+")[0] in url, (dist, version, url)
    assert all(len(r) in (2, 4) for r in rows), rows


# ---- carried patches (patches/<name>/*.patch + APPLIED.md), for b12x and for the vLLM fork alike ----

APPLY = IMAGE_DIR / "build" / "apply_carried_patches.sh"
# Per checkout: a tracked file the patch modifies, the file it creates, and the created file's body.
CARRY = {
    "b12x": ("b12x/attention/mla.py", "b12x/attention/routes.py", "ROUTES = 'switchless ring routes'\n"),
    "vllm": ("vllm/models/deepseek_v4_1/attention.py", "vllm/models/deepseek_v4_1/trellis.py",
             "CODEBOOKS = 'switchless ring routes'\n"),
}
CARRIED = pytest.mark.parametrize("name", sorted(CARRY))


def _carry_patch(image, name: str = "b12x", *, apply_extra: str | None = None, record_sha: str | None = None,
                 marker: str = "switchless ring routes", fname: str = "0001-routes.patch",
                 edit: str = "K = 3\n", new: tuple[str, str] | None = None, modify: str | None = None) -> Path:
    """Make a patch on a scratch copy of the <name> checkout, record it, apply it to the real checkout with the
    Dockerfile's script, and install the patched files."""
    modified, created, body = CARRY[name]
    if new is not None:
        created, body = new
    if modify is not None:
        modified = modify
    co = image["src"] / name
    scratch = co.parent / f"{name}-scratch"
    shutil.copytree(co, scratch)
    # Earlier carried patches are already applied to the checkout: this patch is made on top of them, in order.
    _git(scratch, "add", "-A")
    _git(scratch, "commit", "-q", "--allow-empty", "-m", "pin plus earlier carried patches")
    (scratch / modified).write_text(edit)
    (scratch / created).write_text(body)
    _git(scratch, "add", "-A")
    patch = _git(scratch, "diff", "--cached", "--binary") + "\n"
    shutil.rmtree(scratch)
    pdir = image["root"] / "patches" / name
    pdir.mkdir(parents=True, exist_ok=True)
    p = pdir / fname
    p.write_text(patch)
    sha = record_sha or hashlib.sha256(p.read_bytes()).hexdigest()
    applied = pdir / "APPLIED.md"
    if not applied.is_file():
        applied.write_text("| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |\n"
                           "|------|----------|----------|--------|-------|------|--------|--------|\n")
    with applied.open("a") as f:
        f.write(f"| {fname} | example/{name}#1 | {'c' * 40} | t | x | exact PR diff | {marker} | {sha} |\n")
    # Apply only this patch (a scratch dir holding just it), as the Dockerfile's script does for each patch in order.
    one = image["root"] / f"one-{name}"
    one.mkdir(exist_ok=True)
    for old in one.glob("*.patch"):
        old.unlink()
    shutil.copy(p, one / fname)
    subprocess.run(["bash", str(APPLY), str(one), str(co)], check=True, capture_output=True, text=True)
    if apply_extra:
        (co / apply_extra).write_text("drift\n")
    for rel in (modified, created):
        (image["site"] / rel).write_text((co / rel).read_text())
    return p


@CARRIED
def test_carried_patch_recorded_and_applied_passes(image, name):
    _carry_patch(image, name)
    r = run_gate(image)
    assert r.returncode == 0, r.stderr


def test_carried_patches_on_both_checkouts_pass(image):
    _carry_patch(image, "b12x")
    _carry_patch(image, "vllm")
    r = run_gate(image)
    assert r.returncode == 0, r.stderr


def test_two_patches_on_one_checkout_pass_and_each_needs_its_row(image):
    _carry_patch(image, "b12x")
    _carry_patch(image, "b12x", fname="0002-mcg.patch", edit="K = 4\n", marker="mcg trellis",
                 new=("b12x/attention/independent.py", "MCG = 'mcg trellis'\n"), modify="b12x/attention/__init__.py")
    r = run_gate(image)
    assert r.returncode == 0, r.stderr
    applied = image["root"] / "patches" / "b12x" / "APPLIED.md"
    applied.write_text("".join(l for l in applied.read_text().splitlines(True) if "0002-mcg.patch" not in l))
    r = run_gate(image)
    assert r.returncode == 1 and "0002-mcg.patch" in r.stderr


@CARRIED
def test_carried_patch_survives_git_clean(image, name):
    # The image runs `git clean -fdxq` after installing b12x; `git add -N` keeps the patch's new file.
    _carry_patch(image, name)
    _git(image["src"] / name, "clean", "-fdxq")
    r = run_gate(image)
    assert r.returncode == 0, r.stderr


@CARRIED
def test_change_beyond_the_carried_patch_fails(image, name):
    _carry_patch(image, name, apply_extra=f"{name}/__init__.py")
    r = run_gate(image)
    assert r.returncode == 1 and f"{name}/__init__.py" in r.stderr


@CARRIED
def test_carried_patch_with_wrong_recorded_sha_fails(image, name):
    _carry_patch(image, name, record_sha="d" * 64)
    r = run_gate(image)
    assert r.returncode == 1 and "0001-routes.patch" in r.stderr and "sha256" in r.stderr


@CARRIED
def test_carried_patch_without_an_applied_row_fails(image, name):
    p = _carry_patch(image, name)
    (p.parent / "APPLIED.md").write_text("| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |\n")
    r = run_gate(image)
    assert r.returncode == 1 and "0001-routes.patch" in r.stderr


@CARRIED
def test_installed_new_file_from_a_carried_patch_must_match(image, name):
    _carry_patch(image, name)
    created = CARRY[name][1]
    (image["site"] / created).write_text("STALE = 1\n")
    r = run_gate(image)
    assert r.returncode == 1 and created in r.stderr


@CARRIED
def test_installed_modified_file_from_a_carried_patch_must_match(image, name):
    # The vLLM case: a wheel built from the unpatched pin installs the pin's file, not the patched one.
    _carry_patch(image, name)
    modified = CARRY[name][0]
    (image["site"] / modified).write_text(dict(VLLM_FILES, **B12X_FILES)[modified])
    r = run_gate(image)
    assert r.returncode == 1 and modified in r.stderr


@CARRIED
def test_carried_patch_marker_must_be_installed(image, name):
    _carry_patch(image, name, marker="not in any installed file")
    r = run_gate(image)
    assert r.returncode == 1 and "marker" in r.stderr


@CARRIED
def test_recorded_but_unapplied_patch_fails(image, name):
    p = _carry_patch(image, name)
    _git(image["src"] / name, "apply", "-R", str(p))
    _git(image["src"] / name, "reset", "-q")
    r = run_gate(image)
    assert r.returncode == 1 and "0001-routes.patch" in r.stderr


@CARRIED
def test_reversing_the_carried_patch_must_restore_the_pin(image, name):
    # A tracked file changed by the patch AND edited further: the changed-file set still matches, but reversing the
    # recorded patch no longer applies, so the checkout is not the pin plus that patch.
    _carry_patch(image, name)
    modified = CARRY[name][0]
    co = image["src"] / name
    (co / modified).write_text("K = 99\n")
    (image["site"] / modified).write_text("K = 99\n")
    r = run_gate(image)
    assert r.returncode == 1 and "reversing" in r.stderr


def test_unpatched_dirty_vllm_checkout_still_fails(image):
    # Without patches/vllm, any change to the fork checkout is drift, as before.
    (image["src"] / "vllm" / "vllm" / "__init__.py").write_text("V = 2\n")
    r = run_gate(image)
    assert r.returncode == 1 and "vllm/__init__.py" in r.stderr


# ---- a third-party series (patches/<name>/<dir>/series + APPLIED.md), applied after the checkout's own patches ----

SERIES = "spark-ds41f"


def _series_steps(name: str) -> list[tuple[str, dict[str, str], str]]:
    """Two stacked patches on one file, numbered like ours (0001), so only the listed order applies."""
    modified = CARRY[name][0]
    return [
        ("0001-first.patch", {modified: "K = 'stack one'\n"}, "stack one"),
        ("0002-second.patch", {modified: "K = 'stack one'\nJ = 'stack two'\n"}, "stack two"),
    ]


def _carry_series(image, name: str = "vllm", *, listed: list[str] | None = None, apply: bool = True) -> Path:
    """Make a series on a scratch copy of the <name> checkout (on top of whatever is already applied), record it in
    patches/<name>/spark-ds41f/, apply it with the Dockerfile's script and install the patched files."""
    co = image["src"] / name
    scratch = co.parent / f"{name}-series-scratch"
    shutil.copytree(co, scratch)
    _git(scratch, "add", "-A")
    _git(scratch, "commit", "-q", "--allow-empty", "-m", "pin plus earlier carried patches")
    sdir = image["root"] / "patches" / name / SERIES
    sdir.mkdir(parents=True, exist_ok=True)
    rows = ["| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |",
            "|------|----------|----------|--------|-------|------|--------|--------|"]
    touched: list[str] = []
    steps = _series_steps(name)
    for fname, files, marker in steps:
        for rel, body in files.items():
            (scratch / rel).write_text(body)
            touched.append(rel)
        _git(scratch, "add", "-A")
        (sdir / fname).write_text(_git(scratch, "diff", "--cached", "--binary") + "\n")
        _git(scratch, "commit", "-q", "-m", fname)
        sha = hashlib.sha256((sdir / fname).read_bytes()).hexdigest()
        rows.append(f"| {fname} | example/series {fname} | {'c' * 40} | t | x | verbatim | {marker} | {sha} |")
    shutil.rmtree(scratch)
    (sdir / "APPLIED.md").write_text("\n".join(rows) + "\n")
    names = listed if listed is not None else [f for f, _, _ in steps]
    (sdir / "series").write_text("# a comment, then the order\n" + "".join(f"{n}  # why\n" for n in names))
    if apply:
        # The checkout's own patches are applied already; a scratch patch dir holding only the series applies it.
        only = image["root"] / f"only-series-{name}"
        if only.exists():
            shutil.rmtree(only)
        shutil.copytree(sdir, only / SERIES)
        subprocess.run(["bash", str(APPLY), str(only), str(co)], check=True, capture_output=True, text=True)
        for rel in dict.fromkeys(touched):
            (image["site"] / rel).write_text((co / rel).read_text())
    return sdir


@CARRIED
def test_series_after_the_checkouts_own_patch_passes(image, name):
    _carry_patch(image, name)
    _carry_series(image, name)
    r = run_gate(image)
    assert r.returncode == 0, r.stderr


def test_series_alone_on_a_checkout_passes(image):
    _carry_series(image, "vllm")
    r = run_gate(image)
    assert r.returncode == 0, r.stderr


def test_the_apply_script_applies_own_patches_then_the_series_in_listed_order(image, tmp_path):
    # The whole patch directory in one call, as the Dockerfile runs it: 0001-routes.patch first (filename order), then
    # the series' 0001 and 0002 in listed order, although their names sort between and after ours.
    _carry_patch(image, "vllm")
    sdir = _carry_series(image, "vllm")
    pdir = sdir.parent
    fresh = tmp_path / "fresh"
    _checkout(fresh, "vllm", "local-inference-lab/vllm", VLLM_FILES)
    r = subprocess.run(["bash", str(APPLY), str(pdir), str(fresh / "vllm")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    applied = [l.split()[1] for l in r.stdout.splitlines() if l.startswith("applied ")]
    assert applied == ["0001-routes.patch", f"{SERIES}/0001-first.patch", f"{SERIES}/0002-second.patch"]
    for rel in ("vllm/models/deepseek_v4_1/attention.py", "vllm/models/deepseek_v4_1/trellis.py"):
        assert (fresh / "vllm" / rel).read_text() == (image["src"] / "vllm" / rel).read_text()


def test_series_out_of_listed_order_does_not_apply(image):
    with pytest.raises(subprocess.CalledProcessError):
        _carry_series(image, "vllm", listed=["0002-second.patch", "0001-first.patch"])


def test_series_with_an_unlisted_patch_fails_to_apply_and_fails_the_gate(image):
    sdir = _carry_series(image, "vllm")
    shutil.copy(sdir / "0002-second.patch", sdir / "0003-stray.patch")
    r = run_gate(image)
    assert r.returncode == 1 and f"{SERIES}/0003-stray.patch is not listed" in r.stderr
    fresh = image["root"] / "fresh-src"
    _checkout(fresh, "vllm", "local-inference-lab/vllm", VLLM_FILES)
    r = subprocess.run(["bash", str(APPLY), str(sdir.parent), str(fresh / "vllm")], capture_output=True, text=True)
    assert r.returncode != 0 and "0003-stray.patch is not listed" in r.stderr


def test_series_listing_a_missing_file_fails(image):
    sdir = _carry_series(image, "vllm")
    (sdir / "series").write_text((sdir / "series").read_text() + "0009-gone.patch\n")
    r = run_gate(image)
    assert r.returncode == 1 and "lists 0009-gone.patch, which is not in" in r.stderr


def test_series_listing_a_patch_twice_fails(image):
    sdir = _carry_series(image, "vllm")
    (sdir / "series").write_text((sdir / "series").read_text() + "0001-first.patch\n")
    r = run_gate(image)
    assert r.returncode == 1 and "lists 0001-first.patch more than once" in r.stderr


def test_series_patch_needs_its_row_in_the_series_applied_md(image):
    sdir = _carry_series(image, "vllm")
    applied = sdir / "APPLIED.md"
    applied.write_text("".join(l for l in applied.read_text().splitlines(True) if "0002-second.patch" not in l))
    r = run_gate(image)
    assert r.returncode == 1 and f"{SERIES}/0002-second.patch has no row in vllm/{SERIES}/APPLIED.md" in r.stderr


def test_series_patch_with_wrong_recorded_sha_fails(image):
    sdir = _carry_series(image, "vllm")
    applied = sdir / "APPLIED.md"
    text = applied.read_text()
    sha = hashlib.sha256((sdir / "0001-first.patch").read_bytes()).hexdigest()
    applied.write_text(text.replace(sha, "d" * 64))
    r = run_gate(image)
    assert r.returncode == 1 and f"{SERIES}/0001-first.patch sha256" in r.stderr


@pytest.mark.parametrize("own_patch", [True, False])
def test_series_applied_without_its_series_file_is_drift(image, own_patch):
    if own_patch:
        _carry_patch(image, "vllm")
    sdir = _carry_series(image, "vllm")
    (sdir / "series").unlink()
    r = run_gate(image)
    assert r.returncode == 1 and CARRY["vllm"][0] in r.stderr
    # With our own patch on the same file, the stacked file no longer reverses to the pin; without one, it is plain drift.
    assert ("reversing the carried patches" in r.stderr) if own_patch else ("has modified tracked files" in r.stderr), r.stderr


def test_series_installed_file_must_carry_the_whole_stack(image):
    _carry_series(image, "vllm")
    (image["site"] / CARRY["vllm"][0]).write_text("K = 'stack one'\n")  # after the first patch only
    r = run_gate(image)
    assert r.returncode == 1 and CARRY["vllm"][0] in r.stderr and "stack two" in r.stderr


# ---- the repository's own carried patches, as committed ----

def _applied_table(pdir: Path) -> dict[str, list[str]]:
    rows = {}
    for line in (pdir / "APPLIED.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[0].endswith(".patch"):
            rows[cells[0]] = cells
    return rows


# Each checkout's patches may touch only its own tree: the Python package for vllm and b12x, NCCL's src/ for nccl.
SCOPE = {"b12x": "b12x/", "vllm": "vllm/", "nccl": "src/"}

# The next two rules are for this repository's own patches, patches/<name>/*.patch, and deliberately do not reach the
# spark-ds41f series in patches/<name>/spark-ds41f/ (they glob one level only). These are the two in-image tests the
# DSv4.1 Lane A lab layers had to deselect when they recorded the series beside our patches: the series is kept
# byte-for-byte as upstream published it, so its patches also change upstream test files (0178, b12x 0003 and others
# add or update tests/), which the scope rule forbids for ours; and it is a stack applied strictly in its listed order
# (0001 and 0002 both change engram.py, and several change files ours change too), which the no-overlap rule forbids
# for ours so that any one of them applies alone. The series has its own checks below, and the pins gate proves the
# checkout is the pin plus exactly our patches and then the series, in that order.


@pytest.mark.parametrize("name", sorted(SCOPE))
def test_committed_patches_are_recorded_scoped_and_marked(name):
    pdir = IMAGE_DIR / "patches" / name
    patches = sorted(pdir.glob("*.patch"))
    assert patches, f"patches/{name} carries no patch"
    rows = _applied_table(pdir)
    assert sorted(rows) == [p.name for p in patches], "every patch has exactly one APPLIED.md row and vice versa"
    for p in patches:
        _, upstream, head, _, _, _, marker, sha = rows[p.name]
        # "none" only for a patch authored here and not proposed upstream yet, and the Upstream cell must say so.
        assert re.fullmatch(r"[0-9a-f]{40}", head) or (head == "none" and "carried here" in upstream), (p.name, head)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, p.name
        text = p.read_text()
        files = re.findall(r"^diff --git a/(\S+) b/\S+$", text, re.M)
        assert files and all(f.startswith(SCOPE[name]) for f in files), (p.name, files)
        added = "".join(l[1:] for l in text.splitlines(True) if l.startswith("+") and not l.startswith("+++"))
        assert marker in added, f"{p.name}: marker {marker!r} is not in a line the patch adds"


def test_committed_patch_files_do_not_overlap():
    seen: dict[str, str] = {}
    for p in sorted((IMAGE_DIR / "patches").glob("*/*.patch")):
        for f in re.findall(r"^diff --git a/(\S+) b/\S+$", p.read_text(), re.M):
            assert f not in seen, f"{f} is changed by both {seen[f]} and {p.name}"
            seen[f] = p.name


# ---- the spark-ds41f series, as committed ----

DS41F_COMMIT = "a2a3abae7b"
# The DSv4.1 Lane A order: lab layer b1 (vLLM 0024, 0178; b12x 0003-0005), then b2 (13 vLLM patches), then b3 (6 vLLM
# patches, sequence-parallel prefill). A reorder or an added or dropped patch has to change this list too.
DS41F_ORDER = {
    "vllm": ["0024", "0178",
             "0001", "0002", "0005", "0007", "0010", "0012", "0013", "0019", "0020", "0039", "0041", "0042", "0144",
             "0014", "0015", "0016", "0017", "0018", "0025"],
    "b12x": ["0003", "0004", "0005"],
}
# A series patch may change its package tree and upstream's own tests, nothing else.
DS41F_SCOPE = {"vllm": ("vllm/", "tests/"), "b12x": ("b12x/", "tests/")}


def _series_file(sdir: Path) -> list[str]:
    return [n for n in (l.split("#", 1)[0].strip() for l in (sdir / "series").read_text().splitlines()) if n]


@pytest.mark.parametrize("name", sorted(DS41F_ORDER))
def test_committed_series_is_listed_recorded_and_marked(name):
    sdir = IMAGE_DIR / "patches" / name / "spark-ds41f"
    listed = _series_file(sdir)
    assert [n[:4] for n in listed] == DS41F_ORDER[name], listed
    assert sorted(listed) == sorted(p.name for p in sdir.glob("*.patch")), "series lists every patch once, and only those"
    rows = _applied_table(sdir)
    assert sorted(rows) == sorted(listed), "every series patch has exactly one APPLIED.md row and vice versa"
    for n in listed:
        p = sdir / n
        _, upstream, head, author, _, form, marker, sha = rows[n]
        text = p.read_text(encoding="utf-8")  # 0178 carries non-ASCII test fixtures
        assert text.startswith(f"From {head} "), (n, "Head SHA is the patch's own From line")
        assert upstream.startswith("christopherowen/spark-ds41f ") and upstream.endswith(f" at {DS41F_COMMIT}"), n
        assert author == "Christopher Owen" and "Christopher Owen" in text.splitlines()[1], n
        assert form.startswith("verbatim upstream patch file"), n
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, n
        files = re.findall(r"^diff --git a/(\S+) b/\S+$", text, re.M)
        assert files and all(f.startswith(DS41F_SCOPE[name]) for f in files), (n, files)
        assert any(f.startswith(name + "/") for f in files), (n, "changes nothing the image installs")
        added = "".join(l[1:] for l in text.splitlines(True) if l.startswith("+") and not l.startswith("+++"))
        assert marker in added, f"{n}: marker {marker!r} is not in a line the patch adds"


# ---- our own patches stacked on the series, as committed ----

# patches/vllm/stacked/: this repository's patches that change code a spark-ds41f patch adds, so they apply after that
# series. Series directories apply in directory-name order, so the name has to sort after spark-ds41f.
STACKED = "stacked"


def test_committed_stacked_series_is_ours_listed_recorded_scoped_and_marked():
    pdir = IMAGE_DIR / "patches" / "vllm"
    assert sorted(d.name for d in pdir.iterdir() if (d / "series").is_file()) == ["spark-ds41f", STACKED]
    sdir = pdir / STACKED
    listed = _series_file(sdir)
    assert listed and sorted(listed) == sorted(p.name for p in sdir.glob("*.patch")), listed
    rows = _applied_table(sdir)
    assert sorted(rows) == sorted(listed), "every stacked patch has exactly one APPLIED.md row and vice versa"
    for n in listed:
        p = sdir / n
        _, upstream, head, author, _, _, marker, sha = rows[n]
        assert head == "none" and "carried here" in upstream and author == "Defilan", n
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, n
        text = p.read_text()
        files = re.findall(r"^diff --git a/(\S+) b/\S+$", text, re.M)
        assert files and all(f.startswith("vllm/") for f in files), (n, files)
        added = "".join(l[1:] for l in text.splitlines(True) if l.startswith("+") and not l.startswith("+++"))
        assert marker in added, f"{n}: marker {marker!r} is not in a line the patch adds"


def test_md5sums_list_exactly_the_vendored_series_files():
    rows = [l.split() for l in (IMAGE_DIR / "patches" / "MD5SUMS.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    vendored = sorted(p.relative_to(IMAGE_DIR / "patches").as_posix()
                      for p in (IMAGE_DIR / "patches").glob("*/spark-ds41f/*.patch"))
    assert len(vendored) == sum(len(v) for v in DS41F_ORDER.values()) == 24
    assert sorted(r[1] for r in rows) == vendored
    for md5, rel in rows:
        assert hashlib.md5((IMAGE_DIR / "patches" / rel).read_bytes()).hexdigest() == md5, rel


def test_spark_ds41f_license_and_notice_travel():
    # Both in the source tree and in the image (/opt/llmkube), the license text sits beside NOTICE.
    lic = (IMAGE_DIR / "LICENSE.spark-ds41f-Apache-2.0").read_text()
    assert "Apache License" in lic and "Version 2.0, January 2004" in lic
    notice = (IMAGE_DIR / "NOTICE").read_text()
    assert "christopherowen/spark-ds41f" in notice and "a2a3abae7b395f98ef64dcc77beef1584d756cd6" in notice
    for name in DS41F_ORDER:
        for n in _series_file(IMAGE_DIR / "patches" / name / "spark-ds41f"):
            assert n in notice, f"NOTICE does not list {name}/spark-ds41f/{n}"


@pytest.mark.skipif(not (IMAGE_DIR / "Dockerfile").is_file(), reason="the image ships no Dockerfile")
def test_dockerfile_commit_args_match_the_pins():
    pins = {r.split()[0]: r.split()[2] for r in (IMAGE_DIR / "patches" / "UPSTREAM_COMMITS.txt").read_text().splitlines()
            if r.strip() and not r.startswith("#")}
    df = (IMAGE_DIR / "Dockerfile").read_text()
    assert f"ARG B12X_COMMIT={pins['b12x']}\n" in df
    assert f"ARG VLLM_FORK_COMMIT={pins['vllm']}\n" in df
    assert f"ARG NCCL_COMMIT={pins['nccl']}\n" in df


# ---- the native NCCL pin: checkout + carried patches + build record + installed library ----

FENCE = "  // Order the nreqs load after the idx load above\n"


def _carry_nccl_patch(image, *, marker: str = "Order the nreqs load", apply: bool = True) -> str:
    co = image["src"] / "nccl"
    scratch = co.parent / "nccl-scratch"
    shutil.copytree(co, scratch)
    p2p = scratch / "src/transport/net_ib/p2p.cc"
    p2p.write_text(p2p.read_text().replace("  return 0;\n", FENCE + "  return 0;\n"))
    patch = _git(scratch, "diff") + "\n"
    shutil.rmtree(scratch)
    pdir = image["root"] / "patches" / "nccl"
    pdir.mkdir(parents=True, exist_ok=True)
    p = pdir / "0001-fence.patch"
    p.write_text(patch)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    (pdir / "APPLIED.md").write_text(
        "| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |\n"
        "|------|----------|----------|--------|-------|------|--------|--------|\n"
        f"| 0001-fence.patch | NVIDIA/nccl#1 | {'c' * 40} | t | x | exact PR diff | {marker} | {sha} |\n")
    if apply:
        subprocess.run(["bash", str(APPLY), str(pdir), str(co)], check=True, capture_output=True, text=True)
    return sha


def test_nccl_carried_patch_with_matching_record_passes(image):
    sha = _carry_nccl_patch(image)
    _nccl_record(image, patches=sha)
    r = run_gate(image)
    assert r.returncode == 0, r.stderr


def test_nccl_installed_library_must_be_the_recorded_build(image):
    (image["site"] / NCCL_LIB).write_bytes(b"\x7fELF the wheel's unfenced libnccl")
    r = run_gate(image)
    assert r.returncode == 1 and "libnccl.so.2" in r.stderr and "nccl-build stage recorded" in r.stderr


def test_nccl_missing_library_fails(image):
    (image["site"] / NCCL_LIB).unlink()
    r = run_gate(image)
    assert r.returncode == 1 and "is not installed" in r.stderr


def test_nccl_missing_build_record_fails(image):
    (image["root"] / "nccl-build.txt").unlink()
    r = run_gate(image)
    assert r.returncode == 1 and "nccl-build.txt" in r.stderr


def test_nccl_record_of_another_commit_fails(image):
    _nccl_record(image, commit="e" * 40)
    r = run_gate(image)
    assert r.returncode == 1 and "e" * 40 in r.stderr and image["nccl"] in r.stderr


def test_nccl_record_must_name_exactly_the_carried_patches(image):
    sha = _carry_nccl_patch(image)
    _nccl_record(image, patches="none")  # a library built without the fence
    r = run_gate(image)
    assert r.returncode == 1 and "records patches none" in r.stderr and sha in r.stderr


def test_nccl_patch_recorded_but_not_applied_fails(image):
    sha = _carry_nccl_patch(image, apply=False)
    _nccl_record(image, patches=sha)
    r = run_gate(image)
    assert r.returncode == 1 and "src/transport/net_ib/p2p.cc" in r.stderr


def test_nccl_patch_marker_must_be_in_the_patched_source(image):
    sha = _carry_nccl_patch(image, marker="not in the source")
    _nccl_record(image, patches=sha)
    r = run_gate(image)
    assert r.returncode == 1 and "marker" in r.stderr


def test_nccl_build_arg_that_disagrees_fails(image):
    r = run_gate(image, NCCL_COMMIT="1" * 40)
    assert r.returncode == 1 and "NCCL_COMMIT" in r.stderr
