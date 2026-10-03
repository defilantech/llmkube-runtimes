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
PIN_ENV = ("VLLM_FORK_COMMIT", "B12X_COMMIT", "VLLM_WHEEL_URL", "VLLM_WHEEL_SHA256")


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


@pytest.fixture()
def image(tmp_path):
    """A consistent synthetic image. Tests break exactly one thing and expect the gate to name it."""
    src, site, root = tmp_path / "src", tmp_path / "site", tmp_path / "opt"
    vllm_sha = _checkout(src, "vllm", "local-inference-lab/vllm", VLLM_FILES)
    b12x_sha = _checkout(src, "b12x", "local-inference-lab/b12x", B12X_FILES)
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
        f"# name repo commit\nvllm local-inference-lab/vllm {vllm_sha}\nb12x local-inference-lab/b12x {b12x_sha}\n")
    (root / "patches" / "PINNED_DISTS.txt").write_text(
        "# dist version [sha256 url]\ntorch 2.13.0+cu130\nb12x 1.3.0\n"
        f"flashinfer-python 0.6.18.post1 {'a' * 64} https://example.invalid/flashinfer_python-0.6.18.post1-py3-none-any.whl\n")
    (root / "patches" / "MD5SUMS.txt").write_text("# md5-prefix file (vendored files only; none today)\n")
    (root / "vllm-install.txt").write_text(f"source {'b' * 64} /src/vllm\n")
    return {"src": src, "site": site, "root": root, "vllm": vllm_sha, "b12x": b12x_sha}


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
    assert "vllm: no entry in UPSTREAM_COMMITS.txt" in r.stderr and "b12x: no entry in UPSTREAM_COMMITS.txt" in r.stderr


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
    args = dict(re.findall(r"^ARG (VLLM_FORK_COMMIT|B12X_COMMIT)=([0-9a-f]{40})$", dockerfile, re.M))
    pins = _pins()
    assert args == {"VLLM_FORK_COMMIT": pins["vllm"][1], "B12X_COMMIT": pins["b12x"][1]}
    assert pins["vllm"][0] == "local-inference-lab/vllm" and pins["b12x"][0] == "local-inference-lab/b12x"


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


# ---- the repository's own carried patches, as committed ----

def _applied_table(pdir: Path) -> dict[str, list[str]]:
    rows = {}
    for line in (pdir / "APPLIED.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[0].endswith(".patch"):
            rows[cells[0]] = cells
    return rows


@pytest.mark.parametrize("name", ["b12x", "vllm"])
def test_committed_patches_are_recorded_scoped_and_marked(name):
    pdir = IMAGE_DIR / "patches" / name
    patches = sorted(pdir.glob("*.patch"))
    assert patches, f"patches/{name} carries no patch"
    rows = _applied_table(pdir)
    assert sorted(rows) == [p.name for p in patches], "every patch has exactly one APPLIED.md row and vice versa"
    for p in patches:
        _, _, head, _, _, _, marker, sha = rows[p.name]
        assert re.fullmatch(r"[0-9a-f]{40}", head), (p.name, head)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, p.name
        text = p.read_text()
        files = re.findall(r"^diff --git a/(\S+) b/\S+$", text, re.M)
        assert files and all(f.startswith(name + "/") for f in files), (p.name, files)
        added = "".join(l[1:] for l in text.splitlines(True) if l.startswith("+") and not l.startswith("+++"))
        assert marker in added, f"{p.name}: marker {marker!r} is not in a line the patch adds"


def test_committed_patch_files_do_not_overlap():
    seen: dict[str, str] = {}
    for p in sorted((IMAGE_DIR / "patches").glob("*/*.patch")):
        for f in re.findall(r"^diff --git a/(\S+) b/\S+$", p.read_text(), re.M):
            assert f not in seen, f"{f} is changed by both {seen[f]} and {p.name}"
            seen[f] = p.name


@pytest.mark.skipif(not (IMAGE_DIR / "Dockerfile").is_file(), reason="the image ships no Dockerfile")
def test_dockerfile_commit_args_match_the_pins():
    pins = {r.split()[0]: r.split()[2] for r in (IMAGE_DIR / "patches" / "UPSTREAM_COMMITS.txt").read_text().splitlines()
            if r.strip() and not r.startswith("#")}
    df = (IMAGE_DIR / "Dockerfile").read_text()
    assert f"ARG B12X_COMMIT={pins['b12x']}\n" in df
    assert f"ARG VLLM_FORK_COMMIT={pins['vllm']}\n" in df
