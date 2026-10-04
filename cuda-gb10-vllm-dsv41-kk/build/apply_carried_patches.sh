#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# apply_carried_patches.sh <patch-dir> <checkout>: apply every <patch-dir>/*.patch to <checkout> in filename order.
# Each patch must apply cleanly (git apply --check first). Its new files are then recorded with `git add -N` (intent to
# add), so a later `git clean` keeps them and build/pins_gate.py sees them as part of the checkout. Not
# `git apply --intent-to-add`: git 2.43 (the base's git) drops every other index entry when it is used without --index.
# A missing or empty patch directory applies nothing.
set -eu
pdir="$1"
co="$2"
for p in "$pdir"/*.patch; do
  [ -e "$p" ] || continue
  git -C "$co" apply --check "$p"
  git -C "$co" apply "$p"
  git -C "$co" apply --summary "$p" | awk '/ create mode /{print $NF}' | while read -r f; do
    git -C "$co" add -N -- "$f"
  done
  echo "applied $(basename "$p") to $co"
done
