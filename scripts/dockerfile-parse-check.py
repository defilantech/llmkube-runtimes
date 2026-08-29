#!/usr/bin/env python3
"""Catch Dockerfile parse errors without a daemon, before the expensive pull.

This exists because a real build spent a full multi-GB base pull on an arm64
runner only to die with `unknown instruction: }`. The cause was mixing shell
heredocs with backslash continuations inside one RUN: the heredoc terminator
ends the physical line, so the next line is read as a new instruction. That is
a static property of the file and should never cost a base pull to discover.

Scope is deliberately narrow: it verifies that every line the parser would treat
as an instruction actually starts with one, tracking the two things that decide
which lines those are -- backslash continuations and heredocs. It is not a
substitute for a real build.
"""
import re
import sys

INSTRUCTIONS = {
    "FROM", "RUN", "CMD", "LABEL", "MAINTAINER", "EXPOSE", "ENV", "ADD", "COPY",
    "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL",
    "HEALTHCHECK", "SHELL",
}
# RUN cmd <<EOT / <<'EOT' / <<-"EOT"; capture the delimiter and whether dashed.
HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")


def check(path):
    errors = []
    lines = open(path, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # Blank lines and comments never start an instruction.
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        head = stripped.split(None, 1)[0].upper()
        if head not in INSTRUCTIONS:
            errors.append(
                f"{path}:{i+1}: line starts a new instruction but '{head}' is not one.\n"
                f"    {raw}\n"
                f"    Usually a heredoc/continuation mix: the previous instruction ended\n"
                f"    earlier than intended. Inside RUN <<'EOT' use plain newlines and no\n"
                f"    trailing backslashes."
            )
            i += 1
            continue

        # Consume this instruction: all heredoc bodies it opens, plus any
        # backslash-continued lines.
        pending = [(m.group(3), m.group(1) == "-") for m in HEREDOC.finditer(raw)]
        cont = raw.rstrip().endswith("\\")
        i += 1

        while pending or cont:
            if pending:
                delim, dashed = pending.pop(0)
                closed = False
                while i < len(lines):
                    body = lines[i]
                    cand = body.lstrip("\t") if dashed else body
                    i += 1
                    if cand.rstrip() == delim:
                        closed = True
                        break
                if not closed:
                    errors.append(f"{path}: heredoc <<{delim} opened but never closed.")
                    return errors
                # A heredoc body ends the physical line; continuation does not
                # carry across it.
                cont = False
                continue
            if i >= len(lines):
                break
            nxt = lines[i]
            # A comment line INSIDE a continuation is stripped by the parser and
            # does not end the instruction. Treating it as a terminator was this
            # checker's own first bug: it flagged every sibling Dockerfile in the
            # repo, all of which build fine, because they comment mid-RUN.
            if nxt.strip().startswith("#"):
                i += 1
                continue
            pending = [(m.group(3), m.group(1) == "-") for m in HEREDOC.finditer(nxt)]
            cont = nxt.rstrip().endswith("\\")
            i += 1
    return errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: dockerfile-parse-check.py <Dockerfile> [...]")
    bad = []
    for p in sys.argv[1:]:
        bad += check(p)
    for e in bad:
        print(e)
    if bad:
        print(f"\nFAIL: {len(bad)} Dockerfile parse problem(s)")
        sys.exit(1)
    print(f"PASS: {len(sys.argv)-1} Dockerfile(s) parse cleanly")
