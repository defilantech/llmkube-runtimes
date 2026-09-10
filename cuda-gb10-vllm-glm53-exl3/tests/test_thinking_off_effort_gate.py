#!/usr/bin/env python3
"""The reasoning-effort hint must not appear when thinking is disabled.

WHY THIS TEST EXISTS

The vendored template defaults `effective_reasoning_effort` to 'max' whenever
the caller does not pass 'low' or 'high', and the line that emits the
`<|system|>Reasoning Effort: ...` hint originally did not consult
`thinking_enabled`. So a request with `enable_thinking=false` got BOTH a
max-effort instruction AND a closed thinking block (`<think></think>`) in the
generation prompt. The model has nowhere to put reasoning and puts it in the
answer.

That is bad on its own and worse for measurement: every published decode number
for this model is quoted with thinking off, so a benchmark run against the
unfixed template is measuring a different workload than the one being compared
against.

This test asserts on the RENDERED prompt rather than on the template source, so
it keeps working if the implementation of the gate changes.

Run: python3 tests/test_thinking_off_effort_gate.py
"""

from __future__ import annotations

import pathlib
import sys

try:
    from jinja2 import Environment
except ImportError:  # pragma: no cover
    sys.exit("jinja2 is required: pip install jinja2")

_DEFAULT = pathlib.Path(__file__).resolve().parent.parent / "files" / "chat_template.jinja"
# In the built image the template and this test both sit in /opt/glm53, so the
# repo-relative path does not exist. Fall back to a sibling before failing.
_SIBLING = pathlib.Path(__file__).resolve().parent / "chat_template.jinja"
TEMPLATE = _DEFAULT if _DEFAULT.is_file() else _SIBLING
HINT = "Reasoning Effort"

MESSAGES = [{"role": "user", "content": "hello"}]


def render(**kwargs: object) -> str:
    # loopcontrols is required: the template uses {% break %}, which is not
    # core Jinja. vLLM enables this extension when it loads a chat template,
    # so rendering without it fails on syntax and tells you nothing about
    # the gate under test.
    env = Environment(trim_blocks=False, lstrip_blocks=False,
                      extensions=["jinja2.ext.loopcontrols"])
    tpl = env.from_string(TEMPLATE.read_text())
    return tpl.render(messages=MESSAGES, add_generation_prompt=True, **kwargs)


def main() -> int:
    failures: list[str] = []

    # 1. Thinking OFF: no effort hint, and the thinking block is closed.
    off = render(enable_thinking=False)
    if HINT in off:
        failures.append(
            "enable_thinking=False still emits the reasoning-effort hint. "
            "The model will be told to reason at max effort with no thinking "
            "block and will reason inside its answer."
        )
    if "<think></think>" not in off:
        failures.append(
            "enable_thinking=False did not close the thinking block; the "
            "template's generation prompt changed and this test's premise no "
            "longer holds."
        )

    # 2. Thinking ON: the hint must still be there. A gate that suppresses it
    #    always would pass check 1 while silently removing the feature, so the
    #    positive case is what stops this being a one-sided assertion.
    on = render(enable_thinking=True)
    if HINT not in on:
        failures.append(
            "enable_thinking=True lost the reasoning-effort hint; the gate is "
            "too aggressive and thinking-on serves without its effort setting."
        )

    # 3. Default (neither flag): the template treats thinking as ON, so the
    #    hint belongs. This pins the default so a future edit cannot flip it
    #    quietly.
    default = render()
    if HINT not in default:
        failures.append(
            "with no thinking flag the hint is absent, but the template "
            "defaults thinking_enabled to true; default behaviour changed."
        )

    # 4. An explicit effort must survive with thinking on, and must not
    #    reappear with thinking off.
    if "High" not in render(enable_thinking=True, reasoning_effort="high"):
        failures.append("explicit reasoning_effort=high is not rendered with thinking on.")
    if HINT in render(enable_thinking=False, reasoning_effort="high"):
        failures.append(
            "explicit reasoning_effort=high leaks the hint even with thinking "
            "off; the gate must win over an explicit effort."
        )

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS: reasoning-effort hint is gated on thinking_enabled")
    print("  thinking off -> no hint, thinking block closed")
    print("  thinking on / default / explicit effort -> hint present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
