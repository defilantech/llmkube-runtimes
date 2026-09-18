"""Regression test for the OpenAI usage object, with no GPU and no model.

The count is the part that broke silently once already: `tokenizer.encode` returns
a (batch, length) tensor, so counting `len()` of it reports 1 for every string.
This drives the endpoint functions with a stubbed engine, so it asserts the
wiring rather than the model. Run it in the built image:

    docker run --rm --workdir /app --entrypoint python3 <image> -m server.test_usage
"""
from __future__ import annotations

import sys

from server import app as appmod


class StubEngine:
    """Stands in for the loaded engine: no model, no GPU, a countable tokenizer."""

    state = "ready"
    model_name = "stub"
    model_dir = "/nonexistent"
    draft_model_dir = ""
    error = ""

    def snapshot(self) -> dict:
        return {
            "server_ready": True,
            "status": "ready",
            "model_loaded": True,
            "model": self.model_name,
            "model_dir": self.model_dir,
            "draft_model_dir": "",
            "error": "",
        }

    @property
    def ready(self) -> bool:
        return True

    # One "token" per whitespace-separated word: enough to prove the wiring, and
    # deliberately not 1, so a len()-of-batch bug shows up as a mismatch.
    def count_tokens(self, text: str, add_bos: bool = False) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        return "one two three four five"


def check(label: str, condition: bool) -> bool:
    print(f"{'ok  ' if condition else 'FAIL'} {label}")
    return condition


def main() -> int:
    appmod.engine = StubEngine()
    ok = True

    chat = appmod.ChatRequest(
        messages=[appmod.ChatMessage(role="user", content="hello there")],
        max_tokens=5,
    )
    body = appmod.chat_completions(chat, None)
    usage = body.get("usage") or {}
    text = body["choices"][0]["message"]["content"]

    ok &= check("chat exposes usage", bool(usage))
    ok &= check(
        f"completion_tokens equals the tokenizer count ({usage.get('completion_tokens')} vs {len(text.split())})",
        usage.get("completion_tokens") == len(text.split()),
    )
    ok &= check(
        f"prompt_tokens equals the tokenizer count ({usage.get('prompt_tokens')})",
        usage.get("prompt_tokens") == appmod.engine.count_tokens(chat.messages[0].content, add_bos=True),
    )
    ok &= check(
        "total is the sum",
        usage.get("total_tokens") == usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
    )
    ok &= check("counts are not the batch axis", usage.get("completion_tokens", 0) > 1)

    comp = appmod.completions(appmod.CompletionRequest(prompt="a b c", max_tokens=5), None)
    ok &= check("completions exposes usage", bool(comp.get("usage")))

    # The same assertions against a counter that always reports the batch axis,
    # which is the bug this test exists for. They must fail.
    appmod.engine.count_tokens = lambda text, add_bos=False: 1
    blanked = appmod.chat_completions(chat, None)["usage"]
    ok &= check(
        "control: a bs=batch-style count is caught",
        blanked.get("completion_tokens") != len(text.split()),
    )

    print("PASS: usage wiring" if ok else "FAILED: usage wiring")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
