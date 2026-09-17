"""OpenAI-compatible HTTP server for the GB10 native ExLlamaV3 runtime.

`app.py` is the FastAPI surface, `engine.py` owns the loaded model and the lock
that serializes generation, `chat.py` renders the checkpoint's own chat template.

This is our server rather than TabbyAPI because TabbyAPI is AGPL-3.0 and this
repository is Apache-2.0 with a "none is AGPL" policy that CI enforces;
ExLlamaV3 itself is MIT. See `docs/exllamav3-gb10-runtime.md`.

Measurement parity with the community recipe is a requirement, not a nicety:
temperature 0 and thinking off are the defaults here because those are the
settings the recipe's published numbers were produced under. A server with
different defaults produces incomparable numbers, and an incomparable number is
worse than none.
"""
