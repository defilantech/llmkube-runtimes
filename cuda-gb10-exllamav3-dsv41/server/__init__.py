"""SCAFFOLD ONLY: placeholder HTTP server for the GB10 native ExLlamaV3 runtime.

This package exists so the image, the operator path and the Tier-1 gate can be
exercised end to end before the real server lands. `app.py` binds the port and
answers liveness/readiness and an empty `/v1/models`; it does NOT load a model
and must not be used to serve inference.

The OpenAI-compatible implementation replaces `app.py`. It is not meant to grow
out of this file: what this file proves is only that the image starts, binds and
reports, on the pinned engine, with no x86 kernel linked.
"""
