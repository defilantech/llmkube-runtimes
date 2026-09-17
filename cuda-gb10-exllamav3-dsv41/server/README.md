# Server (scaffold)

**This directory is a placeholder.** `app.py` starts, binds `EXL3_PORT` (default
5000) and answers `/health` and `/v1/models`. It does not load a model.

It exists so that the image builds and runs end to end, the LLMKube `generic`
runtime's TCP probes have something to reach, and `scripts/exllamav3-gb10-gate.sh`
has a process to start, before the real server is written.

## What the real server must do

Replaces `app.py`. Wraps the MIT ExLlamaV3 library's own API (`Config, Model,
Cache, Tokenizer, Generator, Job`), which the upstream `examples/` confirm:

- Load the checkpoint from `EXL3_MODEL_DIR` with the recipe's knobs passed
  through unchanged (`EXL3_ATS_MMAP`, `EXL3_ATS_COPY`, `EXL3_DSPARK_CONF`,
  `CHUNK`, `CTX`); load the DSpark / MTP drafter.
- Serve `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`
  (streaming and non-streaming).
- Chat templating via the MIT `examples/chat_templates.py` / `chat_util.py`
  helpers rather than a hand-rolled template engine.
- Out of scope: multi-model routing, LoRA, auth, `config.yml` compatibility.

## Measurement parity

The community recipe publishes its decode and prefill numbers under a specific
protocol: temperature 0, thinking off, DSpark drafting at the same block size and
confidence gate. The server must default to that protocol, or its numbers are not
comparable to the recipe's and the POC loses the comparison it exists to make.

## Why this is ours

TabbyAPI, the usual ExLlamaV3 server, is AGPL-3.0, and this repository is
Apache-2.0 with a "none is AGPL" policy that CI enforces. ExLlamaV3 itself is
MIT. See `docs/exllamav3-gb10-runtime.md`.
