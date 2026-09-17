# Server

OpenAI-compatible HTTP server for the GB10 native ExLlamaV3 runtime. Three
modules, each with one job:

| File | Owns |
|---|---|
| `app.py` | The FastAPI surface and request/response shapes |
| `engine.py` | The loaded model, and the lock that serializes generation |
| `chat.py` | Rendering the checkpoint's own chat template |

## Endpoints

- `GET /health` liveness: 200 whenever the process is up. Reports load state and
  any error, so a failed load does not look like a dead process.
- `GET /ready` readiness: 200 only when a model is loaded, 503 otherwise. Point a
  Kubernetes readiness probe here; TCP being open is not readiness.
- `GET /v1/models`
- `POST /v1/completions` and `POST /v1/chat/completions`, streaming and
  non-streaming.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `EXL3_MODEL_DIR` | `/models` | Checkpoint directory |
| `EXL3_DRAFT_MODEL_DIR` | unset | Draft / MTP drafter, passed to `Generator(draft_model=...)` |
| `EXL3_DEVICE` | `cuda:0` | Device for `model.load` |
| `EXL3_CACHE_TOKENS` | `32768` | KV cache tokens |
| `EXL3_MAX_BATCH` | `1` | Cache batch size |
| `EXL3_THINKING` | `0` | Template kwarg `enable_thinking` |
| `EXL3_REQUIRE_MODEL` | `0` | `1` makes a failed load fatal instead of reported |

The `EXL3_ATS_*` and `EXL3_DSPARK_*` knobs the loader reads are set by the
InferenceService and passed straight through; the server does not touch them.

## Measurement parity

The community recipe publishes its decode and prefill numbers under a specific
protocol: temperature 0, thinking off, and its drafting settings. The defaults
here match the first two, and `chat_template_kwargs` (vLLM-style passthrough) is
exposed so a client can set the rest without the server inventing a flag per
model. Numbers produced under different settings are not comparable to the
recipe's, and an incomparable number is worse than none.

## What this deliberately does not do

Multi-model routing, LoRA, authentication, `config.yml` compatibility, and
concurrent streams. `Generator` is serialized behind one lock, which is the
honest shape for a single-stream POC: queueing is a scheduling decision, not
something to hide behind a thread pool.

## Why this is ours

TabbyAPI, the usual ExLlamaV3 server, is AGPL-3.0, and this repository is
Apache-2.0 with a "none is AGPL" policy that CI enforces. ExLlamaV3 itself is
MIT. See `docs/exllamav3-gb10-runtime.md`.
