"""SCAFFOLD ONLY. See server/__init__.py.

Endpoints here are deliberately minimal. `/health` reports that the process is
up, not that a model is loaded, and says so in the payload, so a passing
readiness probe cannot be mistaken for a servable model. `/v1/models` returns an
empty list for the same reason.

The real server replaces this module: it loads the checkpoint, applies the chat
template, drives the ExLlamaV3 generator (including DSpark / MTP drafting), and
serves `/v1/completions` and `/v1/chat/completions`. Measurement parity with the
community recipe (temperature 0, thinking off, the same drafting knobs) is a
requirement on that implementation, not a nicety, or its throughput numbers are
not comparable to the published ones.
"""

from fastapi import FastAPI

app = FastAPI(title="llmkube-exllamav3", version="scaffold")


@app.get("/health")
def health() -> dict:
    return {"status": "scaffold", "model_loaded": False, "server_ready": True}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": []}
