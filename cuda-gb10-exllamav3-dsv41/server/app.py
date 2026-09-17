"""OpenAI-compatible HTTP server for the GB10 native ExLlamaV3 runtime.

Endpoints are the subset an agent loop actually uses: `/v1/models`,
`/v1/completions`, `/v1/chat/completions`, with streaming. Two health surfaces,
deliberately separate:

- `/health` is liveness: 200 whenever the process is up, and it reports the load
  state and any error. A load that failed must not look like a dead process.
- `/ready` is readiness: 200 only when a model is loaded, 503 otherwise. A
  Kubernetes readiness probe points here, so a pod with an empty model directory
  stays NotReady instead of accepting traffic and answering with an error.

The model loads in a background thread while the port is already bound, so a
client can watch the transition rather than seeing connection refused for a
minute. `EXL3_REQUIRE_MODEL=1` turns a failed load into a fatal one.

Scope is one model and one stream at a time. `/health`, `/ready`, `/v1/models`
answer without the generate lock; completions serialize behind it in the engine.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .chat import ChatRenderer, ChatTemplateError
from .engine import Engine, EngineError

log = logging.getLogger("exl3.server")

engine = Engine.from_env()
_renderer: ChatRenderer | None = None
_renderer_lock = threading.Lock()


def _get_renderer() -> ChatRenderer:
    """Build the renderer once, from the checkpoint's own template."""
    global _renderer
    with _renderer_lock:
        if _renderer is None:
            thinking = os.environ.get("EXL3_THINKING", "0") == "1"
            _renderer = ChatRenderer(engine.model_dir, thinking=thinking)
        return _renderer


def _require_ready() -> None:
    if not engine.ready:
        snap = engine.snapshot()
        raise HTTPException(
            status_code=503,
            detail=f"model not ready: status={snap['status']} error={snap['error'] or 'none'}",
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Bind first, load in the background. Daemon thread: a Ctrl-C during a long
    # load should not wait for it.
    threading.Thread(target=_load_and_warm, name="exl3-load", daemon=True).start()
    yield


def _load_and_warm() -> None:
    engine.load()
    if engine.ready:
        try:
            _get_renderer()
        except ChatTemplateError as exc:
            # Completions still work without a template; chat will report it.
            log.warning("chat template unavailable: %s", exc)


app = FastAPI(title="llmkube-exllamav3", lifespan=lifespan)


# -- request models --------------------------------------------------------


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stream: bool = False
    # vLLM-style passthrough so a client can set template kwargs (thinking, tools)
    # without the server inventing a flag per model.
    chat_template_kwargs: dict[str, Any] | None = None


def _single_prompt(prompt: str | list[str]) -> str:
    if isinstance(prompt, list):
        if len(prompt) != 1:
            raise HTTPException(
                status_code=400,
                detail="multiple prompts per request are not supported; send one prompt at a time",
            )
        return prompt[0]
    return prompt


def _completion_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


# -- health ----------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    snap = engine.snapshot()
    return {"server_ready": True, **snap}


@app.get("/ready")
def ready() -> JSONResponse:
    snap = engine.snapshot()
    code = 200 if snap["model_loaded"] else 503
    return JSONResponse(status_code=code, content=snap)


@app.get("/v1/models")
def models() -> dict:
    data = []
    if engine.ready:
        data.append(
            {
                "id": engine.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llmkube",
            }
        )
    return {"object": "list", "data": data}


# -- completions -----------------------------------------------------------


def _sse(payload: dict, event: str | None = None) -> str:
    line = f"data: {json.dumps(payload)}\n\n"
    if event:
        return f"event: {event}\n{line}"
    return line


@app.post("/v1/completions")
def completions(req: CompletionRequest, request: Request):
    _require_ready()
    prompt = _single_prompt(req.prompt)
    created = int(time.time())
    cid = _completion_id("cmpl")
    model_name = req.model or engine.model_name

    if req.stream:

        def gen() -> Iterator[str]:
            first = True
            try:
                for chunk in engine.stream(prompt, req.max_tokens, req.temperature, req.top_p):
                    delta = {"choices": [{"index": 0, "text": chunk}]}
                    if first:
                        delta["id"] = cid
                        delta["object"] = "text_completion"
                        delta["created"] = created
                        delta["model"] = model_name
                        first = False
                    yield _sse(delta)
            except EngineError as exc:
                yield _sse({"error": {"message": str(exc), "type": "engine_error"}})
            yield _sse(
                {
                    "id": cid,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        text = engine.generate(prompt, req.max_tokens, req.temperature, req.top_p)
    except EngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "id": cid,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
    }


# -- chat ------------------------------------------------------------------


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, request: Request):
    _require_ready()
    try:
        renderer = _get_renderer()
    except ChatTemplateError as exc:
        # 500, not 400: the checkpoint is the problem, not the request.
        raise HTTPException(status_code=500, detail=f"chat template unavailable: {exc}") from exc

    try:
        prompt = renderer.render(
            [m.model_dump() for m in req.messages],
            add_generation_prompt=True,
            template_kwargs=req.chat_template_kwargs,
        )
    except ChatTemplateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    created = int(time.time())
    cid = _completion_id("chatcmpl")
    model_name = req.model or engine.model_name

    if req.stream:

        def gen() -> Iterator[str]:
            first = True
            try:
                for chunk in engine.stream(prompt, req.max_tokens, req.temperature, req.top_p):
                    delta = {"choices": [{"index": 0, "delta": {"content": chunk}}]}
                    if first:
                        delta["id"] = cid
                        delta["object"] = "chat.completion.chunk"
                        delta["created"] = created
                        delta["model"] = model_name
                        first = False
                    yield _sse(delta)
            except EngineError as exc:
                yield _sse({"error": {"message": str(exc), "type": "engine_error"}})
            yield _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        text = engine.generate(prompt, req.max_tokens, req.temperature, req.top_p)
    except EngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
