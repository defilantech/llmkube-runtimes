"""Model engine: load the checkpoint once, serialize generation behind it.

The library's API is small and the interesting decisions are all about failure
and concurrency:

- Loading takes tens of seconds and consumes the whole box (the ATS loader
  aliases ~107 GiB out of page cache), so it happens once, off the request path,
  and the HTTP port is bound before it starts. A client can therefore watch
  `/health` during a load instead of seeing connection refused.
- `Generator` is not safe to call concurrently. One lock serializes it, which is
  the honest shape for a single-stream POC: queueing is a scheduling decision,
  not something to hide behind a thread pool.
- A failed load is a reported state, not a crash. `EXL3_REQUIRE_MODEL=1` makes it
  fatal instead, which is what a serving manifest should set; the default keeps
  the process alive so CI and the Tier-1 gate can start the image with no model
  staged and still read `/ready` as not-ready.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Iterator

log = logging.getLogger("exl3.engine")

DEFAULT_CACHE_TOKENS = 32768
DEFAULT_MAX_BATCH = 1
DEFAULT_DRAFT_CONFIDENCE = 0.4


class EngineError(RuntimeError):
    """A load or generation failure that the caller should surface."""


@dataclass
class Engine:
    """Holds the loaded model and the lock that serializes it."""

    model_dir: str = ""
    draft_model_dir: str = ""
    device: str = "cuda:0"
    cache_tokens: int = DEFAULT_CACHE_TOKENS
    max_batch: int = DEFAULT_MAX_BATCH
    mtp_draft: bool = False
    draft_tokens: int | None = None
    draft_confidence: float = DEFAULT_DRAFT_CONFIDENCE

    state: str = "idle"          # idle | loading | ready | failed
    error: str = ""
    model_name: str = ""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _generate_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _model: object | None = field(default=None, repr=False)
    _cache: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)
    _generator: object | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "Engine":
        draft_tokens = os.environ.get("EXL3_DRAFT_TOKENS", "")
        return cls(
            model_dir=os.environ.get("EXL3_MODEL_DIR", "/models"),
            draft_model_dir=os.environ.get("EXL3_DRAFT_MODEL_DIR", ""),
            device=os.environ.get("EXL3_DEVICE", "cuda:0"),
            cache_tokens=int(os.environ.get("EXL3_CACHE_TOKENS", DEFAULT_CACHE_TOKENS)),
            max_batch=int(os.environ.get("EXL3_MAX_BATCH", DEFAULT_MAX_BATCH)),
            mtp_draft=os.environ.get("EXL3_MTP", "0") == "1",
            draft_tokens=int(draft_tokens) if draft_tokens else None,
            draft_confidence=float(os.environ.get("EXL3_DSPARK_CONF", DEFAULT_DRAFT_CONFIDENCE)),
        )

    # -- loading ------------------------------------------------------------

    def load(self) -> None:
        """Load synchronously. Call from a background thread, not the request path."""
        with self._lock:
            if self.state == "ready":
                return
            self.state = "loading"
            self.error = ""
        try:
            self._load_locked()
        except Exception as exc:  # noqa: BLE001 - surfaced to /health, never swallowed
            with self._lock:
                self.state = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
            log.exception("model load failed")
            if os.environ.get("EXL3_REQUIRE_MODEL", "0") == "1":
                raise EngineError(f"model load failed: {exc}") from exc
            return
        with self._lock:
            self.state = "ready"
        log.info("model ready: %s", self.model_name)

    def _load_locked(self) -> None:
        # Imported here, not at module import: the extension initializes CUDA when
        # it loads, so keeping it inside load() means `import server.app` stays
        # cheap and GPU-free for the gate and for anything that only probes /health.
        from exllamav3 import Cache, Config, Generator, Model, Tokenizer

        model_dir = self.model_dir
        if not model_dir or not os.path.isdir(model_dir):
            raise EngineError(f"model directory not found: {model_dir!r} (set EXL3_MODEL_DIR)")

        draft_model = None
        draft_cache = None
        if self.draft_model_dir:
            draft_config = Config.from_directory(self.draft_model_dir)
            draft_model = Model.from_config(draft_config)
            draft_cache = Cache(draft_model, max_num_tokens=self.cache_tokens)
            draft_model.load(progressbar=False, device=self.device)

        config = Config.from_directory(model_dir)
        model = Model.from_config(config)

        # The MTP drafter is a second component of the same checkpoint, not a
        # separate directory: DSPARK layers stored under the mtp.* namespace. It
        # needs its own cache, and that cache must be the size of the main one.
        if self.mtp_draft and draft_model is None:
            draft_model = Model.from_config(config, component="mtp")
            draft_cache = Cache(
                draft_model,
                max_num_tokens=self.cache_tokens,
                max_batch_size=self.max_batch,
            )
            draft_model.load(progressbar=False, device=self.device)

        # max_history is required only for drafting against a recurrent target.
        cache = Cache(
            model,
            max_num_tokens=self.cache_tokens,
            max_batch_size=self.max_batch,
            max_history=draft_model.caps.get("default_draft_size", 4) if draft_model else 0,
        )
        model.load(progressbar=False, device=self.device)
        tokenizer = Tokenizer.from_config(config)
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            draft_model=draft_model,
            draft_cache=draft_cache,
            num_draft_tokens=self.draft_tokens,
            dynamic_draft_tokens=True,
            draft_confidence=self.draft_confidence,
        )

        self._model = model
        self._cache = cache
        self._tokenizer = tokenizer
        self._generator = generator
        self.model_name = os.path.basename(model_dir.rstrip("/")) or "exllamav3"

    # -- introspection ------------------------------------------------------

    def count_tokens(self, text: str, add_bos: bool = False) -> int:
        """The tokenizer's count for `text`, for the OpenAI usage object.

        Prompt and completion counts both come from here, so a client that adds
        them gets one consistent number rather than one counted by us and one by
        the tokenizer. Zero before a model is loaded, which is the honest answer
        while nothing can be tokenized.
        """
        tokenizer = self._tokenizer
        if tokenizer is None or not text:
            return 0
        ids = tokenizer.encode(text, add_bos=add_bos, encode_special_tokens=True)
        # encode returns a (batch, length) tensor, so the token count is the last
        # axis: len() on it would count the batch, which is always 1.
        shape = getattr(ids, "shape", None)
        return int(shape[-1]) if shape is not None else len(ids)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.state,
                "model_loaded": self.state == "ready",
                "model": self.model_name,
                "model_dir": self.model_dir,
                "draft_model_dir": self.draft_model_dir,
                "error": self.error,
            }

    @property
    def ready(self) -> bool:
        with self._lock:
            return self.state == "ready"

    # -- generation ---------------------------------------------------------

    def _sampler(self, temperature: float, top_p: float):
        from exllamav3 import TopPSampler

        return TopPSampler(temperature=temperature, top_p=top_p, temperature_last=True)

    def _stop_conditions(self) -> list:
        return [self._tokenizer.eos_token_id]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        if not self.ready:
            raise EngineError(f"model not ready (state={self.state}: {self.error})")
        with self._generate_lock:
            return self._generator.generate(
                prompt=prompt,
                stop_conditions=self._stop_conditions(),
                max_new_tokens=max_new_tokens,
                sampler=self._sampler(temperature, top_p),
                completion_only=True,
                add_bos=True,
            )

    def stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        identifier: object = 0,
    ) -> Iterator[str]:
        """Yield text chunks. Holds the generate lock for the whole stream, so a
        second request waits rather than interleaving into the same KV cache."""
        if not self.ready:
            raise EngineError(f"model not ready (state={self.state}: {self.error})")
        from exllamav3 import Job

        with self._generate_lock:
            input_ids = self._tokenizer.encode(
                prompt, add_bos=True, encode_special_tokens=True
            )
            job = Job(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_conditions=self._stop_conditions(),
                sampler=self._sampler(temperature, top_p),
                identifier=identifier,
            )
            self._generator.enqueue(job)
            while self._generator.num_remaining_jobs():
                for result in self._generator.iterate():
                    if result.get("identifier") != identifier:
                        continue
                    text = result.get("text", "")
                    if text:
                        yield text
