"""Chat templating: render the model's own template, do not invent one.

Formatting a chat prompt by hand is the classic way to make a model look worse
than it is, and it fails silently: the model still answers, just badly. So this
reads the template the checkpoint ships (`tokenizer_config.json`'s
`chat_template`, or a `chat_template.jinja` beside it) and renders it with
Jinja2, which is what the template is written for. If no template is present
this raises rather than guessing.

Attention control is a template kwarg, not a prompt trick. The community recipe
measures with thinking off, so `enable_thinking=False` is the default here and
`EXL3_THINKING=1` (or a `chat_template_kwargs` field on the request) flips it.
A template that ignores the kwarg is unaffected, which is the correct failure
mode for a control this soft.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, TemplateError
from jinja2.sandbox import SandboxedEnvironment


class ChatTemplateError(RuntimeError):
    """The checkpoint ships no usable chat template."""


def _env() -> Environment:
    # Sandboxed: the template comes from a model repository, not from us.
    env = SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    # Helpers Hugging Face templates expect from transformers. Without
    # `strftime_now` a template that dates itself raises UndefinedError.
    env.globals["strftime_now"] = lambda fmt="%Y-%m-%d": datetime.now().strftime(fmt)

    def raise_exception(message: str):
        raise ChatTemplateError(message)

    env.globals["raise_exception"] = raise_exception
    return env


def load_chat_template(model_dir: str | os.PathLike[str]) -> str | None:
    base = Path(model_dir)
    jinja = base / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    cfg = base / "tokenizer_config.json"
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        template = data.get("chat_template")
        if isinstance(template, str) and template.strip():
            return template
    return None


@dataclass
class ChatRenderer:
    """Renders the model's template once, then reuses the compiled form."""

    model_dir: str
    thinking: bool = False

    _template: object | None = None

    def __post_init__(self) -> None:
        source = load_chat_template(self.model_dir)
        if source is None:
            raise ChatTemplateError(
                f"no chat template found in {self.model_dir!r} "
                "(expected chat_template.jinja or tokenizer_config.json chat_template)"
            )
        try:
            self._template = _env().from_string(source)
        except TemplateError as exc:  # a malformed template is a checkpoint problem
            raise ChatTemplateError(f"chat template did not compile: {exc}") from exc

    def render(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        template_kwargs: dict | None = None,
    ) -> str:
        kwargs = {"enable_thinking": self.thinking}
        if template_kwargs:
            kwargs.update(template_kwargs)
        try:
            return self._template.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                bos_token="",
                eos_token="",
                **kwargs,
            )
        except TemplateError as exc:
            raise ChatTemplateError(f"chat template failed to render: {exc}") from exc
