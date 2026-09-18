"""Chat templating: render the model's own template, do not invent one.

Formatting a chat prompt by hand is the classic way to make a model look worse
than it is, and it fails silently: the model still answers, just badly. So this
reads the template the checkpoint ships (`tokenizer_config.json`'s
`chat_template`, or a `chat_template.jinja` beside it) and renders it with
Jinja2, which is what the template is written for.

Some packs ship no template at all: DeepSeek-V4.1-Flash EXL3 does not. For those
a fallback is built from the checkpoint's own special tokens, read out of
`tokenizer.json`, so the turn boundaries and prompt opener are the model's, not
ours. A pack template always wins when present. A template that is present but
malformed still raises.

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


# Turn markers as they appear in this checkpoint's tokenizer.json added_tokens.
# \uff5c is the full-width vertical bar the family uses around role names.
_BOS = "<\uff5cbegin_of_sentence\uff5c>"
_EOS = "<\uff5cend_of_sentence\uff5c>"
_USER = "<\uff5cUser\uff5c>"
_ASSISTANT = "<\uff5cAssistant\uff5c>"
_THINK = " thinking"

DEEPSEEK_FALLBACK_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '" + _BOS + "' + message['content'] }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '" + _USER + "' + message['content'] + '" + _EOS + "' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '" + _ASSISTANT + "' + message['content'] + '" + _EOS + "' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '" + _ASSISTANT + "' + ('" + _THINK + "' if enable_thinking else '') }}"
    "{% endif %}"
)


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
            source = DEEPSEEK_FALLBACK_TEMPLATE
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
