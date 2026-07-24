# Laguna variant runtime (`vulkan-laguna/`)

`ghcr.io/defilantech/llmkube-llama-vulkan-laguna`

A Vulkan llama.cpp server runtime for serving [poolside](https://huggingface.co/poolside)
Laguna models **with working tool calling**. It is a variant, not a replacement:
the production runtime (`vulkan/`) stays pure upstream and is unaffected by
anything here.

## Why this variant exists

Upstream `ggml-org/llama.cpp` can load Laguna but cannot parse its tool calls.

The Laguna arch merged upstream on 2026-07-22 and first shipped in release
`b10103`, so the production image serves Laguna fine: it loads, it is fast, it
answers. What it does not do is emit tool calls. Laguna produces poolside's
native format, and upstream's tool-call parser (`common/chat.cpp`) has no
handler for it, so a tool call arrives as raw text inside `message.content`
with `finish_reason: stop`:

```
<tool_call>read_file<arg_key>path</arg_key><arg_value>internal/controller/scheduling.go</arg_value></tool_call>
```

An OpenAI-API client sees no `tool_calls` array at all. The model decided
correctly and the plumbing threw the decision away. For an agent loop (LLMKube's
Foreman coder, for one) that is fatal, and it fails in the worst way: quietly,
looking like a model quality problem.

## Provenance: TheTom's llama.cpp fork

This image builds from
[`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant),
branch `laguna/port`, pinned by commit SHA in `vulkan-laguna/Dockerfile`. That
fork is also the home of the TurboQuant KV-cache work and has been a source of
gfx1151 fixes we already depend on (the integrated-GPU submit fix for Vulkan
device-lost came from there).

Rather than hand-write a parser per model, the fork derives one. Its
**differential autoparser** (`common/chat-auto-parser*.cpp`,
`common/chat-diff-analyzer.cpp`) renders the model's jinja chat template with
varying inputs, diffs the outputs to work out where tool names, argument keys,
argument values, and reasoning blocks begin and end, then generates a PEG parser
from that. It is the automatic fallback for any template without a hand-written
parser, it takes no flag, and it engages under `--jinja`. With it, the same
request returns a normal OpenAI tool call:

```json
{
  "finish_reason": "tool_calls",
  "message": {
    "content": "I'll read the file for you to understand what it does.",
    "tool_calls": [{
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\":\"internal/controller/scheduling.go\"}"
      }
    }]
  }
}
```

Valid JSON arguments, clean content, and no reasoning-tag leakage. Nothing
downstream needs a poolside-specific code path.

## The chat template gotcha

**poolside's released GGUF embeds a chat template that cannot parse tool calls,
and this image works around it. Do not override that workaround by accident.**

The autoparser learns the format *from the template*. The template published
inside `poolside/Laguna-S-2.1-GGUF` contains `<tool_call>` but does **not**
contain `arg_key`, so the analyzer derives only the outer call delimiters and
never learns the `<arg_key>` / `<arg_value>` argument encoding the model
actually emits. The result is indistinguishable from having no parser at all:
tool calls silently land in `content` again.

The fork ships corrected templates. This image copies them to
`/app/templates/` and defaults to the S-2.1 one:

```
ENV LLAMA_ARG_CHAT_TEMPLATE_FILE=/app/templates/poolside-Laguna-S-2.1.jinja
```

So the image is correct out of the box. llama.cpp applies environment defaults
before command-line arguments, so serving a different Laguna is still just
`--chat-template-file /app/templates/poolside-Laguna-XS-2.1.jinja`. What you
must not do is point the template at the model's own embedded one, which
re-introduces the bug.

Templates present in the image:

- `poolside-Laguna-S-2.1.jinja` (the default)
- `poolside-Laguna-XS-2.1.jinja`
- `poolside-Laguna-XS.2.jinja`

### Diagnosing a suspected template problem

Run the server with `--verbose` and look for what the analyzer derived:

```
D common_chat_templates_apply_jinja: using differential autoparser
D per_call_start: '<tool_call>'
D per_call_end: '</tool_call>'
D arg_name_prefix: '<arg_key>'      <-- absent when the template is the broken one
D arg_name_suffix: '</arg_key>'
```

If `arg_name_prefix` is missing, the template is wrong. Inspect the one actually
in use with `curl http://<host>:8080/props` and check whether it contains
`arg_key`.

One false alarm to know about: the startup line `Chat format: peg-native` is
**not** a failure signal. It is the normal format label for autoparser output
(and for several other handlers), not evidence that parsing was skipped.

## Serving

Same hardware contract as the production Vulkan image: mount `/dev/dri` via a
device-plugin resource, grant the host render group through
`securityContext.supplementalGroups`, and request no `nvidia.com/gpu`.

Conservative flags for Strix Halo (`gfx1151`), matching what the model was
validated with:

```yaml
extraArgs:
  - --jinja              # required: the autoparser runs on this path
  - --no-mmap            # the Q4_K_M weights exceed the mmap-friendly window
  - --flash-attn
  - "off"                # flash attention on RDNA3.5 is flagged upstream
  - --spec-type
  - none                 # DFlash speculative decode needs flash attention on
  - --cache-type-k
  - f16
  - --cache-type-v
  - f16
```

`--jinja` is load-bearing. Without it the autoparser never runs and tool calls
stop parsing.

Thinking output is handled by the template the image already defaults to; add
`--reasoning off` if you want the model's reasoning suppressed rather than
returned in a separate field.

## Gates

Same two-tier model as the other runtime images (see the README).

Build-stage guards, all GPU-independent, all failing the build rather than
shipping a quietly broken image:

1. The pinned SHA is the commit that got checked out.
2. The laguna arch compiled in (grepped from the shared library, since the arch
   string lives in libllama rather than `llama-server`).
3. The fork's autoparser compiled in. This catches the pin being retargeted at
   an upstream commit, which would build clean and serve Laguna at full speed
   while emitting zero tool calls.
4. The shipped chat template carries `arg_key`.
5. The #725 guard: an `RTLD_NOW` dlopen of `libggml-vulkan.so`, so an
   unresolved shader symbol fails here instead of on a GPU at shader-use time.

CI then runs `scripts/tier1-gate.sh` (the Vulkan backend loads in the runtime
image) and `scripts/laguna-gate.sh`, which re-checks the arch, the autoparser,
and the default template against the **final image**, because a mistaken `COPY`
path or a stale `ENV` default would pass every build-stage guard and still
produce an image that never emits a tool call.

A real GPU round trip stays Tier-2, out of band on `gfx1151` hardware.

## Bumping the pin

Edit `LLAMACPP_SHA` in `vulkan-laguna/Dockerfile`. There is no companion ref to
keep in sync: the pin is a bare commit SHA on purpose, because `laguna/port` is
an active branch whose HEAD moves and a branch clone would silently build
different source over time. The build fetches that exact commit, so every bump
is a reviewed edit.

Bumps are independent of the production image's `LLAMACPP_REF` / `LLAMACPP_SHA`.
The two runtimes intentionally do not move together.

## Scope

Server image only. Benchmark numbers come from the pure-upstream tools image so
they stay comparable across models and runtimes; this variant exists to serve
Laguna, not to measure it.

## Credit

The Laguna port, the differential autoparser, and the corrected chat templates
are [TheTom](https://github.com/TheTom)'s work in
[`llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant). This
repo only packages that build and gates it. Laguna itself is poolside's, released
under OpenMDW-1.1.
