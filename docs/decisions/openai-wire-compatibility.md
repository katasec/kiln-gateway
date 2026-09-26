# OpenAI wire compatibility — architecture note

> Informal note capturing the reasoning behind how kiln-gateway handles
> OpenAI-compatible request/response shapes, so it isn't lost in future
> discussions. Not a formal ADR — we'll decide later whether Kiln adopts an ADR
> process, numbering, Mission Control integration, etc.
>
> Captured: 2026-06-01.

## Context

kiln-gateway implements the **server** side of the OpenAI API (Chat Completions
and Responses) so host CLIs like Codex can point at a kiln agent. This is the
*northbound* direction.

This is the inverse of kiln, which uses native provider SDKs (OpenAI,
Anthropic, …) as a **client** — the *southbound* direction — precisely so it does
not own third-party payload shapes. That principle does not transfer directly to
the gateway, because the OpenAI Go SDK is a client library and there is no
official server-side OpenAI contract for Go.

We investigated whether a reusable contract exists so we would not hand-maintain
wire shapes:

- **Official OpenAPI spec (`openai/openai-openapi`)** — canonical, but OpenAPI
  **3.1.0**, ~26k lines, 284 schemas, heavy `oneOf`, and "best-effort" maintained.
  `oapi-codegen` does not support 3.1; `ogen` does but emits a large package with
  its own runtime, and the spec's inaccuracies would be inherited.
- **`sashabaranov/go-openai`** — community lib with plain, both-direction structs,
  but it is a client dependency and its Responses API coverage lags.
- **`openai/openai-go`** (already a transitive dep) — good for response/stream
  *emission*, poor for request *ingestion* (send-oriented param/union types).
- **LocalAI** (a mature Go OpenAI-compatible server) hand-rolls its own internal
  schema package rather than depending on a generated contract.

The surface we actually serve (chat/completions + responses; text, model, stream)
is small and de-facto frozen; the churn in OpenAI's API is in features we do not
expose.

## Current approach

1. **Do not** generate Go types from the official OpenAI OpenAPI spec.
2. **Do not** add `sashabaranov/go-openai` solely for its types.
3. Keep a **small hand-rolled compatibility slice** covering only the
   endpoints/features we actually serve.
4. Isolate **all** OpenAI-compatible wire concerns behind an internal boundary
   package: **`internal/oaiwire`**.
5. The rest of kiln-gateway deals in **Kiln-native request/result types**, not
   OpenAI payload structs. `oaiwire` is the only place that knows "chat" vs
   "responses" or imports any OpenAI SDK.
6. Use the official OpenAI spec as a future **validation/reference** mechanism
   (e.g. contract tests against its schemas), not as generated code.
7. **Revisit** a generated or third-party types dependency if the supported
   surface expands significantly (e.g. tools, vision, audio).

## Consequences

**Positive**
- Minimal dependency surface; no codegen pipeline or regen discipline to own.
- The core gateway stays provider-agnostic; OpenAI compatibility lives in one
  swappable package.
- The contract source (hand-rolled → go-openai → generated) can change later
  behind the `oaiwire` seam without touching the core.

**Costs / risks**
- We own the compatibility slice and must track OpenAI changes for the endpoints
  we serve.
- *Mitigation:* the served surface is small and stable; a future spec-based
  validation test catches drift; this note is revisited if the surface grows.

## Alternatives considered

- **Generate from the spec (oapi-codegen / ogen):** set aside — 3.1 blocks
  oapi-codegen; ogen is heavy; spec accuracy is best-effort.
- **Adopt `go-openai` for types:** set aside for now — a client dependency for a
  tiny surface; Responses API coverage lags.
- **Use `openai-go` for everything:** evaluated via a marshal spike and rejected
  for emission — see below.

## Marshal spike & decision (2026-06-01)

Before any `oaiwire` refactor we ran a throwaway spike: construct the candidate
`openai-go` response/event types, `json.Marshal` them, and diff against the
gateway's current emitted JSON. Findings:

- **`openai.Model`** — marshals to the exact `{id, created, object:"model",
  owned_by}` shape. The only clean 1:1 replacement.
- **`openai.ChatCompletion` / `ChatCompletionChunk`** — emit the correct core
  fields but as a noisy, partly off-spec superset: `service_tier:""`, an
  always-present `audio:{}` object, `refusal:""`, `tool_calls:null`, `logprobs`,
  zero `usage` on every stream chunk, `finish_reason:""` on every delta.
- **`responses.Response`** — unusable. Its `Output` is a flattened decode union,
  so marshaling dumps every variant's fields (`type:"computer_screenshot"`,
  `arguments:{"OfString":""}`, `action:{…}`, …) plus `error:{}` instead of
  `null`. Flat events like `ResponseTextDeltaEvent` marshal fine, but the
  `created`/`in_progress`/`completed` events embed a full `Response` and inherit
  the breakage.

Root cause: `openai-go`'s response types are built to be **decoded** (no
`omitempty`, `respjson` metadata, decode-oriented unions), not produced. They do
not yield clean server emission.

**Decision:** SDK response types were evaluated and rejected for gateway emission
because they marshal noisy/off-spec server responses; Kiln Gateway will keep its
small hand-rolled compatibility slice for now. Concretely:

- Keep the small hand-rolled wire slice.
- Do **not** introduce SDK coupling for response emission.
- Do **not** create `oaiwire` as an isolation-only refactor today.
- Do **not** replace the chat/stream/responses structs with SDK types.
- `modelInfo -> openai.Model` is possible (the one clean win) but not worth a
  separate change right now.

Revisit only if the served surface grows materially or the maintenance cost of
the hand-rolled slice becomes real.

## References

- `openai/openai-openapi` — https://github.com/openai/openai-openapi
- `oapi-codegen` — https://github.com/oapi-codegen/oapi-codegen ·
  `ogen` — https://github.com/ogen-go/ogen
- `sashabaranov/go-openai` — https://github.com/sashabaranov/go-openai ·
  `openai/openai-go` — https://github.com/openai/openai-go
- LocalAI — https://localai.io/
