# ReAct Responses Stop Handling Design

**Goal:** Make `openai-responseapi-dify-plugin` compatible with Dify ReAct mode by keeping ReAct as a text-based protocol over `/v1/responses`, stripping unsupported upstream `stop`, and enforcing the stop sequence locally.

## Context

Dify ReAct is not using native model tool-calling. It builds a text transcript with `Thought`, `Action`, and `Observation`, and it sends a stop sequence so the model stops right before `Observation`. This matches the semantics of Chat Completions, but not OpenAI Responses requests.

The current plugin forwards Dify's `stop` field directly into the `/v1/responses` payload. For ReAct traffic this produces upstream `400 Unsupported parameter: stop`, which bubbles up as a plugin-side `502`.

## Design

### 1. Keep ReAct in text compatibility mode

When Dify ReAct calls the plugin, the plugin should continue converting prompt messages into Responses `input` items. ReAct history remains plain text transcript content and is parsed by Dify's existing CoT/ReAct parser.

We do **not** convert ReAct into native Responses tool calls. That would change the contract from text `Action: {...}` output into structured function call events and would break the current Dify ReAct parser.

### 2. Never forward `stop` to `/v1/responses`

The plugin should treat `stop` as a local post-processing concern for Responses traffic. This applies to both streaming and non-streaming requests.

Allowed parameter mapping remains:

- `max_tokens` -> `max_output_tokens`
- `response_format`/`json_schema` -> `text.format`
- GPT-5.4 reasoning controls -> Responses-compatible fields

But `stop` must be removed from the outgoing payload.

### 3. Enforce stop locally

The plugin should apply Dify stop semantics after receiving text from Responses:

- Non-streaming:
  - Parse the full Responses output text.
  - Truncate at the earliest matching stop token.
  - Return the truncated assistant content.

- Streaming:
  - Accumulate text emitted from `response.output_text.delta`.
  - Emit only the prefix that is safe to expose without leaking a stop token.
  - If a stop token is detected, stop yielding further text deltas.
  - Continue consuming upstream SSE events until terminal completion so usage can still be collected.
  - Emit the final chunk with `finish_reason="stop"` and usage when available.

### 4. Scope of the fix

This change is specifically for Responses transport compatibility. It should not alter existing native Responses tool-call behavior used by Dify function-calling agent mode.

The safest detection strategy is transport-level rather than mode-level:

- For all Responses requests, do not forward `stop`.
- Apply local stop enforcement only to assistant text output.
- Leave tool-call event parsing untouched.

This keeps the change minimal and avoids special-casing Dify internals unnecessarily.

## Testing Strategy

Add regression tests for:

1. Payload construction:
   - `build_responses_request_payload(..., stop=["Observation"])` must not include `stop`.

2. Non-streaming local enforcement:
   - Responses output containing `Observation:` must be truncated before that token.

3. Streaming local enforcement:
   - A streamed sequence that spans the stop token across deltas must stop emitting text before the token.
   - The parser should still produce the terminal usage/final chunk.

## Risks

- Streaming stop handling can accidentally leak partial stop tokens if matching is done per chunk instead of across the accumulated buffer.
- Stopping local output too aggressively can drop legitimate content near the boundary.

To control that risk, the implementation should use a rolling text buffer with longest-stop-prefix protection instead of naive per-delta replacement.
