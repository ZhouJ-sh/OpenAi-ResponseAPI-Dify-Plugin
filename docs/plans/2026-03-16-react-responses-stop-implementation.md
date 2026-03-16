# ReAct Responses Stop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Responses-based Dify plugin compatible with Dify ReAct mode by stripping upstream `stop` and enforcing stop tokens locally for both block and stream outputs.

**Architecture:** Keep ReAct as text-mode over Responses. The request mapper will no longer send `stop` to `/v1/responses`. Instead, the response parser will truncate assistant text locally, including streaming-safe incremental truncation that preserves usage collection from terminal SSE events.

**Tech Stack:** Python 3.13, Dify plugin SDK entities, pytest

---

### Task 1: Add failing payload regression test

**Files:**
- Modify: `/Users/zhou/Code/dify/openai-responseapi-dify-plugin/tests/test_responses_mapper.py`

**Step 1: Write the failing test**

Add a test asserting that `build_responses_request_payload(..., stop=["Observation"])` omits `stop` from the returned payload while preserving `max_output_tokens`, `input`, and any tool metadata.

**Step 2: Run test to verify it fails**

Run: `uv run --project . pytest tests/test_responses_mapper.py -q`
Expected: FAIL because the payload still contains `stop`.

**Step 3: Write minimal implementation**

Update request payload construction so `stop` is not copied into Responses payloads.

**Step 4: Run test to verify it passes**

Run: `uv run --project . pytest tests/test_responses_mapper.py -q`
Expected: PASS

### Task 2: Add failing non-stream stop enforcement regression test

**Files:**
- Modify: `/Users/zhou/Code/dify/openai-responseapi-dify-plugin/tests/test_llm_invoke.py`

**Step 1: Write the failing test**

Add a non-streaming invoke test where the mocked Responses output text includes trailing `Observation:` content. Assert that the resulting assistant message content is truncated before the stop token.

**Step 2: Run test to verify it fails**

Run: `uv run --project . pytest tests/test_llm_invoke.py -q`
Expected: FAIL because the plugin currently returns the untrimmed text.

**Step 3: Write minimal implementation**

Apply local stop enforcement during non-stream Responses parsing.

**Step 4: Run test to verify it passes**

Run: `uv run --project . pytest tests/test_llm_invoke.py -q`
Expected: PASS

### Task 3: Add failing stream stop enforcement regression test

**Files:**
- Modify: `/Users/zhou/Code/dify/openai-responseapi-dify-plugin/tests/test_llm_invoke.py`

**Step 1: Write the failing test**

Add a streaming invoke test where `Observation:` is split across multiple `response.output_text.delta` SSE events. Assert that:

- emitted text stops before `Observation:`
- no stop token text is leaked
- the terminal chunk still contains usage and `finish_reason="stop"`

**Step 2: Run test to verify it fails**

Run: `uv run --project . pytest tests/test_llm_invoke.py -q`
Expected: FAIL because the plugin currently streams the raw deltas.

**Step 3: Write minimal implementation**

Add rolling local stop handling for streamed assistant text while continuing to consume terminal SSE events.

**Step 4: Run test to verify it passes**

Run: `uv run --project . pytest tests/test_llm_invoke.py -q`
Expected: PASS

### Task 4: Verify the focused regression suite

**Files:**
- Test: `/Users/zhou/Code/dify/openai-responseapi-dify-plugin/tests/test_responses_mapper.py`
- Test: `/Users/zhou/Code/dify/openai-responseapi-dify-plugin/tests/test_llm_invoke.py`
- Test: `/Users/zhou/Code/dify/openai-responseapi-dify-plugin/tests/test_responses_parser.py`

**Step 1: Run focused suite**

Run: `uv run --project . pytest tests/test_responses_mapper.py tests/test_llm_invoke.py tests/test_responses_parser.py -q`
Expected: PASS

**Step 2: Review for regressions**

Confirm that existing Responses tool-call parsing tests still pass and the new stop behavior only affects assistant text output.
