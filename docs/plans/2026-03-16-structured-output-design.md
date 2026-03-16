# Structured Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add structured output support to the Responses-based Dify plugin so Dify `response_format` and `json_schema` parameters are exposed and mapped to OpenAI Responses `text.format`.

**Architecture:** Extend the plugin's runtime model schema to advertise structured-output parameters for customizable models, then normalize those parameters into Responses-native payload fields during request construction. Keep parsing unchanged because structured output still arrives as assistant text.

**Tech Stack:** Python 3.12, Dify plugin SDK, pytest

---

### Task 1: Add failing mapper tests

**Files:**
- Modify: `tests/test_responses_mapper.py`

**Step 1: Write the failing tests**
- Add tests for `text`, `json_object`, and `json_schema` mapping.
- Add a GPT-5.4 test that ensures `verbosity` and structured output share the same `text` object.
- Add validation tests for invalid or misplaced `json_schema`.

**Step 2: Run test to verify it fails**

Run: `uv run --project . pytest tests/test_responses_mapper.py -q`

Expected: FAIL because the plugin does not yet normalize `response_format` into Responses `text.format`.

### Task 2: Implement runtime parameter normalization

**Files:**
- Modify: `models/llm/llm.py`

**Step 1: Write minimal implementation**
- Add helpers to parse Dify `json_schema` input and normalize it into Responses `text.format`.
- Merge structured-output config with existing GPT-5.4 `text.verbosity` handling.
- Reject invalid parameter combinations clearly.

**Step 2: Run focused tests**

Run: `uv run --project . pytest tests/test_responses_mapper.py -q`

Expected: PASS

### Task 3: Expose runtime schema parameters

**Files:**
- Modify: `models/llm/llm.py`
- Modify: `tests/test_provider_schema.py`

**Step 1: Add failing schema tests**
- Assert non-GPT-5.4 customizable models expose base sampling, max tokens, `response_format`, and `json_schema`.
- Assert GPT-5.4 additionally exposes structured-output parameters together with reasoning controls.

**Step 2: Implement minimal schema changes**
- Return base parameter rules for all models.
- Restrict `response_format` options by model family where needed.

**Step 3: Run focused tests**

Run: `uv run --project . pytest tests/test_provider_schema.py -q`

Expected: PASS

### Task 4: Regression verification

**Files:**
- Verify: `tests/test_llm_invoke.py`
- Verify: `tests/test_responses_parser.py`

**Step 1: Run targeted regression suite**

Run: `uv run --project . pytest tests/test_responses_mapper.py tests/test_provider_schema.py tests/test_llm_invoke.py tests/test_responses_parser.py -q`

Expected: PASS

**Step 2: Run full test suite**

Run: `uv run --project . pytest -q`

Expected: PASS
