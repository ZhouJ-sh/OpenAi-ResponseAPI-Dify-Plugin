# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportPrivateUsage=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownMemberType=false

"""Responses parser 的真实契约测试。"""

from __future__ import annotations

import json
from typing import cast

import pytest

from dify_plugin.entities.model.message import UserPromptMessage

from models.llm.llm import Sub2apiPluginLargeLanguageModel


def test_responses_stream_fixture_covers_required_event_types(
    responses_stream_fixture: str,
) -> None:
    """确认流式样本覆盖文本、工具调用与完成事件。"""

    assert '"type":"response.created"' in responses_stream_fixture
    assert '"type":"response.output_item.added"' in responses_stream_fixture
    assert '"type":"response.output_text.delta"' in responses_stream_fixture
    assert '"type":"response.output_text.done"' in responses_stream_fixture
    assert '"type":"response.function_call_arguments.delta"' in responses_stream_fixture
    assert '"type":"response.function_call_arguments.done"' in responses_stream_fixture
    assert '"type":"response.output_item.done"' in responses_stream_fixture
    assert '"type":"response.completed"' in responses_stream_fixture
    assert "data: [DONE]" in responses_stream_fixture


def test_parse_responses_response_extracts_text_and_function_calls(
    responses_response_fixture: dict[str, object],
) -> None:
    """确认非流式响应会合并文本并提取 function_call。"""

    llm = Sub2apiPluginLargeLanguageModel([])
    response = {
        **responses_response_fixture,
        "output": [
            cast(list[object], responses_response_fixture["output"])[0],
            {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_weather",
                "name": "get_weather",
                "arguments": '{"city":"Hangzhou"}',
            },
        ],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 7,
            "total_tokens": 10,
        },
    }

    result = llm._parse_responses_response(
        model="gpt-4.1-mini",
        credentials={},
        prompt_messages=[UserPromptMessage(content="hello")],
        response=response,
    )

    assert result.model == "gpt-4.1-mini"
    assert result.message.content == "hello"
    assert len(result.message.tool_calls) == 1
    tool_call = result.message.tool_calls[0]
    assert tool_call.id == "call_weather"
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {"city": "Hangzhou"}
    assert result.system_fingerprint == "resp_test"
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 7


def test_parse_responses_response_rejects_incomplete_status() -> None:
    """确认非流式 incomplete 会显式失败，而不是静默返回半成品。"""

    llm = Sub2apiPluginLargeLanguageModel([])

    with pytest.raises(ValueError, match="max_output_tokens"):
        llm._parse_responses_response(
            model="gpt-4.1-mini",
            credentials={},
            prompt_messages=[UserPromptMessage(content="hello")],
            response={
                "id": "resp_incomplete",
                "model": "gpt-4.1-mini",
                "status": "incomplete",
                "output": [],
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        )


def test_parse_responses_stream_emits_text_tool_call_and_stop_chunk(
    responses_stream_fixture: str,
) -> None:
    """确认流式解析会按状态机输出文本块、完整工具调用块与结束块。"""

    llm = Sub2apiPluginLargeLanguageModel([])

    chunks = list(
        llm._parse_responses_stream(
            model="gpt-4.1-mini",
            credentials={},
            prompt_messages=[UserPromptMessage(content="hello")],
            response_lines=responses_stream_fixture.splitlines(),
        )
    )

    assert [chunk.delta.message.content for chunk in chunks[:2]] == ["hel", "lo"]
    assert chunks[2].delta.message.content == ""
    assert len(chunks[2].delta.message.tool_calls) == 1
    tool_call = chunks[2].delta.message.tool_calls[0]
    assert tool_call.id == "call_weather"
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {"city": "Hangzhou"}
    final_chunk = chunks[-1]
    assert final_chunk.delta.finish_reason == "stop"
    assert final_chunk.delta.usage is not None
    assert final_chunk.delta.usage.prompt_tokens == 4
    assert final_chunk.delta.usage.completion_tokens == 6


def test_parse_responses_stream_rejects_failed_event() -> None:
    """确认流式 failed 事件会立即中断并抛出明确异常。"""

    llm = Sub2apiPluginLargeLanguageModel([])
    response_lines = [
        'data: {"type":"response.created","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
        'data: {"type":"response.failed","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"failed","error":{"code":"server_error","message":"boom"}}}',
        "data: [DONE]",
    ]

    with pytest.raises(ValueError, match="boom"):
        list(
            llm._parse_responses_stream(
                model="gpt-4.1-mini",
                credentials={},
                prompt_messages=[UserPromptMessage(content="hello")],
                response_lines=response_lines,
            )
        )


def test_parse_responses_stream_handles_null_incomplete_details() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    response_lines = [
        'data: {"type":"response.created","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
        'data: {"type":"response.incomplete","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"incomplete","incomplete_details":null}}',
        "data: [DONE]",
    ]

    with pytest.raises(ValueError, match="Responses 请求未完成"):
        list(
            llm._parse_responses_stream(
                model="gpt-4.1-mini",
                credentials={},
                prompt_messages=[UserPromptMessage(content="hello")],
                response_lines=response_lines,
            )
        )


def test_parse_responses_stream_rejects_error_event() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    response_lines = [
        'data: {"type":"response.created","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
        'data: {"type":"error","error":{"type":"upstream_error","message":"Upstream request failed"}}',
        "data: [DONE]",
    ]

    with pytest.raises(ValueError, match="Upstream request failed"):
        list(
            llm._parse_responses_stream(
                model="gpt-4.1-mini",
                credentials={},
                prompt_messages=[UserPromptMessage(content="hello")],
                response_lines=response_lines,
            )
        )


def test_parse_responses_stream_waits_until_done_before_emitting_tool_call() -> None:
    """确认参数 delta 只累计，不会在 done 之前提前产出半成品 tool call。"""

    llm = Sub2apiPluginLargeLanguageModel([])
    response_lines = [
        'data: {"type":"response.created","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
        'data: {"type":"response.output_item.added","item":{"id":"fc_test","type":"function_call","call_id":"call_weather","name":"get_weather","arguments":""}}',
        "data: " + json.dumps({"type": "response.function_call_arguments.delta", "item_id": "fc_test", "delta": '{\"city\":'}),
        "data: " + json.dumps({"type": "response.function_call_arguments.delta", "item_id": "fc_test", "delta": '\"Hangzhou\"}'}),
        'data: {"type":"response.completed","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}',
        "data: [DONE]",
    ]

    chunks = list(
        llm._parse_responses_stream(
            model="gpt-4.1-mini",
            credentials={},
            prompt_messages=[UserPromptMessage(content="hello")],
            response_lines=response_lines,
        )
    )

    assert all(not chunk.delta.message.tool_calls for chunk in chunks[:-1])
    assert chunks[-1].delta.finish_reason == "stop"


def test_parse_responses_stream_explicitly_ignores_output_text_done() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    response_lines = [
        'data: {"type":"response.created","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
        'data: {"type":"response.output_text.delta","item_id":"msg_test","delta":"hello"}',
        'data: {"type":"response.output_text.done","item_id":"msg_test","text":"hello"}',
        'data: {"type":"response.completed","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}',
        "data: [DONE]",
    ]

    chunks = list(
        llm._parse_responses_stream(
            model="gpt-4.1-mini",
            credentials={},
            prompt_messages=[UserPromptMessage(content="hello")],
            response_lines=response_lines,
        )
    )

    assert [chunk.delta.message.content for chunk in chunks[:-1]] == ["hello"]
    assert chunks[-1].delta.finish_reason == "stop"


def test_parse_responses_stream_ignores_response_in_progress_event() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    response_lines = [
        'data: {"type":"response.created","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
        'data: {"type":"response.in_progress","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
        'data: {"type":"response.output_text.delta","item_id":"msg_test","delta":"hello"}',
        'data: {"type":"response.completed","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}',
        "data: [DONE]",
    ]

    chunks = list(
        llm._parse_responses_stream(
            model="gpt-4.1-mini",
            credentials={},
            prompt_messages=[UserPromptMessage(content="hello")],
            response_lines=response_lines,
        )
    )

    assert [chunk.delta.message.content for chunk in chunks[:-1]] == ["hello"]
    assert chunks[-1].delta.finish_reason == "stop"


def test_parse_responses_stream_ignores_content_part_events() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    response_lines = [
        'data: {"type":"response.created","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
        'data: {"type":"response.output_item.added","item":{"id":"msg_test","type":"message","status":"in_progress","content":[],"role":"assistant"}}',
        'data: {"type":"response.content_part.added","content_index":0,"item_id":"msg_test","output_index":0,"part":{"type":"output_text","text":""}}',
        'data: {"type":"response.output_text.delta","content_index":0,"item_id":"msg_test","delta":"hello"}',
        'data: {"type":"response.output_text.done","content_index":0,"item_id":"msg_test","text":"hello"}',
        'data: {"type":"response.content_part.done","content_index":0,"item_id":"msg_test","output_index":0,"part":{"type":"output_text","text":"hello"}}',
        'data: {"type":"response.output_item.done","item":{"id":"msg_test","type":"message","status":"completed","content":[{"type":"output_text","text":"hello"}],"role":"assistant"}}',
        'data: {"type":"response.completed","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}',
        "data: [DONE]",
    ]

    chunks = list(
        llm._parse_responses_stream(
            model="gpt-4.1-mini",
            credentials={},
            prompt_messages=[UserPromptMessage(content="hello")],
            response_lines=response_lines,
        )
    )

    assert [chunk.delta.message.content for chunk in chunks[:-1]] == ["hello"]
    assert chunks[-1].delta.finish_reason == "stop"


def test_parse_responses_stream_rejects_unknown_business_event() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    response_lines = [
        'data: {"type":"response.created","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
        'data: {"type":"response.output_audio.delta","item_id":"msg_test","delta":"abc"}',
        "data: [DONE]",
    ]

    with pytest.raises(ValueError, match="response.output_audio.delta"):
        list(
            llm._parse_responses_stream(
                model="gpt-4.1-mini",
                credentials={},
                prompt_messages=[UserPromptMessage(content="hello")],
                response_lines=response_lines,
            )
        )
