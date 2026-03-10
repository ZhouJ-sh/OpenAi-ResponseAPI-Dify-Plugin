# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportPrivateUsage=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnusedCallResult=false

"""LLM invoke 集成契约测试。"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Protocol, cast

import pytest

from dify_plugin.entities.model.llm import LLMResultChunk
from dify_plugin.entities.model.message import AssistantPromptMessage, PromptMessageTool, ToolPromptMessage, UserPromptMessage

from models.llm.llm import Sub2apiPluginLargeLanguageModel


class _LLMInvokeRequestProtocol(Protocol):
    provider: str
    model: str
    credentials: dict[str, str]


class _AssistantMessageProtocol(Protocol):
    content: str


class _LLMResultProtocol(Protocol):
    message: _AssistantMessageProtocol


class _FakeResponsesClient:
    """最小可注入 Responses 客户端，用于拦截 `_invoke()` 的 HTTP 调用参数。"""

    response: object

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self.response


class _RejectFunctionCallOutputClient(_FakeResponsesClient):
    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        input_items = cast(list[object], kwargs.get("input") or [])
        if any(isinstance(item, dict) and item.get("type") == "function_call_output" for item in input_items):
            raise ValueError('{"error":{"message":"function_call_output not supported","type":"upstream_error"}}')
        return self.response


def test_llm_invoke_request_uses_customizable_model_credentials(
    llm_invoke_request: _LLMInvokeRequestProtocol,
) -> None:
    """确认集成测试沿用 customizable-model 的凭证字段。"""

    assert llm_invoke_request.provider == "sub2api-plugin"
    assert llm_invoke_request.model == "gpt-4.1-mini"
    assert llm_invoke_request.credentials["endpoint_url"] == "https://example.com/v1"


def test_create_responses_client_normalizes_endpoint_url_to_v1_responses() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    assert llm._create_responses_client({"endpoint_url": "https://host"})._responses_url == "https://host/v1/responses"
    assert llm._create_responses_client({"endpoint_url": "https://host/"})._responses_url == "https://host/v1/responses"
    assert llm._create_responses_client({"endpoint_url": "https://host/v1"})._responses_url == "https://host/v1/responses"
    assert llm._create_responses_client({"endpoint_url": "https://host/v1/"})._responses_url == "https://host/v1/responses"
    assert (
        llm._create_responses_client({"endpoint_url": "https://host/v1/responses"})._responses_url
        == "https://host/v1/responses"
    )


def test_invoke_calls_responses_api_without_previous_response_id() -> None:
    """确认 HTTP 路径默认不发送 previous_response_id，并能解析非流式响应。"""

    llm = Sub2apiPluginLargeLanguageModel([])
    client = _FakeResponsesClient(
        {
            "id": "resp_test",
            "model": "gpt-4.1-mini",
            "status": "completed",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        }
    )
    llm._create_responses_client = lambda credentials: client  # type: ignore[method-assign]

    result = cast(
        _LLMResultProtocol,
        cast(
            object,
            llm._invoke(
                model="gpt-4.1-mini",
                credentials={"endpoint_url": "https://example.com/v1", "api_key": "test-key"},
                prompt_messages=[UserPromptMessage(content="hello")],
                model_parameters={"temperature": 0.2},
                tools=None,
                stop=None,
                stream=False,
                user=None,
            ),
        ),
    )

    assert result.message.content == "hello"

    assert client.calls == [
        {
            "model": "gpt-4.1-mini",
            "stream": False,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "temperature": 0.2,
        }
    ]


def test_invoke_streams_responses_chunks_with_mock_client(
    responses_stream_fixture: str,
) -> None:
    """确认 `_invoke()` 可通过 mock 客户端走完整的 Responses 流式解析链路。"""

    llm = Sub2apiPluginLargeLanguageModel([])
    client = _FakeResponsesClient(responses_stream_fixture)
    llm._create_responses_client = lambda credentials: client  # type: ignore[method-assign]

    result = llm._invoke(
        model="gpt-4.1-mini",
        credentials={"endpoint_url": "https://example.com/v1", "api_key": "test-key"},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={},
        tools=[
            PromptMessageTool(
                name="get_weather",
                description="查询城市天气",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ],
        stop=None,
        stream=True,
        user="user-123",
    )

    chunks = list(cast(Generator[LLMResultChunk, None, None], result))

    assert "user" not in client.calls[0]
    assert "previous_response_id" not in client.calls[0]
    assert [chunk.delta.message.content for chunk in chunks[:2]] == ["hel", "lo"]
    assert chunks[2].delta.message.tool_calls[0].function.name == "get_weather"
    assert chunks[-1].delta.finish_reason == "stop"


def test_invoke_omits_user_field_for_real_sub2api_compatibility() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    client = _FakeResponsesClient(
        {
            "id": "resp_test",
            "model": "gpt-4.1-mini",
            "status": "completed",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        }
    )
    llm._create_responses_client = lambda credentials: client  # type: ignore[method-assign]

    _ = llm._invoke(
        model="gpt-4.1-mini",
        credentials={"endpoint_url": "https://example.com/v1", "api_key": "test-key"},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user="real-user-id",
    )

    assert "user" not in client.calls[0]


def test_invoke_stream_tolerates_response_in_progress_event() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    client = _FakeResponsesClient(
        "\n".join(
            [
                'data: {"type":"response.created","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
                'data: {"type":"response.in_progress","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
                'data: {"type":"response.output_text.delta","item_id":"msg_test","delta":"hello"}',
                'data: {"type":"response.completed","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}',
                "data: [DONE]",
            ]
        )
    )
    llm._create_responses_client = lambda credentials: client  # type: ignore[method-assign]

    chunks = list(
        cast(
            Generator[LLMResultChunk, None, None],
            llm._invoke(
                model="gpt-4.1-mini",
                credentials={"endpoint_url": "https://example.com/v1", "api_key": "test-key"},
                prompt_messages=[UserPromptMessage(content="hello")],
                model_parameters={},
                tools=None,
                stop=None,
                stream=True,
                user=None,
            ),
        )
    )

    assert [chunk.delta.message.content for chunk in chunks[:-1]] == ["hello"]
    assert chunks[-1].delta.finish_reason == "stop"


def test_invoke_rewrites_tool_result_follow_up_for_sub2api_compatibility() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    client = _RejectFunctionCallOutputClient(
        {
            "id": "resp_test",
            "model": "gpt-4.1-mini",
            "status": "completed",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        }
    )
    llm._create_responses_client = lambda credentials: client  # type: ignore[method-assign]

    result = cast(
        _LLMResultProtocol,
        cast(
            object,
            llm._invoke(
                model="gpt-4.1-mini",
                credentials={"endpoint_url": "https://example.com/v1", "api_key": "test-key"},
                prompt_messages=[
                    UserPromptMessage(content="现在几点"),
                    AssistantPromptMessage(
                        content="我先调用时间工具。",
                        tool_calls=[
                            AssistantPromptMessage.ToolCall(
                                id="call_time",
                                type="function",
                                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                    name="current_time",
                                    arguments="{}",
                                ),
                            )
                        ],
                    ),
                    ToolPromptMessage(content="2026-03-10T18:00:00+08:00", tool_call_id="call_time"),
                ],
                model_parameters={"temperature": 0.1},
                tools=[
                    PromptMessageTool(
                        name="current_time",
                        description="获取当前时间",
                        parameters={"type": "object", "properties": {}},
                    )
                ],
                stop=None,
                stream=False,
                user=None,
            ),
        ),
    )

    assert result.message.content == "done"
    assert all(
        not (isinstance(item, dict) and item.get("type") == "function_call_output")
        for item in cast(list[object], client.calls[0]["input"])
    )
    assert any(
        isinstance(item, dict)
        and item.get("role") == "assistant"
        and "[tool_result id=call_time name=current_time]" in json.dumps(item, ensure_ascii=False)
        for item in cast(list[object], client.calls[0]["input"])
    )


def test_invoke_raises_for_failed_stream_event() -> None:
    """确认 `_invoke()` 遇到 failed 流事件时会向上抛错。"""

    llm = Sub2apiPluginLargeLanguageModel([])
    client = _FakeResponsesClient(
        "\n".join(
            [
                'data: {"type":"response.created","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"in_progress"}}',
                'data: {"type":"response.failed","response":{"id":"resp_test","model":"gpt-4.1-mini","status":"failed","error":{"code":"server_error","message":"boom"}}}',
                "data: [DONE]",
            ]
        )
    )
    llm._create_responses_client = lambda credentials: client  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="boom"):
        list(
            llm._invoke(
                model="gpt-4.1-mini",
                credentials={"endpoint_url": "https://example.com/v1", "api_key": "test-key"},
                prompt_messages=[UserPromptMessage(content="hello")],
                model_parameters={},
                tools=None,
                stop=None,
                stream=True,
                user=None,
            )
        )
