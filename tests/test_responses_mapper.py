# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnusedCallResult=false

"""Responses request mapper 的真实契约测试。"""

from models.llm.llm import Sub2apiPluginLargeLanguageModel
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)


def test_responses_request_fixture_defines_minimal_contract(
    responses_request_fixture: dict[str, object],
) -> None:
    """确认最小协议样本仍可作为 Responses request 基线。"""

    assert responses_request_fixture["model"] == "gpt-4.1-mini"
    assert responses_request_fixture["stream"] is True
    assert responses_request_fixture["input"]


def test_mapper_translates_prompt_messages_and_tools_into_responses_payload() -> None:
    """确认 mapper 会把 prompt/tool 历史转换成 Responses input/items。"""

    llm = Sub2apiPluginLargeLanguageModel([])
    payload = llm.build_responses_request_payload(
        model="gpt-4.1-mini",
        credentials={},
        prompt_messages=[
            SystemPromptMessage(content="你是可靠的工具编排助手。"),
            UserPromptMessage(content="帮我查询杭州天气"),
            AssistantPromptMessage(
                content="我先调用天气工具。",
                tool_calls=[
                    AssistantPromptMessage.ToolCall(
                        id="call_weather",
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name="get_weather",
                            arguments='{"city":"杭州"}',
                        ),
                    )
                ],
            ),
            ToolPromptMessage(content='{"temperature":25}', tool_call_id="call_weather"),
            AssistantPromptMessage(content="杭州当前 25 摄氏度。"),
        ],
        model_parameters={"temperature": 0.3, "max_tokens": 256},
        tools=[
            PromptMessageTool(
                name="get_weather",
                description="查询城市天气",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"}
                    },
                    "required": ["city"],
                },
            )
        ],
        stop=["STOP"],
        stream=True,
        user="user-123",
    )

    assert payload == {
        "model": "gpt-4.1-mini",
        "stream": True,
        "temperature": 0.3,
        "max_output_tokens": 256,
        "stop": ["STOP"],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "查询城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"}
                    },
                    "required": ["city"],
                },
            }
        ],
        "tool_choice": "auto",
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "你是可靠的工具编排助手。"}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "帮我查询杭州天气"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "我先调用天气工具。"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": '[tool_call id=call_weather name=get_weather]\n{"city":"杭州"}',
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": '[tool_result id=call_weather name=get_weather]\n{"temperature":25}',
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "杭州当前 25 摄氏度。"}],
            },
        ],
    }
    assert "instructions" not in payload
    assert "previous_response_id" not in payload
    assert "user" not in payload


def test_mapper_always_uses_dify_model_argument() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    payload = llm.build_responses_request_payload(
        model="gpt-4.1-mini",
        credentials={},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert payload["model"] == "gpt-4.1-mini"


def test_mapper_omits_tool_choice_when_no_tool_definitions() -> None:
    """确认没有工具定义时不会向 Responses 发送 tools/tool_choice。"""

    llm = Sub2apiPluginLargeLanguageModel([])
    payload = llm.build_responses_request_payload(
        model="gpt-4.1-mini",
        credentials={},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={},
        tools=[],
        stop=None,
        stream=False,
        user=None,
    )

    assert payload == {
        "model": "gpt-4.1-mini",
        "stream": False,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }


def test_mapper_deduplicates_duplicate_tool_results_by_call_id() -> None:
    """确认重复 tool result 会按 call_id 去重，避免生成重复 output item。"""

    llm = Sub2apiPluginLargeLanguageModel([])
    payload = llm.build_responses_request_payload(
        model="gpt-4.1-mini",
        credentials={},
        prompt_messages=[
            AssistantPromptMessage(
                tool_calls=[
                    AssistantPromptMessage.ToolCall(
                        id="call_weather",
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name="get_weather",
                            arguments='{"city":"杭州"}',
                        ),
                    )
                ]
            ),
            ToolPromptMessage(content="first", tool_call_id="call_weather"),
            ToolPromptMessage(content="second", tool_call_id="call_weather"),
        ],
        model_parameters={},
        tools=None,
        stop=None,
        stream=True,
        user=None,
    )

    assert payload["input"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": '[tool_call id=call_weather name=get_weather]\n{"city":"杭州"}',
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "[tool_result id=call_weather name=get_weather]\nfirst",
                }
            ],
        },
    ]


def test_mapper_raises_for_orphan_tool_result() -> None:
    """确认没有对应 tool call 历史时会明确抛错。"""

    llm = Sub2apiPluginLargeLanguageModel([])

    try:
        llm.build_responses_request_payload(
            model="gpt-4.1-mini",
            credentials={},
            prompt_messages=[ToolPromptMessage(content="orphan", tool_call_id="call_missing")],
            model_parameters={},
            tools=None,
            stop=None,
            stream=True,
            user=None,
        )
    except ValueError as exc:
        assert str(exc) == "找不到 tool_call_id=call_missing 对应的 tool call 历史。"
    else:
        raise AssertionError("缺少 tool call 历史时应抛出 ValueError")


def test_mapper_keeps_first_turn_without_tool_result_in_native_shape() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    payload = llm.build_responses_request_payload(
        model="gpt-4.1-mini",
        credentials={},
        prompt_messages=[
            UserPromptMessage(content="现在几点"),
            AssistantPromptMessage(
                tool_calls=[
                    AssistantPromptMessage.ToolCall(
                        id="call_time",
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name="current_time",
                            arguments="{}",
                        ),
                    )
                ]
            ),
        ],
        model_parameters={},
        tools=None,
        stop=None,
        stream=True,
        user=None,
    )

    assert payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "现在几点"}],
        },
        {
            "type": "function_call",
            "call_id": "call_time",
            "name": "current_time",
            "arguments": "{}",
            "id": "call_time",
        },
    ]


def test_mapper_rewrites_gpt_5_4_reasoning_effort_and_verbosity_to_responses_shape() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    payload = llm.build_responses_request_payload(
        model="gpt-5.4",
        credentials={},
        prompt_messages=[UserPromptMessage(content="写一首关于代码的俳句")],
        model_parameters={
            "reasoning_effort": "low",
            "verbosity": "high",
        },
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert payload == {
        "model": "gpt-5.4",
        "stream": False,
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "high"},
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "写一首关于代码的俳句"}],
            }
        ],
    }


def test_mapper_rejects_sampling_parameters_when_gpt_5_4_reasoning_is_not_none() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    try:
        llm.build_responses_request_payload(
            model="gpt-5.4",
            credentials={},
            prompt_messages=[UserPromptMessage(content="hello")],
            model_parameters={
                "reasoning_effort": "medium",
                "temperature": 0.3,
            },
            tools=None,
            stop=None,
            stream=False,
            user=None,
        )
    except ValueError as exc:
        assert str(exc) == "gpt-5.4 仅在 reasoning_effort=none 时支持 temperature/top_p。"
    else:
        raise AssertionError("gpt-5.4 在非 none 推理档位下应拒绝 temperature/top_p")


def test_mapper_ignores_dify_default_sampling_parameters_for_gpt_5_4_reasoning_modes() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    payload = llm.build_responses_request_payload(
        model="gpt-5.4",
        credentials={},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={
            "reasoning_effort": "low",
            "temperature": 1,
            "top_p": 1,
        },
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert payload == {
        "model": "gpt-5.4",
        "stream": False,
        "reasoning": {"effort": "low"},
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }


def test_mapper_keeps_sampling_parameters_for_gpt_5_4_when_reasoning_effort_is_omitted() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    payload = llm.build_responses_request_payload(
        model="gpt-5.4",
        credentials={},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={
            "temperature": 0.4,
            "top_p": 0.8,
        },
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert payload == {
        "model": "gpt-5.4",
        "stream": False,
        "temperature": 0.4,
        "top_p": 0.8,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }


def test_mapper_applies_gpt_5_4_rules_to_versioned_model_name() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    payload = llm.build_responses_request_payload(
        model="gpt-5.4-2026-03-05",
        credentials={},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={
            "reasoning_effort": "xhigh",
            "verbosity": "medium",
        },
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert payload == {
        "model": "gpt-5.4-2026-03-05",
        "stream": False,
        "reasoning": {"effort": "xhigh"},
        "text": {"verbosity": "medium"},
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }


def test_mapper_maps_text_response_format_to_responses_text_format() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    payload = llm.build_responses_request_payload(
        model="gpt-4.1-mini",
        credentials={},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={"response_format": "text"},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert payload == {
        "model": "gpt-4.1-mini",
        "stream": False,
        "text": {"format": {"type": "text"}},
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }


def test_mapper_maps_json_object_response_format_to_responses_text_format() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    payload = llm.build_responses_request_payload(
        model="gpt-4.1-mini",
        credentials={},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={"response_format": "json_object"},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert payload == {
        "model": "gpt-4.1-mini",
        "stream": False,
        "text": {"format": {"type": "json_object"}},
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }


def test_mapper_maps_json_schema_to_responses_text_format() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    payload = llm.build_responses_request_payload(
        model="gpt-4.1-mini",
        credentials={},
        prompt_messages=[UserPromptMessage(content="提取城市和温度")],
        model_parameters={
            "response_format": "json_schema",
            "json_schema": '{"type":"object","properties":{"city":{"type":"string"},"temperature":{"type":"number"}},"required":["city","temperature"]}',
        },
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert payload == {
        "model": "gpt-4.1-mini",
        "stream": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "structured_output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "temperature": {"type": "number"},
                    },
                    "required": ["city", "temperature"],
                },
                "strict": True,
            }
        },
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "提取城市和温度"}],
            }
        ],
    }


def test_mapper_merges_gpt_5_4_verbosity_with_structured_output() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    payload = llm.build_responses_request_payload(
        model="gpt-5.4",
        credentials={},
        prompt_messages=[UserPromptMessage(content="提取摘要")],
        model_parameters={
            "response_format": "json_schema",
            "json_schema": '{"name":"summary","schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]},"strict":false}',
            "verbosity": "high",
        },
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert payload == {
        "model": "gpt-5.4",
        "stream": False,
        "text": {
            "verbosity": "high",
            "format": {
                "type": "json_schema",
                "name": "summary",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                "strict": False,
            },
        },
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "提取摘要"}],
            }
        ],
    }


def test_mapper_rejects_json_schema_without_json_schema_response_format() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    try:
        llm.build_responses_request_payload(
            model="gpt-4.1-mini",
            credentials={},
            prompt_messages=[UserPromptMessage(content="hello")],
            model_parameters={
                "response_format": "text",
                "json_schema": '{"type":"object"}',
            },
            tools=None,
            stop=None,
            stream=False,
            user=None,
        )
    except ValueError as exc:
        assert str(exc) == "json_schema 仅在 response_format=json_schema 时可用。"
    else:
        raise AssertionError("response_format 不是 json_schema 时应拒绝 json_schema 参数")


def test_mapper_rejects_invalid_json_schema_string() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])

    try:
        llm.build_responses_request_payload(
            model="gpt-4.1-mini",
            credentials={},
            prompt_messages=[UserPromptMessage(content="hello")],
            model_parameters={
                "response_format": "json_schema",
                "json_schema": "{not-json}",
            },
            tools=None,
            stop=None,
            stream=False,
            user=None,
        )
    except ValueError as exc:
        assert str(exc) == "not correct json_schema format: {not-json}"
    else:
        raise AssertionError("非法 json_schema 应抛出 ValueError")
