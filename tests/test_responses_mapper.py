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
