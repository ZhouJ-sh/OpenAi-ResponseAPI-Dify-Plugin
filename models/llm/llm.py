# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportImplicitOverride=false, reportIncompatibleMethodOverride=false, reportDeprecated=false, reportUnknownMemberType=false, reportUnnecessaryCast=false

import logging
import json
import hashlib
from collections.abc import Generator, Iterable, Iterator, Mapping
from typing import Optional, Protocol, Union, cast
from urllib import error, request
from urllib.parse import urlsplit, urlunsplit

from dify_plugin import LargeLanguageModel
from dify_plugin.config.logger_format import plugin_logger_handler
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelFeature,
    ParameterRule,
    ParameterType,
    ModelPropertyKey,
    ModelType,
)
from dify_plugin.entities.model.llm import LLMMode, LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageRole,
    PromptMessageTool,
    TextPromptMessageContent,
    ToolPromptMessage,
)
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(plugin_logger_handler)
logger.propagate = False

_GPT_5_4_REASONING_OPTIONS = ["none", "low", "medium", "high", "xhigh"]
_GPT_5_4_VERBOSITY_OPTIONS = ["low", "medium", "high"]
_GPT_5_4_ALLOWED_DEFAULT_TEMPERATURES = {0, 1}
_GPT_5_4_ALLOWED_DEFAULT_TOP_P = 1
_DEFAULT_STRUCTURED_OUTPUT_NAME = "structured_output"
_REMOTE_API_USER_AGENT = "Codex Desktop/0.108.0-alpha.12 (Mac OS 26.3.0; arm64) unknown (Codex Desktop; 26.305.950)"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _safe_endpoint_for_logging(endpoint_url: str) -> str:
    split_result = urlsplit(endpoint_url)
    return urlunsplit((split_result.scheme, split_result.netloc, split_result.path, "", ""))


def _summarize_content_part_for_logging(part: Mapping[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {"type": part.get("type")}
    text = part.get("text")
    if isinstance(text, str):
        summary["text_len"] = len(text)
        summary["text_sha"] = _hash_text(text)
    return summary


def _summarize_input_item_for_logging(item: Mapping[str, object]) -> dict[str, object]:
    item_type = item.get("type")
    if item_type == "function_call":
        arguments = item.get("arguments")
        serialized_arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True) if isinstance(arguments, dict) else str(arguments or "")
        return {
            "type": "function_call",
            "name": item.get("name"),
            "call_id": item.get("call_id"),
            "arguments_len": len(serialized_arguments),
            "arguments_sha": _hash_text(serialized_arguments),
        }

    if item_type == "function_call_output":
        output = str(item.get("output") or "")
        return {
            "type": "function_call_output",
            "call_id": item.get("call_id"),
            "output_len": len(output),
            "output_sha": _hash_text(output),
        }

    content = item.get("content")
    return {
        "role": item.get("role"),
        "content_parts": [
            _summarize_content_part_for_logging(cast(Mapping[str, object], part))
            for part in cast(list[object], content or [])
            if isinstance(part, Mapping)
        ],
    }


def _summarize_payload_for_logging(payload: Mapping[str, object]) -> dict[str, object]:
    serialized_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    input_items = cast(list[object], payload.get("input") or [])
    tools = cast(list[object], payload.get("tools") or [])
    return {
        "payload_sha": _hash_text(serialized_payload),
        "payload_bytes": len(serialized_payload.encode("utf-8")),
        "model": payload.get("model"),
        "stream": payload.get("stream"),
        "payload_keys": sorted(str(key) for key in payload.keys()),
        "tool_names": [
            cast(Mapping[str, object], tool).get("name")
            for tool in tools
            if isinstance(tool, Mapping)
        ],
        "input_items": [
            _summarize_input_item_for_logging(cast(Mapping[str, object], item))
            for item in input_items
            if isinstance(item, Mapping)
        ],
    }


def _summarize_error_body_for_logging(error_body: str) -> dict[str, object]:
    summary: dict[str, object] = {
        "body_len": len(error_body),
        "body_sha": _hash_text(error_body) if error_body else "",
    }
    if not error_body:
        return summary

    try:
        parsed = cast(object, json.loads(error_body))
    except json.JSONDecodeError:
        return summary

    if isinstance(parsed, Mapping):
        nested_error = parsed.get("error")
        error_payload = cast(Mapping[str, object], nested_error if isinstance(nested_error, Mapping) else parsed)
        error_message = error_payload.get("message")
        summary["error_type"] = error_payload.get("type")
        summary["error_code"] = error_payload.get("code")
        if isinstance(error_message, str):
            summary["error_message_len"] = len(error_message)
            summary["error_message_sha"] = _hash_text(error_message)
    return summary


class _ResponsesClientProtocol(Protocol):
    def create(self, **payload: object) -> object: ...


class _HTTPResponseProtocol(Protocol):
    def __enter__(self) -> "_HTTPResponseProtocol": ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...

    def read(self) -> bytes: ...

    def __iter__(self) -> Iterator[bytes]: ...


class _ResponsesHTTPClient:
    _responses_url: str
    _api_key: Optional[str]

    def __init__(self, endpoint_url: str, api_key: Optional[str]) -> None:
        self._responses_url = self._normalize_responses_url(endpoint_url)
        self._api_key = api_key

    def create(self, **payload: object) -> object:
        request_body = json.dumps(payload).encode("utf-8")
        stream = payload.get("stream") is True
        http_request = request.Request(
            self._responses_url,
            data=request_body,
            headers=self._build_headers(stream=stream),
            method="POST",
        )

        if stream:
            return self._iter_stream_response_lines(http_request)

        try:
            with cast(_HTTPResponseProtocol, request.urlopen(http_request, timeout=120)) as http_response:
                response_body = http_response.read().decode("utf-8")
        except error.HTTPError as exception:
            error_body = self._read_http_error_body(exception)
            logger.error(
                "sub2api responses http error url=%s stream=%s summary=%s error=%s",
                _safe_endpoint_for_logging(self._responses_url),
                stream,
                json.dumps(_summarize_payload_for_logging(cast(Mapping[str, object], payload)), ensure_ascii=False, sort_keys=True),
                json.dumps(_summarize_error_body_for_logging(error_body), ensure_ascii=False, sort_keys=True),
            )
            raise ValueError(error_body) from exception

        return cast(dict[str, object], json.loads(response_body))

    def _iter_stream_response_lines(self, http_request: request.Request) -> Generator[str, None, None]:
        request_body = cast(bytes, http_request.data or b"")
        try:
            with cast(_HTTPResponseProtocol, request.urlopen(http_request, timeout=120)) as http_response:
                for raw_line in http_response:
                    yield raw_line.decode("utf-8")
        except error.HTTPError as exception:
            error_body = self._read_http_error_body(exception)
            logger.error(
                "sub2api responses stream http error url=%s summary=%s error=%s",
                _safe_endpoint_for_logging(self._responses_url),
                json.dumps(
                    _summarize_payload_for_logging(cast(Mapping[str, object], json.loads(request_body.decode("utf-8") or "{}"))),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                json.dumps(_summarize_error_body_for_logging(error_body), ensure_ascii=False, sort_keys=True),
            )
            raise ValueError(error_body) from exception

    def _build_headers(self, *, stream: bool) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": _REMOTE_API_USER_AGENT,
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _normalize_responses_url(self, endpoint_url: str) -> str:
        normalized_url = endpoint_url.rstrip("/")
        if normalized_url.endswith("/v1/responses"):
            return normalized_url
        if normalized_url.endswith("/v1"):
            return f"{normalized_url}/responses"
        return f"{normalized_url}/v1/responses"

    def _read_http_error_body(self, exception: error.HTTPError) -> str:
        error_body = exception.read().decode("utf-8", errors="ignore").strip()
        if error_body:
            return error_body
        return f"HTTP {exception.code} {exception.reason}"


class Sub2apiPluginLargeLanguageModel(LargeLanguageModel):
    """sub2api 的 LLM provider，仅在当前阶段负责 Responses 请求映射与 runtime schema。"""

    class _ResponsesPendingToolCall:
        """保存流式 `function_call` 的增量状态，确保只在完成后输出完整 tool call。"""

        item_id: str
        call_id: str
        name: str
        arguments: str

        def __init__(self, item_id: str, call_id: str, name: str, arguments: str = "") -> None:
            self.item_id = item_id
            self.call_id = call_id
            self.name = name
            self.arguments = arguments

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """当前阶段尚未接入真实 HTTP 调用，因此先返回空映射以满足 SDK 抽象约束。"""

        return {}

    def _invoke(
        self,
        model: str,
        credentials: dict[str, object],
        prompt_messages: list[PromptMessage],
        model_parameters: dict[str, object],
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator[LLMResultChunk, None, None]]:
        """调用 `/v1/responses`，并把非流式/流式结果统一转换为 Dify LLM 输出。"""

        payload = self.build_responses_request_payload(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )
        client = cast(_ResponsesClientProtocol, self._create_responses_client(credentials))
        logger.info(
            "sub2api responses request endpoint=%s summary=%s",
            _safe_endpoint_for_logging(cast(str, credentials.get("endpoint_url") or "")),
            json.dumps(_summarize_payload_for_logging(payload), ensure_ascii=False, sort_keys=True),
        )
        response = client.create(**payload)

        if stream:
            return self._parse_responses_stream(
                model=model,
                credentials=credentials,
                prompt_messages=prompt_messages,
                response_lines=self._iter_sse_lines(response),
            )

        return self._parse_responses_response(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            response=cast(dict[str, object], response),
        )

    def _create_responses_client(self, credentials: dict[str, object]) -> _ResponsesClientProtocol:
        endpoint_url = cast(str, credentials.get("endpoint_url") or "")
        if not endpoint_url:
            raise ValueError("缺少 endpoint_url，无法调用 /v1/responses。")

        api_key = credentials.get("api_key")
        return _ResponsesHTTPClient(
            endpoint_url=endpoint_url,
            api_key=cast(Optional[str], api_key if isinstance(api_key, str) else None),
        )

    def _parse_responses_response(
        self,
        model: str,
        credentials: dict[str, object],
        prompt_messages: list[PromptMessage],
        response: dict[str, object],
    ) -> LLMResult:
        """解析非流式 Responses 响应，提取 assistant 文本、tool call 与 usage。"""

        self._raise_for_terminal_response_error(response)
        assistant_prompt_message = AssistantPromptMessage(
            content=self._extract_responses_output_text(cast(list[object], response.get("output", []))),
            tool_calls=self._extract_responses_output_tool_calls(cast(list[object], response.get("output", []))),
        )
        usage = cast(dict[str, object], response.get("usage", {}))

        return LLMResult(
            model=cast(str, response.get("model") or model),
            prompt_messages=prompt_messages,
            message=assistant_prompt_message,
            usage=self._calc_response_usage(
                model,
                credentials,
                self._coerce_int(usage.get("input_tokens", 0)),
                self._coerce_int(usage.get("output_tokens", 0)),
            ),
            system_fingerprint=cast(Optional[str], response.get("id")),
        )

    def _parse_responses_stream(
        self,
        model: str,
        credentials: dict[str, object],
        prompt_messages: list[PromptMessage],
        response_lines: Iterable[str],
    ) -> Generator[LLMResultChunk, None, None]:
        """按状态机解析 Responses SSE，只在文本增量与完整 tool call 可用时产出 chunk。"""

        index = 0
        response_id: Optional[str] = None
        usage_payload: dict[str, object] = {}
        pending_tool_calls: dict[str, Sub2apiPluginLargeLanguageModel._ResponsesPendingToolCall] = {}

        for event in self._iter_sse_events(response_lines):
            event_type = cast(str, event.get("type", ""))
            if event_type == "response.created":
                response_payload = cast(dict[str, object], event.get("response", {}))
                response_id = cast(Optional[str], response_payload.get("id"))
                continue

            if event_type == "response.in_progress":
                response_payload = cast(dict[str, object], event.get("response", {}))
                response_id = cast(Optional[str], response_payload.get("id") or response_id)
                continue

            if event_type == "response.output_item.added":
                item = cast(dict[str, object], event.get("item", {}))
                if item.get("type") == "function_call":
                    item_id = cast(str, item.get("id") or event.get("item_id") or "")
                    pending_tool_calls[item_id] = self._ResponsesPendingToolCall(
                        item_id=item_id,
                        call_id=cast(str, item.get("call_id") or item_id),
                        name=cast(str, item.get("name") or ""),
                        arguments=cast(str, item.get("arguments") or ""),
                    )
                continue

            if event_type == "response.output_text.delta":
                delta_text = cast(str, event.get("delta") or "")
                if delta_text:
                    yield self._build_chunk(model, prompt_messages, index, delta_text, [], None, cast(Optional[str], event.get("item_id")))
                    index += 1
                continue

            if event_type == "response.output_text.done":
                continue

            if event_type == "keepalive":
                continue

            if event_type in {"response.content_part.added", "response.content_part.done"}:
                continue

            if event_type == "response.function_call_arguments.delta":
                pending = self._get_pending_tool_call(pending_tool_calls, event)
                if pending is not None:
                    # 参数 delta 只能累计字符串，严禁提前发出半成品 tool call。
                    pending.arguments += cast(str, event.get("delta") or "")
                continue

            if event_type == "response.function_call_arguments.done":
                pending = self._get_pending_tool_call(pending_tool_calls, event)
                if pending is not None:
                    pending.arguments = cast(str, event.get("arguments") or pending.arguments)
                continue

            if event_type == "response.output_item.done":
                item = cast(dict[str, object], event.get("item", {}))
                if item.get("type") == "function_call":
                    item_id = cast(str, item.get("id") or event.get("item_id") or "")
                    pending = pending_tool_calls.pop(item_id, None)
                    yield self._build_chunk(
                        model,
                        prompt_messages,
                        index,
                        "",
                        [
                            self._build_tool_call(
                                call_id=cast(str, item.get("call_id") or (pending.call_id if pending else item_id)),
                                name=cast(str, item.get("name") or (pending.name if pending else "")),
                                arguments=cast(str, item.get("arguments") or (pending.arguments if pending else "") or "{}"),
                            )
                        ],
                        None,
                        cast(Optional[str], item.get("id")),
                    )
                    index += 1
                continue

            if event_type in {"response.completed", "response.incomplete", "response.failed"}:
                response_payload = cast(dict[str, object], event.get("response", {}))
                response_id = cast(Optional[str], response_payload.get("id") or response_id)
                usage_payload = cast(dict[str, object], response_payload.get("usage", {}))
                incomplete_details = response_payload.get("incomplete_details")
                incomplete_reason = cast(Mapping[str, object], incomplete_details).get("reason") if isinstance(incomplete_details, Mapping) else None
                logger.info(
                    "sub2api responses stream terminal event=%s response_id=%s error=%s incomplete=%s",
                    event_type,
                    response_id,
                    json.dumps(_summarize_error_body_for_logging(json.dumps(cast(dict[str, object], response_payload.get("error", {})), ensure_ascii=False)), ensure_ascii=False, sort_keys=True),
                    json.dumps({"reason": incomplete_reason}, ensure_ascii=False, sort_keys=True),
                )
                self._raise_for_terminal_response_error(response_payload)
                if event_type == "response.completed":
                    yield self._build_chunk(
                        model,
                        prompt_messages,
                        index,
                        "",
                        [],
                        self._calc_response_usage(
                            model,
                            credentials,
                            self._coerce_int(usage_payload.get("input_tokens", 0)),
                            self._coerce_int(usage_payload.get("output_tokens", 0)),
                        ),
                        response_id,
                        finish_reason="stop",
                    )
                return

            if event_type == "error":
                error_payload = cast(dict[str, object], event.get("error", {}))
                logger.error(
                    "sub2api responses stream error event response_id=%s error=%s",
                    response_id,
                    json.dumps(_summarize_error_body_for_logging(json.dumps(error_payload, ensure_ascii=False)), ensure_ascii=False, sort_keys=True),
                )
                raise ValueError(cast(str, error_payload.get("message") or error_payload.get("code") or "Responses 流请求失败"))

            logger.error(
                "sub2api responses unsupported stream event type=%s response_id=%s",
                event_type,
                response_id,
            )
            raise ValueError(f"不支持的 Responses 流事件类型: {event_type}")

        if usage_payload:
            yield self._build_chunk(
                model,
                prompt_messages,
                index,
                "",
                [],
                self._calc_response_usage(
                    model,
                    credentials,
                    self._coerce_int(usage_payload.get("input_tokens", 0)),
                    self._coerce_int(usage_payload.get("output_tokens", 0)),
                ),
                response_id,
                finish_reason="stop",
            )

    def _iter_sse_lines(self, response: object) -> Iterable[str]:
        """把 mock/string/逐行迭代器统一折叠为 SSE 文本行序列。"""

        if isinstance(response, str):
            return response.splitlines()
        if isinstance(response, bytes):
            return response.decode("utf-8").splitlines()
        return cast(Iterable[str], response)

    def _iter_sse_events(self, response_lines: Iterable[str]) -> Generator[dict[str, object], None, None]:
        """解析 `data:` 行并跳过空行与 `[DONE]`，为状态机提供结构化事件。"""

        for raw_line in response_lines:
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            yield cast(dict[str, object], json.loads(data))

    def _raise_for_terminal_response_error(self, response: dict[str, object]) -> None:
        """把 `failed` / `incomplete` 终态转换为明确异常，避免上层误把半成品当成功结果。"""

        status = cast(str, response.get("status") or "")
        if status == "failed":
            error = cast(dict[str, object], response.get("error", {}))
            logger.error(
                "sub2api responses failed response_id=%s error=%s",
                response.get("id"),
                json.dumps(_summarize_error_body_for_logging(json.dumps(error, ensure_ascii=False)), ensure_ascii=False, sort_keys=True),
            )
            raise ValueError(cast(str, error.get("message") or error.get("code") or "Responses 请求失败"))
        if status == "incomplete":
            raw_details = response.get("incomplete_details")
            details = cast(dict[str, object], raw_details if isinstance(raw_details, Mapping) else {})
            logger.error(
                "sub2api responses incomplete response_id=%s details=%s",
                response.get("id"),
                json.dumps({"reason": details.get("reason")}, ensure_ascii=False, sort_keys=True),
            )
            raise ValueError(cast(str, details.get("reason") or "Responses 请求未完成"))

    def _extract_responses_output_text(self, output_items: list[object]) -> str:
        """从 `ResponsesResponse.output` 中提取并拼接 assistant 文本。"""

        text_parts: list[str] = []
        for raw_item in output_items:
            item = cast(dict[str, object], raw_item)
            item_type = item.get("type")
            if item_type in {"output_text", "text"} and isinstance(item.get("text"), str):
                text_parts.append(cast(str, item["text"]))
            if item_type != "message":
                continue
            for raw_part in cast(list[object], item.get("content", [])):
                part = cast(dict[str, object], raw_part)
                if part.get("type") in {"output_text", "text", "input_text"} and isinstance(part.get("text"), str):
                    text_parts.append(cast(str, part["text"]))
        return "".join(text_parts)

    def _extract_responses_output_tool_calls(
        self,
        output_items: list[object],
    ) -> list[AssistantPromptMessage.ToolCall]:
        """从非流式 `output` 中抽取 function_call，并转换为 Dify assistant tool_calls。"""

        tool_calls: list[AssistantPromptMessage.ToolCall] = []
        for raw_item in output_items:
            item = cast(dict[str, object], raw_item)
            if item.get("type") != "function_call":
                continue
            arguments = item.get("arguments", "{}")
            tool_calls.append(
                self._build_tool_call(
                    call_id=cast(str, item.get("call_id") or item.get("id") or ""),
                    name=cast(str, item.get("name") or ""),
                    arguments=json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else cast(str, arguments),
                )
            )
        return tool_calls

    def _get_pending_tool_call(
        self,
        pending_tool_calls: dict[str, _ResponsesPendingToolCall],
        event: dict[str, object],
    ) -> Optional[_ResponsesPendingToolCall]:
        """通过 `item_id` 优先定位流式 tool call 状态，兼容少数字段变体。"""

        item_id = cast(str, event.get("item_id") or event.get("call_id") or "")
        return pending_tool_calls.get(item_id)

    def _build_tool_call(self, call_id: str, name: str, arguments: str) -> AssistantPromptMessage.ToolCall:
        """构造 Dify `AssistantPromptMessage.ToolCall`，统一 tool call 序列化入口。"""

        return AssistantPromptMessage.ToolCall(
            id=call_id,
            type="function",
            function=AssistantPromptMessage.ToolCall.ToolCallFunction(name=name, arguments=arguments),
        )

    def _build_chunk(
        self,
        model: str,
        prompt_messages: list[PromptMessage],
        index: int,
        content: str,
        tool_calls: list[AssistantPromptMessage.ToolCall],
        usage: Optional[LLMUsage],
        system_fingerprint: Optional[str],
        finish_reason: Optional[str] = None,
    ) -> LLMResultChunk:
        """统一构造 chunk，避免流式各分支重复拼装 `LLMResultChunkDelta`。"""

        return LLMResultChunk(
            model=model,
            prompt_messages=prompt_messages,
            system_fingerprint=system_fingerprint,
            delta=LLMResultChunkDelta(
                index=index,
                message=AssistantPromptMessage(content=content, tool_calls=tool_calls),
                usage=usage,
                finish_reason=finish_reason,
            ),
        )

    def build_responses_request_payload(
        self,
        model: str,
        credentials: dict[str, object],
        prompt_messages: list[PromptMessage],
        model_parameters: dict[str, object],
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> dict[str, object]:
        """把 Dify 的 LLM 调用参数映射为 OpenAI Responses request payload。"""

        filtered_model_parameters = {
            key: value for key, value in model_parameters.items() if value is not None and key != "tool_choice"
        }
        normalized_model_parameters = self._normalize_responses_model_parameters(
            model=model,
            model_parameters=filtered_model_parameters,
        )
        uses_tool_result_compatibility = any(isinstance(message, ToolPromptMessage) for message in prompt_messages)
        payload: dict[str, object] = {
            "model": model,
            "stream": stream,
            "input": self._convert_prompt_messages_to_sub2api_compatible_input(prompt_messages)
            if uses_tool_result_compatibility
            else self._convert_prompt_messages_to_responses_input(prompt_messages),
            **normalized_model_parameters,
        }

        if "max_tokens" in payload:
            payload["max_output_tokens"] = payload.pop("max_tokens")

        if stop:
            payload["stop"] = stop

        del user

        if tools:
            payload["tools"] = self._convert_prompt_tools_to_responses_tools(tools)
            payload["tool_choice"] = "auto"

        return payload

    def _normalize_responses_model_parameters(
        self,
        model: str,
        model_parameters: dict[str, object],
    ) -> dict[str, object]:
        normalized_model_parameters = dict(model_parameters)
        text_config = self._normalize_responses_text_config(normalized_model_parameters)

        if self._is_gpt_5_4_model(model):
            reasoning_effort = normalized_model_parameters.pop("reasoning_effort", None)
            verbosity = normalized_model_parameters.pop("verbosity", None)
            if isinstance(reasoning_effort, str):
                if reasoning_effort != "none":
                    self._drop_implicit_gpt_5_4_sampling_defaults(normalized_model_parameters)
                self._validate_gpt_5_4_sampling_parameter_compatibility(
                    reasoning_effort=reasoning_effort,
                    model_parameters=normalized_model_parameters,
                )
                normalized_model_parameters["reasoning"] = {"effort": reasoning_effort}
            if isinstance(verbosity, str):
                text_config["verbosity"] = verbosity
        else:
            normalized_model_parameters.pop("reasoning_effort", None)
            normalized_model_parameters.pop("verbosity", None)

        if text_config:
            normalized_model_parameters["text"] = text_config

        return normalized_model_parameters

    def _normalize_responses_text_config(self, model_parameters: dict[str, object]) -> dict[str, object]:
        text_config = dict(cast(Mapping[str, object], model_parameters.pop("text", {})))
        response_format = model_parameters.pop("response_format", None)
        json_schema = model_parameters.pop("json_schema", None)

        if json_schema is not None and response_format != "json_schema":
            raise ValueError("json_schema 仅在 response_format=json_schema 时可用。")

        if response_format == "text":
            text_config["format"] = {"type": "text"}
        elif response_format == "json_object":
            text_config["format"] = {"type": "json_object"}
        elif response_format == "json_schema":
            text_config["format"] = self._build_responses_json_schema_format(json_schema)
        elif isinstance(response_format, str) and response_format:
            text_config["format"] = {"type": response_format}

        return text_config

    def _build_responses_json_schema_format(self, json_schema: object) -> dict[str, object]:
        if json_schema is None:
            raise ValueError("Must define JSON Schema when the response format is json_schema")

        parsed_schema: object = json_schema
        if isinstance(json_schema, str):
            try:
                parsed_schema = json.loads(json_schema)
            except json.JSONDecodeError as exc:
                raise ValueError(f"not correct json_schema format: {json_schema}") from exc

        if not isinstance(parsed_schema, Mapping):
            raise ValueError(f"not correct json_schema format: {json_schema}")

        normalized_schema_payload = dict(parsed_schema)
        if normalized_schema_payload.get("type") == "json_schema":
            normalized_schema_payload.pop("type", None)

        wrapped_schema = normalized_schema_payload.get("schema")
        if isinstance(wrapped_schema, Mapping):
            schema = dict(wrapped_schema)
            name = normalized_schema_payload.get("name")
            strict = normalized_schema_payload.get("strict", True)
        else:
            schema = dict(normalized_schema_payload)
            name = normalized_schema_payload.get("name", _DEFAULT_STRUCTURED_OUTPUT_NAME)
            strict = normalized_schema_payload.get("strict", True)

        return {
            "type": "json_schema",
            "name": name if isinstance(name, str) and name else _DEFAULT_STRUCTURED_OUTPUT_NAME,
            "schema": schema,
            "strict": strict if isinstance(strict, bool) else True,
        }

    def _validate_gpt_5_4_sampling_parameter_compatibility(
        self,
        reasoning_effort: str,
        model_parameters: Mapping[str, object],
    ) -> None:
        if reasoning_effort == "none":
            return
        if "temperature" in model_parameters or "top_p" in model_parameters:
            raise ValueError("gpt-5.4 仅在 reasoning_effort=none 时支持 temperature/top_p。")

    def _drop_implicit_gpt_5_4_sampling_defaults(self, model_parameters: dict[str, object]) -> None:
        temperature = model_parameters.get("temperature")
        if temperature in _GPT_5_4_ALLOWED_DEFAULT_TEMPERATURES:
            del model_parameters["temperature"]

        top_p = model_parameters.get("top_p")
        if top_p == _GPT_5_4_ALLOWED_DEFAULT_TOP_P:
            del model_parameters["top_p"]

    def _build_parameter_rules_for_model(self, model: str) -> list[ParameterRule]:
        response_format_options = ["text", "json_schema"] if self._is_gpt_5_4_model(model) else ["text", "json_object", "json_schema"]
        parameter_rules = [
            ParameterRule(
                name="temperature",
                use_template="temperature",
                label=I18nObject(zh_Hans="温度", en_US="Temperature"),
                type=ParameterType.FLOAT,
                help=I18nObject(
                    zh_Hans="仅在推理努力程度为 none 时建议设置。若推理努力程度不是 none，OpenAI GPT-5.4 Responses API 不支持 temperature。",
                    en_US="Recommended only when reasoning effort is none. When reasoning effort is not none, the OpenAI GPT-5.4 Responses API does not support temperature.",
                ),
            ),
            ParameterRule(
                name="top_p",
                use_template="top_p",
                label=I18nObject(zh_Hans="Top P", en_US="Top P"),
                type=ParameterType.FLOAT,
                help=I18nObject(
                    zh_Hans="仅在推理努力程度为 none 时建议设置。若推理努力程度不是 none，OpenAI GPT-5.4 Responses API 不支持 top_p。",
                    en_US="Recommended only when reasoning effort is none. When reasoning effort is not none, the OpenAI GPT-5.4 Responses API does not support top_p.",
                ),
            ),
            ParameterRule(
                name="frequency_penalty",
                use_template="frequency_penalty",
                label=I18nObject(zh_Hans="频率惩罚", en_US="Frequency Penalty"),
                type=ParameterType.FLOAT,
                help=I18nObject(
                    zh_Hans="当前插件会按顶层字段透传该参数；OpenAI 官方文档尚未明确说明它会像 temperature/top_p 一样受推理努力程度限制。",
                    en_US="The plugin currently forwards this parameter as a top-level field. OpenAI official docs do not yet clearly state that it is restricted by reasoning effort in the same way as temperature/top_p.",
                ),
            ),
            ParameterRule(
                name="presence_penalty",
                use_template="presence_penalty",
                label=I18nObject(zh_Hans="存在惩罚", en_US="Presence Penalty"),
                type=ParameterType.FLOAT,
                help=I18nObject(
                    zh_Hans="当前插件会按顶层字段透传该参数；OpenAI 官方文档尚未明确说明它会像 temperature/top_p 一样受推理努力程度限制。",
                    en_US="The plugin currently forwards this parameter as a top-level field. OpenAI official docs do not yet clearly state that it is restricted by reasoning effort in the same way as temperature/top_p.",
                ),
            ),
            ParameterRule(
                name="max_tokens",
                use_template="max_tokens",
            ),
            ParameterRule(
                name="response_format",
                label=I18nObject(zh_Hans="回复格式", en_US="Response Format"),
                type=ParameterType.STRING,
                help=I18nObject(
                    zh_Hans="指定模型必须输出的格式。",
                    en_US="Specifies the output format the model must follow.",
                ),
                options=response_format_options,
            ),
            ParameterRule(
                name="json_schema",
                use_template="json_schema",
            ),
        ]

        if not self._is_gpt_5_4_model(model):
            return parameter_rules

        return parameter_rules + [
            ParameterRule(
                name="reasoning_effort",
                label=I18nObject(zh_Hans="推理努力程度", en_US="Reasoning Effort"),
                type=ParameterType.STRING,
                help=I18nObject(
                    zh_Hans="约束 gpt-5.4 的推理努力程度。支持 none、low、medium、high、xhigh；none 为低延迟模式。若取值不是 none，temperature 与 top_p 不应设置；frequency_penalty 与 presence_penalty 当前仍按顶层透传。",
                    en_US="Controls reasoning effort for gpt-5.4. Supported values are none, low, medium, high, and xhigh; none is the low-latency mode. When the value is not none, temperature and top_p should not be set; frequency_penalty and presence_penalty are currently still forwarded as top-level fields.",
                ),
                default="none",
                options=list(_GPT_5_4_REASONING_OPTIONS),
            ),
            ParameterRule(
                name="verbosity",
                label=I18nObject(zh_Hans="详细程度", en_US="Verbosity"),
                type=ParameterType.STRING,
                help=I18nObject(
                    zh_Hans="约束 gpt-5.4 响应的详细程度。支持 low、medium、high，运行时会映射到 Responses API 的 text.verbosity。",
                    en_US="Controls response verbosity for gpt-5.4. Supported values are low, medium, and high. The runtime maps it to text.verbosity for the Responses API.",
                ),
                default="medium",
                options=list(_GPT_5_4_VERBOSITY_OPTIONS),
            ),
        ]

    def _is_gpt_5_4_model(self, model: str) -> bool:
        return model == "gpt-5.4" or model.startswith("gpt-5.4-")

    def _convert_prompt_messages_to_sub2api_compatible_input(
        self,
        prompt_messages: list[PromptMessage],
    ) -> list[dict[str, object]]:
        input_items: list[dict[str, object]] = []
        known_tool_calls: dict[str, AssistantPromptMessage.ToolCall] = {}
        emitted_tool_result_ids: set[str] = set()

        for message in prompt_messages:
            if isinstance(message, ToolPromptMessage):
                compatibility_item = self._convert_tool_message_to_compatibility_role_item(
                    message=message,
                    known_tool_calls=known_tool_calls,
                    emitted_tool_result_ids=emitted_tool_result_ids,
                )
                if compatibility_item is not None:
                    input_items.append(compatibility_item)
                continue

            role_item = self._convert_prompt_message_to_role_input_item(message)
            if role_item is not None:
                input_items.append(role_item)

            if isinstance(message, AssistantPromptMessage) and message.tool_calls:
                input_items.extend(self._convert_assistant_tool_calls_to_compatibility_items(message.tool_calls))
                for tool_call in message.tool_calls:
                    call_id = tool_call.id or tool_call.function.name
                    if call_id:
                        known_tool_calls[call_id] = tool_call

        return input_items

    def _convert_prompt_messages_to_responses_input(self, prompt_messages: list[PromptMessage]) -> list[dict[str, object]]:
        """按时间顺序把 Dify transcript 映射为 Responses `input` 数组。"""

        input_items: list[dict[str, object]] = []
        known_tool_call_ids: set[str] = set()
        emitted_tool_result_ids: set[str] = set()

        for message in prompt_messages:
            if isinstance(message, ToolPromptMessage):
                tool_result_item = self._convert_tool_message_to_function_call_output(
                    message=message,
                    known_tool_call_ids=known_tool_call_ids,
                    emitted_tool_result_ids=emitted_tool_result_ids,
                )
                if tool_result_item is not None:
                    input_items.append(tool_result_item)
                continue

            role_item = self._convert_prompt_message_to_role_input_item(message)
            if role_item is not None:
                input_items.append(role_item)

            if isinstance(message, AssistantPromptMessage):
                for tool_call_item in self._convert_assistant_tool_calls_to_function_calls(message.tool_calls):
                    known_tool_call_ids.add(cast(str, tool_call_item["call_id"]))
                    input_items.append(tool_call_item)

        return input_items

    def _convert_prompt_message_to_role_input_item(self, message: PromptMessage) -> Optional[dict[str, object]]:
        """把 system/user/assistant/developer 消息映射为 Responses role item。"""

        if message.role == PromptMessageRole.TOOL:
            return None

        role = message.role.value
        if message.role == PromptMessageRole.DEVELOPER:
            # Responses input 不支持 developer role，因此收敛为 system transcript item。
            role = PromptMessageRole.SYSTEM.value

        content_parts = self._convert_prompt_message_content_to_responses_parts(
            message=message,
            use_output_text=message.role == PromptMessageRole.ASSISTANT,
        )
        if not content_parts:
            return None

        return {
            "role": role,
            "content": content_parts,
        }

    def _convert_prompt_message_content_to_responses_parts(
        self,
        message: PromptMessage,
        use_output_text: bool,
    ) -> list[dict[str, object]]:
        """把 Dify message content 转为 Responses content parts。"""

        if isinstance(message.content, str):
            if message.content == "":
                return []
            return [
                {
                    "type": "output_text" if use_output_text else "input_text",
                    "text": message.content,
                }
            ]

        if not message.content:
            return []

        parts: list[dict[str, object]] = []
        for content in message.content:
            if isinstance(content, TextPromptMessageContent):
                parts.append(
                    {
                        "type": "output_text" if use_output_text else "input_text",
                        "text": content.data,
                    }
                )
                continue

            raise ValueError("当前仅支持文本 content 映射到 Responses input。")

        return parts

    def _convert_prompt_tools_to_responses_tools(self, tools: list[PromptMessageTool]) -> list[dict[str, object]]:
        """把 Dify 的 PromptMessageTool 定义映射为 Responses `tools[]`。"""

        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": cast(object, tool.parameters),
            }
            for tool in tools
        ]

    def _convert_assistant_tool_calls_to_function_calls(
        self,
        tool_calls: list[AssistantPromptMessage.ToolCall],
    ) -> list[dict[str, object]]:
        """把 assistant 历史 tool call 转成 Responses `function_call` items。"""

        items: list[dict[str, object]] = []
        for tool_call in tool_calls:
            call_id = tool_call.id or tool_call.function.name
            if not call_id:
                raise ValueError("tool call 缺少可用的 call_id。")

            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                    "id": call_id,
                }
            )

        return items

    def _convert_assistant_tool_calls_to_compatibility_items(
        self,
        tool_calls: list[AssistantPromptMessage.ToolCall],
    ) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for tool_call in tool_calls:
            call_id = tool_call.id or tool_call.function.name
            if not call_id:
                raise ValueError("tool call 缺少可用的 call_id。")
            items.append(
                {
                    "role": PromptMessageRole.ASSISTANT.value,
                    "content": [
                        {
                            "type": "output_text",
                            "text": self._format_tool_call_compatibility_text(
                                call_id=call_id,
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                        }
                    ],
                }
            )
        return items

    def _convert_tool_message_to_function_call_output(
        self,
        message: ToolPromptMessage,
        known_tool_call_ids: set[str],
        emitted_tool_result_ids: set[str],
    ) -> Optional[dict[str, object]]:
        """把 tool 历史结果转成 Responses `function_call_output`，并处理孤儿/重复场景。"""

        call_id = message.tool_call_id
        if call_id not in known_tool_call_ids:
            raise ValueError(f"找不到 tool_call_id={call_id} 对应的 tool call 历史。")

        if call_id in emitted_tool_result_ids:
            return None

        emitted_tool_result_ids.add(call_id)
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": self._extract_text_from_message_content(message.content),
        }

    def _convert_tool_message_to_compatibility_role_item(
        self,
        message: ToolPromptMessage,
        known_tool_calls: dict[str, AssistantPromptMessage.ToolCall],
        emitted_tool_result_ids: set[str],
    ) -> Optional[dict[str, object]]:
        call_id = message.tool_call_id
        tool_call = known_tool_calls.get(call_id)
        if tool_call is None:
            raise ValueError(f"找不到 tool_call_id={call_id} 对应的 tool call 历史。")

        if call_id in emitted_tool_result_ids:
            return None

        emitted_tool_result_ids.add(call_id)
        return {
            "role": PromptMessageRole.ASSISTANT.value,
            "content": [
                {
                    "type": "output_text",
                    "text": self._format_tool_result_compatibility_text(
                        call_id=call_id,
                        name=tool_call.function.name,
                        output=self._extract_text_from_message_content(message.content),
                    ),
                }
            ],
        }

    def _format_tool_call_compatibility_text(self, call_id: str, name: str, arguments: str) -> str:
        return f"[tool_call id={call_id} name={name}]\n{arguments}"

    def _format_tool_result_compatibility_text(self, call_id: str, name: str, output: str) -> str:
        return f"[tool_result id={call_id} name={name}]\n{output}"

    def _extract_text_from_message_content(self, content: object) -> str:
        """从 PromptMessage 的 content 中提取纯文本，供 function_call_output 使用。"""

        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return ""

        text_parts: list[str] = []
        for item in cast(list[object], content):
            if isinstance(item, TextPromptMessageContent):
                text_parts.append(cast(TextPromptMessageContent, item).data)

        return "".join(text_parts)

    def get_num_tokens(
        self,
        model: str,
        credentials: dict[str, object],
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """当前任务不实现真实 token 估算，因此保持最小占位实现。"""

        return 0

    def validate_credentials(self, model: str, credentials: Mapping[str, object]) -> None:
        """当前阶段不做远程校验，仅保留统一异常包装约定。"""

        try:
            pass
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def get_customizable_model_schema(self, model: str, credentials: Mapping[str, object]) -> AIModelEntity:
        """显式声明 Responses/chat 模式下的 runtime schema。"""

        context_size = credentials.get("context_size", 4096)
        entity = AIModelEntity(
            model=model,
            label=I18nObject(zh_Hans=model, en_US=model),
            model_type=ModelType.LLM,
            features=[ModelFeature.MULTI_TOOL_CALL, ModelFeature.STREAM_TOOL_CALL],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: self._coerce_int(context_size),
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
            },
            parameter_rules=self._build_parameter_rules_for_model(model),
        )

        return entity

    def _coerce_int(self, value: object) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value:
            return int(value)
        return 0
