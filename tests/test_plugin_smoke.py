# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false

"""基于 PluginRunner 的真实插件级 smoke 测试。"""

from __future__ import annotations

import importlib
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Protocol, TypeVar, cast, override

from dify_plugin.config.integration_config import IntegrationConfig
from dify_plugin.core.entities.plugin.request import (
    ModelActions,
    ModelInvokeLLMRequest,
    PluginInvokeType,
)
from dify_plugin.entities.model import ModelType
from dify_plugin.entities.model.llm import LLMResultChunk
from dify_plugin.entities.model.message import PromptMessageTool, UserPromptMessage
from dify_plugin.integration.run import PluginRunner


_F = TypeVar("_F", bound=Callable[..., object])


class _PytestMarkProtocol(Protocol):
    def xfail(self, *, raises: type[BaseException], strict: bool, reason: str) -> Callable[[_F], _F]: ...


class _PytestProtocol(Protocol):
    mark: _PytestMarkProtocol

    def skip(self, reason: str) -> None: ...

    def xfail(self, reason: str) -> None: ...


pytest = cast(_PytestProtocol, cast(object, importlib.import_module("pytest")))

from models.llm.llm import _ResponsesHTTPClient


class _RecordedRequest:
    """记录 fake Responses 服务端收到的请求，供 smoke 断言协议路径与 payload。"""

    path: str
    payload: dict[str, object]

    def __init__(self, path: str, payload: dict[str, object]) -> None:
        self.path = path
        self.payload = payload


class _FakeResponsesHandler(BaseHTTPRequestHandler):
    """提供最小 `/v1/responses` 假服务，只覆盖本任务 smoke 所需分支。"""

    non_stream_response: dict[str, object] = {}
    stream_response_lines: list[str] = []
    recorded_requests: list[_RecordedRequest] = []

    def do_POST(self) -> None:  # noqa: N802
        """接收插件请求并按 stream 开关返回非流式 JSON 或 SSE。"""

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        payload = cast(dict[str, object], json.loads(raw_body.decode("utf-8")))
        self.recorded_requests.append(_RecordedRequest(path=self.path, payload=payload))

        if self.path != "/v1/responses":
            self.send_response(404)
            self.end_headers()
            _ = self.wfile.write(b"unsupported path")
            return

        if payload.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for line in self.stream_response_lines:
                _ = self.wfile.write(line.encode("utf-8"))
                _ = self.wfile.write(b"\n")
            self.wfile.flush()
            return

        response_body = json.dumps(self.non_stream_response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        _ = self.wfile.write(response_body)

    @override
    def log_message(self, format: str, *args: object) -> None:
        """关闭测试期 HTTP 访问日志，避免污染 pytest 输出。"""

        del format, args


class _FakeResponsesServer:
    """管理线程化本地假服务，确保 PluginRunner 可真实访问 `/v1/responses`。"""

    url: str
    recorded_requests: list[_RecordedRequest]
    _server: ThreadingHTTPServer
    _thread: threading.Thread

    def __init__(self) -> None:
        _FakeResponsesHandler.recorded_requests = []
        _FakeResponsesHandler.non_stream_response = {
            "id": "resp_non_stream",
            "model": "gpt-4.1-mini",
            "status": "completed",
            "output": [
                {
                    "id": "msg_non_stream",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello from smoke"}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }
        _FakeResponsesHandler.stream_response_lines = [
            'data: {"type":"response.created","response":{"id":"resp_stream","model":"gpt-4.1-mini","status":"in_progress"}}',
            'data: {"type":"response.output_text.delta","item_id":"msg_stream","delta":"tool "}',
            'data: {"type":"response.output_text.delta","item_id":"msg_stream","delta":"call"}',
            'data: {"type":"response.output_item.added","item":{"id":"fc_stream","type":"function_call","call_id":"call_weather","name":"get_weather","arguments":""}}',
            'data: {"type":"response.function_call_arguments.delta","item_id":"fc_stream","delta":"{\\"city\\":\\"Hang"}',
            'data: {"type":"response.function_call_arguments.delta","item_id":"fc_stream","delta":"zhou\\"}"}',
            'data: {"type":"response.function_call_arguments.done","item_id":"fc_stream","arguments":"{\\"city\\":\\"Hangzhou\\"}"}',
            'data: {"type":"response.output_item.done","item":{"id":"fc_stream","type":"function_call","call_id":"call_weather","name":"get_weather","arguments":"{\\"city\\":\\"Hangzhou\\"}"}}',
            'data: {"type":"response.completed","response":{"id":"resp_stream","model":"gpt-4.1-mini","status":"completed","usage":{"input_tokens":5,"output_tokens":6,"total_tokens":11}}}',
            "data: [DONE]",
        ]

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeResponsesHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self._server.server_port}/v1"
        self.recorded_requests = _FakeResponsesHandler.recorded_requests

    def __enter__(self) -> "_FakeResponsesServer":
        """启动后台 HTTP 服务线程，并返回可供测试使用的服务对象。"""

        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """在测试结束后关闭服务与线程，避免端口和后台线程泄漏。"""

        del exc_type, exc_value, traceback
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _build_llm_request(*, endpoint_url: str, stream: bool) -> ModelInvokeLLMRequest:
    """统一构造 smoke 用的 LLM invoke 请求，避免两条链路重复拼装 payload。"""

    return ModelInvokeLLMRequest(
        user_id="smoke-user",
        provider="sub2api-plugin",
        model_type=ModelType.LLM,
        model="gpt-4.1-mini",
        credentials={
            "endpoint_url": endpoint_url,
            "api_key": "smoke-key",
            "context_size": "32768",
        },
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={"temperature": 0.1},
        stop=[],
        tools=(
            [
                PromptMessageTool(
                    name="get_weather",
                    description="查询城市天气",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            ]
            if stream
            else []
        ),
        stream=stream,
    )


def test_test_script_uses_uv_run_pytest(repo_root: Path) -> None:
    """确认统一测试入口脚本存在且遵守 uv run 约束。"""

    script_path = repo_root / "scripts" / "test.sh"
    script_content = script_path.read_text(encoding="utf-8")

    assert "uv run --project . python -m pytest" in script_content


def test_responses_http_client_uses_fixed_user_agent() -> None:
    client = _ResponsesHTTPClient(endpoint_url="https://example.com/v1", api_key="test-key")

    headers = client._build_headers(stream=False)

    assert headers["User-Agent"] == "Codex Desktop/0.108.0-alpha.12 (Mac OS 26.3.0; arm64) unknown (Codex Desktop; 26.305.950)"


def test_plugin_runner_can_start_from_source_tree(repo_root: Path) -> None:
    """验证 PluginRunner 至少可以从源码目录启动插件进程，作为 smoke 基线。"""

    if shutil.which("dify") is None:
        pytest.skip("当前环境缺少 dify CLI，无法执行 PluginRunner smoke。")

    with PluginRunner(
        config=IntegrationConfig(),
        plugin_package_path=str(repo_root),
    ) as runner:
        assert runner.process.poll() is None


def test_plugin_runner_invokes_non_stream_and_stream_tool_call(repo_root: Path) -> None:
    """验证 PluginRunner 可真实完成一次非流式调用与一次流式 tool-call 调用。"""

    if shutil.which("dify") is None:
        pytest.skip("当前环境缺少 dify CLI，无法执行 PluginRunner smoke。")

    with _FakeResponsesServer() as fake_server:
        with PluginRunner(
            config=IntegrationConfig(),
            plugin_package_path=str(repo_root),
        ) as runner:
            non_stream_chunks = list(
                runner.invoke(
                    access_type=PluginInvokeType.Model,
                    access_action=ModelActions.InvokeLLM,
                    payload=_build_llm_request(endpoint_url=fake_server.url, stream=False),
                    response_type=LLMResultChunk,
                )
            )
            stream_chunks = list(
                runner.invoke(
                    access_type=PluginInvokeType.Model,
                    access_action=ModelActions.InvokeLLM,
                    payload=_build_llm_request(endpoint_url=fake_server.url, stream=True),
                    response_type=LLMResultChunk,
                )
            )

    assert len(non_stream_chunks) == 1
    assert non_stream_chunks[0].delta.message.content == "hello from smoke"
    assert [chunk.delta.message.content for chunk in stream_chunks[:2]] == ["tool ", "call"]
    assert stream_chunks[2].delta.message.tool_calls[0].function.name == "get_weather"
    assert stream_chunks[2].delta.message.tool_calls[0].function.arguments == '{"city":"Hangzhou"}'
    assert stream_chunks[-1].delta.finish_reason == "stop"

    assert [request.path for request in fake_server.recorded_requests] == ["/v1/responses", "/v1/responses"]
    assert fake_server.recorded_requests[0].payload["stream"] is False
    assert fake_server.recorded_requests[1].payload["stream"] is True
    assert fake_server.recorded_requests[1].payload["tool_choice"] == "auto"
