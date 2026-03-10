"""测试共享夹具与路径常量。"""

from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Callable, Protocol, TypeVar, cast

_F = TypeVar("_F", bound=Callable[..., object])


class _PytestMarkProtocol(Protocol):
    def xfail(
        self,
        *,
        raises: type[BaseException],
        strict: bool,
        reason: str,
    ) -> Callable[[_F], _F]: ...


class _PytestProtocol(Protocol):
    mark: _PytestMarkProtocol

    def fixture(self, *, scope: str) -> Callable[[_F], _F]: ...


class _LLMInvokeRequestProtocol(Protocol):
    provider: str
    model: str
    credentials: dict[str, str]


pytest = cast(_PytestProtocol, cast(object, importlib.import_module("pytest")))


# 使用仓库根目录作为测试路径锚点，避免各测试文件重复计算路径。
REPO_ROOT = Path(__file__).resolve().parents[1]
# 将 Responses 协议样本统一收敛到 fixtures 目录，供后续任务直接复用。
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _load_json_fixture(file_name: str) -> dict[str, object]:
    """读取 JSON 夹具，供协议契约测试复用。"""

    fixture_path = FIXTURES_DIR / file_name
    # 故意不吞掉文件不存在异常，让 RED 阶段直接暴露缺失的样本文件。
    return cast(dict[str, object], json.loads(fixture_path.read_text(encoding="utf-8")))


def _load_text_fixture(file_name: str) -> str:
    """读取文本夹具，主要用于 SSE/流式事件样本。"""

    fixture_path = FIXTURES_DIR / file_name
    # 文本流样本需要保持原始换行，因此直接返回原文本内容。
    return fixture_path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """暴露仓库根目录，供 smoke 测试定位插件源码。"""

    return REPO_ROOT


@pytest.fixture(scope="session")
def responses_request_fixture() -> dict[str, object]:
    """返回最小 Responses request 样本。"""

    return _load_json_fixture("responses_request_minimal.json")


@pytest.fixture(scope="session")
def responses_response_fixture() -> dict[str, object]:
    """返回最小 Responses response 样本。"""

    return _load_json_fixture("responses_response_minimal.json")


@pytest.fixture(scope="session")
def responses_stream_fixture() -> str:
    """返回最小 SSE 流式事件样本。"""

    return _load_text_fixture("responses_stream_minimal.sse")


@pytest.fixture(scope="session")
def llm_invoke_request() -> _LLMInvokeRequestProtocol:
    """构造后续 LLM 集成测试要复用的最小调用请求。"""

    request_module = importlib.import_module("dify_plugin.core.entities.plugin.request")
    model_module = importlib.import_module("dify_plugin.entities.model")
    message_module = importlib.import_module("dify_plugin.entities.model.message")
    request_factory = cast(Callable[..., object], getattr(request_module, "ModelInvokeLLMRequest"))
    model_type = cast(object, getattr(model_module, "ModelType"))
    user_prompt_message = cast(
        Callable[..., object],
        getattr(message_module, "UserPromptMessage"),
    )

    request = request_factory(
        user_id="test-user",
        provider="sub2api-plugin",
        model_type=getattr(model_type, "LLM"),
        model="gpt-4.1-mini",
        credentials={
            "endpoint_url": "https://example.com/v1",
            "api_key": "test-key",
        },
        prompt_messages=[user_prompt_message(content="hello")],
        model_parameters={},
        stop=[],
        tools=[],
        stream=True,
    )

    return cast(_LLMInvokeRequestProtocol, request)
