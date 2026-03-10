#!/bin/sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)

print_help() {
  cat <<'EOF'
用法: API_BASE_URL=http://host:port/v1 API_KEY=sk-... bash scripts/e2e.sh [MODEL_NAME]

说明:
  - 通过 PluginRunner 对当前插件执行真实端到端验证
  - 默认模型名优先取 MODEL_NAME 环境变量，其次自动探测 /models 的第一个模型
  - 固定验证两条链路：
      1. 非流式文本回复
      2. 流式 tool-call 回复
  - 当前脚本不会向上游透传 user 字段，以兼容真实 sub2api 环境
EOF
}

require_env() {
  variable_name="$1"
  eval "variable_value=\${$variable_name:-}"
  if [ -z "$variable_value" ]; then
    printf '缺少必要环境变量: %s\n' "$variable_name" >&2
    exit 1
  fi
}

if [ "${1:-}" = "--help" ]; then
  print_help
  exit 0
fi

require_env "API_BASE_URL"
require_env "API_KEY"

MODEL_NAME_INPUT="${1:-${MODEL_NAME:-}}"

cd "$ROOT_DIR"

API_BASE_URL="$API_BASE_URL" API_KEY="$API_KEY" MODEL_NAME_INPUT="$MODEL_NAME_INPUT" uv run --project . python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import request

from dify_plugin.config.integration_config import IntegrationConfig
from dify_plugin.core.entities.plugin.request import ModelActions, ModelInvokeLLMRequest, PluginInvokeType
from dify_plugin.entities.model import ModelType
from dify_plugin.entities.model.llm import LLMResultChunk
from dify_plugin.entities.model.message import PromptMessageTool, UserPromptMessage
from dify_plugin.integration.run import PluginRunner


def resolve_model_name(api_base_url: str, api_key: str, configured_model: str) -> str:
    if configured_model:
        return configured_model

    models_request = request.Request(
        api_base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with request.urlopen(models_request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    models = payload.get("data") or []
    if not models:
        raise ValueError("/models 未返回可用模型，请通过 MODEL_NAME 显式指定模型。")

    first_model = models[0].get("id")
    if not isinstance(first_model, str) or not first_model:
        raise ValueError("/models 返回的首个模型缺少有效 id，请通过 MODEL_NAME 显式指定模型。")
    return first_model


api_base_url = os.environ["API_BASE_URL"]
api_key = os.environ["API_KEY"]
model_name = resolve_model_name(api_base_url, api_key, os.environ.get("MODEL_NAME_INPUT", ""))
plugin_path = str(Path(".").resolve())
credentials = {
    "endpoint_url": api_base_url,
    "api_key": api_key,
    "context_size": os.environ.get("CONTEXT_SIZE", "32768"),
}

non_stream_payload = ModelInvokeLLMRequest(
    user_id="",
    provider="sub2api-plugin",
    model_type=ModelType.LLM,
    model=model_name,
    credentials=credentials,
    prompt_messages=[UserPromptMessage(content="请只回复 E2E_OK，不要输出其他内容。")],
    model_parameters={"temperature": 0},
    stop=[],
    tools=[],
    stream=False,
)

stream_tool_payload = ModelInvokeLLMRequest(
    user_id="",
    provider="sub2api-plugin",
    model_type=ModelType.LLM,
    model=model_name,
    credentials=credentials,
    prompt_messages=[UserPromptMessage(content="请调用 get_weather 工具查询杭州天气，不要直接回答。")],
    model_parameters={"temperature": 0},
    stop=[],
    tools=[
        PromptMessageTool(
            name="get_weather",
            description="查询城市天气",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        )
    ],
    stream=True,
)

summary: dict[str, object] = {"model": model_name}
with PluginRunner(config=IntegrationConfig(), plugin_package_path=plugin_path) as runner:
    non_stream_chunks = list(
        runner.invoke(
            access_type=PluginInvokeType.Model,
            access_action=ModelActions.InvokeLLM,
            payload=non_stream_payload,
            response_type=LLMResultChunk,
        )
    )
    stream_chunks = list(
        runner.invoke(
            access_type=PluginInvokeType.Model,
            access_action=ModelActions.InvokeLLM,
            payload=stream_tool_payload,
            response_type=LLMResultChunk,
        )
    )

summary["non_stream"] = {
    "chunk_count": len(non_stream_chunks),
    "text": "".join((chunk.delta.message.content or "") for chunk in non_stream_chunks),
}
summary["stream"] = {
    "chunk_count": len(stream_chunks),
    "text": "".join((chunk.delta.message.content or "") for chunk in stream_chunks),
    "tool_calls": [
        {
            "id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        }
        for chunk in stream_chunks
        for tool_call in (chunk.delta.message.tool_calls or [])
    ],
    "finish_reasons": [chunk.delta.finish_reason for chunk in stream_chunks if chunk.delta.finish_reason],
}

print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
