## sub2api-plugin

**Author:** zhou
**Version:** 0.0.1
**Type:** model

### Description

这是一个面向 OpenAI 兼容 `/v1/responses` 接口的 Dify LLM 插件。支持sub2api项目转发的codex接口。

- 首版只保留 `customizable-model` 合同，provider 入口为最小 no-op bridge。
- 当前只覆盖 `/v1/responses` 语义，不宣传也不兼容其它旧接口。
- 模型配置由 Dify 的自定义模型表单提供：`api_key`、`endpoint_url`、`context_size`。
- 首轮工具调用保持原生 Responses 形态；当出现 `ToolPromptMessage` 的续轮时，会自动切到 sub2api HTTP 兼容模式，避免直接发送 `function_call_output` 导致上游失败。

### Current Capabilities

- 仅支持 `llm`。
- 支持非流式文本回复。
- 支持流式文本回复。
- 支持工具调用与工具结果续链。
- 支持真实 `/v1/responses` 端到端验证。

### Compatibility Notes

- 当前插件只面向 HTTP `/v1/responses` 路径。
- 不支持 `/v1/chat/completions`。
- 不支持 `/v1/messages`。
- 当前实现不会向上游透传 `user` 字段，以兼容真实 sub2api 环境里该字段触发 `upstream_error` 的情况。
- 真实 sub2api 流式响应里可能出现 `response.in_progress`、`response.content_part.added`、`response.content_part.done`、`type=error` 等事件，插件已做兼容处理。

### Scripts

- `bash scripts/test.sh`：通过 `uv run --project . pytest` 运行测试。
- `bash scripts/debug.sh`：读取 `.env`，创建 `logs/`，再以 `uv run --project . python -m main` 启动调试。
- `bash scripts/package.sh`：自动选择当前环境可用的 `dify-plugin` 或 `dify` CLI 打包，并输出 `.difypkg` 位置。
- `API_BASE_URL=... API_KEY=... bash scripts/e2e.sh [MODEL_NAME]`：对真实 `/v1/responses` 服务执行非流式与流式 tool-call 端到端验证。

### Debug Logging

- 调试日志已接入 Dify 插件协议内的 `log` 事件，不会再破坏 PluginRunner 通信。
- 日志会输出脱敏摘要：endpoint、payload hash、payload key、输入结构、工具名、HTTP 错误摘要、流式终态摘要。
- 日志不会输出 API Key、原始 prompt 文本、原始工具参数、原始工具结果或完整错误体。

### Quick Start

1. 复制 `.env.example` 为 `.env`，填入调试安装所需变量。
2. 运行 `bash scripts/test.sh`。
3. 运行 `bash scripts/debug.sh` 进入调试模式。
4. 运行 `bash scripts/package.sh` 生成插件包。
5. 如需真实环境验证，设置 `API_BASE_URL`、`API_KEY` 后运行 `bash scripts/e2e.sh`。

### Recommended E2E Command

```bash
API_BASE_URL=http://10.10.0.172:8080/v1 \
API_KEY=sk-your-key \
MODEL_NAME=gpt-5.4 \
bash scripts/e2e.sh
```

成功时会输出两段摘要：

- `non_stream.text`：非流式回复内容
- `stream.tool_calls`：流式工具调用结果

更多调试与打包说明见 `GUIDE.md`。
