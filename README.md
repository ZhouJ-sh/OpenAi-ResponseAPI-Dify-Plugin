# sub2api-dify-plugin

一个面向 `sub2api` / OpenAI-compatible `/v1/responses` 接口的 Dify LLM 插件。

## 当前能力

- 仅支持 `llm`
- 仅支持 `customizable-model`
- 支持非流式文本回复
- 支持流式文本回复
- 支持工具调用
- 支持工具结果续链兼容转换

## 兼容范围

- 当前只面向 HTTP `/v1/responses`
- 不支持 `/v1/chat/completions`
- 不支持 `/v1/messages`
- 当前不会向上游透传 `user` 字段，以兼容部分真实 `sub2api` 环境中的上游限制

## 为什么有“续链兼容转换”

在部分真实 `sub2api` HTTP 场景下，首轮工具调用可以正常返回，但二轮 continuation 若直接发送原生 `function_call_output`，上游可能返回 `upstream_error`。

因此当前插件采用以下策略：

- 首轮工具调用仍保持原生 Responses 形态
- 当续轮中出现 `ToolPromptMessage` 时，将工具调用轨迹与工具结果转换成兼容的 transcript 文本块
- 同时继续保留 `tools` 与 `tool_choice=auto`

这样可以在不依赖 `previous_response_id` 的 HTTP `/v1/responses` 路径下，尽可能兼容真实 `sub2api` 服务。

## 配置字段

模型配置通过 Dify 的自定义模型表单提供：

- `api_key`：目标接口所需的 API Key，可留空以兼容无鉴权代理
- `endpoint_url`：接口基地址，例如 `https://your-host/v1`
- `context_size`：模型上下文长度，例如 `4096` 或 `32768`

## 脚本入口

- `bash scripts/test.sh`
  - 运行测试，固定走 `uv run --project . pytest`
- `bash scripts/debug.sh`
  - 读取 `.env`，创建 `logs/`，再通过 `uv run --project . python -m main` 启动调试
- `bash scripts/package.sh`
  - 自动调用当前环境可用的 `dify-plugin` 或 `dify` CLI 打包 `.difypkg`
- `API_BASE_URL=... API_KEY=... bash scripts/e2e.sh [MODEL_NAME]`
  - 对真实 `/v1/responses` 服务执行端到端验证

## 调试日志

插件内已经加入最小化调试日志，主要用于定位真实环境中的 `upstream_error`。

日志特点：

- 输出 endpoint、payload 摘要、工具名、流式终态、HTTP 错误摘要
- 不输出 API Key
- 不输出原始 prompt 文本
- 不输出原始工具参数与工具结果
- 不输出完整错误体正文

日志通过 Dify 插件协议内的 `log` 事件输出，不会破坏 PluginRunner 通信。

## 快速开始

### 1. 复制环境文件

```bash
cp .env.example .env
```

按需填写调试安装变量。

### 2. 运行测试

```bash
bash scripts/test.sh
```

### 3. 启动本地调试

```bash
bash scripts/debug.sh
```

### 4. 打包插件

```bash
bash scripts/package.sh
```

### 5. 执行真实端到端验证

```bash
API_BASE_URL=http://your-host:8080/v1 \
API_KEY=sk-your-key \
MODEL_NAME=gpt-5.4 \
bash scripts/e2e.sh
```

如果不传模型名，脚本会先请求 `/models` 并自动选择第一个可用模型。

成功时会输出两段摘要：

- `non_stream.text`：非流式文本回复
- `stream.tool_calls`：流式工具调用结果

## 更多说明

更详细的开发、调试与打包说明见 `GUIDE.md`。

## License

本项目基于 MIT License 开源，见 `LICENSE`。
