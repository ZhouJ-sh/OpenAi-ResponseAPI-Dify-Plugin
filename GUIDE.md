# OpenAi-ResponseAPI-Dify-Plugin 开发说明

## 插件定位

`OpenAi-ResponseAPI-Dify-Plugin` 是一个 Dify LLM 插件，当前版本只桥接 OpenAI 兼容的 `/v1/responses` 接口。

- 仅支持 `customizable-model` 配置方式。
- provider 层不再提供独立凭据表单。
- 当前只面向 `/v1/responses`，不宣传其它旧接口。

## 配置说明

模型配置在 Dify 的自定义模型表单中完成，最小字段如下：

- `api_key`：目标接口所需的 API Key，可留空以兼容无鉴权代理。
- `endpoint_url`：接口基地址，例如 `https://your-host/v1`。
- `context_size`：模型上下文长度，例如 `4096` 或 `32768`。

## gpt-5.4 参数兼容说明

- 当 `reasoning_effort` 为 `none` 时，可以同时使用 `temperature` 与 `top_p`。
- 当 `reasoning_effort` 不是 `none` 时，不应再设置 `temperature` 与 `top_p`。
- 插件参数 UI 会在 `reasoning_effort` 不是 `none` 时自动隐藏 `temperature` 与 `top_p`，并清理已存在的这两个参数值。
- `frequency_penalty` 与 `presence_penalty` 当前仍按顶层字段透传。

调试环境变量放在项目根目录 `.env` 中，可从 `.env.example` 复制：

```env
INSTALL_METHOD=remote
REMOTE_INSTALL_URL=debug.dify.ai:5003
REMOTE_INSTALL_KEY=your-debug-key
```

## 统一脚本入口

### 测试

```bash
bash scripts/test.sh
```

- 固定走 `uv run --project . pytest`。
- 可继续追加 pytest 参数，例如 `bash scripts/test.sh tests/test_provider_schema.py -q`。

### 调试

```bash
bash scripts/debug.sh
```

- 先检查 `.env` 与必要环境变量。
- 自动创建 `logs/`。
- 最终通过 `uv run --project . python -m main` 启动插件。
- 若环境缺失，会直接给出可操作提示，不抛 Python traceback。

### 打包

```bash
bash scripts/package.sh
```

- 自动探测 `dify-plugin` 或 `dify` CLI。
- 调用当前环境可用的 `plugin package` 命令。
- 打包完成后输出可定位的 `.difypkg` 路径。

### 真实端到端测试

```bash
API_BASE_URL=http://your-host:8080/v1 \
API_KEY=sk-your-key \
bash scripts/e2e.sh gpt-5.4
```

- 若不传模型名，脚本会先请求 `/models` 自动选择第一个可用模型。
- 固定验证两条真实链路：非流式文本回复、流式 tool-call 回复。
- 当前脚本不会向上游透传 `user` 字段，以兼容真实 sub2api 环境里该字段触发 `upstream_error` 的问题。

## Manifest 约定

- `manifest.yaml` 明确声明这是 `/v1/responses` LLM 插件。
- 没有可信公开仓库地址时，不填 `repo` 字段，避免用伪造 URL 破坏打包校验。
- provider 入口保持最小桥接，不延续模板里的 provider credential 校验残留。
