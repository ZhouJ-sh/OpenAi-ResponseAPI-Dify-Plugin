## sub2api-plugin

**Author:** zhou
**Version:** 0.0.1
**Type:** model

### Description

这是一个面向 OpenAI 兼容 `/v1/responses` 接口的 Dify LLM 插件。

- 首版只保留 `customizable-model` 合同，provider 入口为最小 no-op bridge。
- 当前只覆盖 `/v1/responses` 语义，不宣传也不兼容其它旧接口。
- 模型配置由 Dify 的自定义模型表单提供：`api_key`、`endpoint_url`、`context_size`。

### Scripts

- `bash scripts/test.sh`：通过 `uv run --project . pytest` 运行测试。
- `bash scripts/debug.sh`：读取 `.env`，创建 `logs/`，再以 `uv run --project . python -m main` 启动调试。
- `bash scripts/package.sh`：自动选择当前环境可用的 `dify-plugin` 或 `dify` CLI 打包，并输出 `.difypkg` 位置。
- `API_BASE_URL=... API_KEY=... bash scripts/e2e.sh [MODEL_NAME]`：对真实 `/v1/responses` 服务执行非流式与流式 tool-call 端到端验证。

### Quick Start

1. 复制 `.env.example` 为 `.env`，填入调试安装所需变量。
2. 运行 `bash scripts/test.sh`。
3. 运行 `bash scripts/debug.sh` 进入调试模式。
4. 运行 `bash scripts/package.sh` 生成插件包。
5. 如需真实环境验证，设置 `API_BASE_URL`、`API_KEY` 后运行 `bash scripts/e2e.sh`。

更多调试与打包说明见 `GUIDE.md`。
