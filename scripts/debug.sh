#!/bin/sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/debug.log"
ENV_FILE="$ROOT_DIR/.env"

print_help() {
  cat <<'EOF'
用法: bash scripts/debug.sh

说明:
  - 读取项目根目录 .env
  - 检查调试所需环境变量
  - 创建 logs/ 并把运行输出追加到 logs/debug.log
  - 使用 uv run --project . python -m main 启动插件
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

if [ "$#" -gt 0 ]; then
  printf '不支持的参数: %s\n\n' "$1" >&2
  print_help >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

if [ ! -f "$ENV_FILE" ]; then
  printf '未找到 %s，请先从 .env.example 复制并补齐配置。\n' "$ENV_FILE" >&2
  exit 1
fi

set -a
. "$ENV_FILE"
set +a

if [ "${INSTALL_METHOD:-}" = "remote" ]; then
  require_env "REMOTE_INSTALL_URL"
  require_env "REMOTE_INSTALL_KEY"
fi

printf '调试日志: %s\n' "$LOG_FILE"
printf '正在启动 sub2api-plugin 调试进程...\n'

cd "$ROOT_DIR"
exec uv run --project . python -m main >>"$LOG_FILE" 2>&1
