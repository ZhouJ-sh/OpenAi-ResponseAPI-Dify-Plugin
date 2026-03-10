#!/bin/sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
PARENT_DIR=$(dirname "$ROOT_DIR")
PLUGIN_DIR_NAME=$(basename "$ROOT_DIR")

find_artifact() {
  for candidate in "$ROOT_DIR"/*.difypkg "$ROOT_DIR"/dist/*.difypkg "$PARENT_DIR"/*.difypkg; do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

run_packager() {
  cd "$PARENT_DIR"

  if command -v dify-plugin >/dev/null 2>&1; then
    printf '使用 dify-plugin plugin package 打包...\n'
    dify-plugin plugin package "$PLUGIN_DIR_NAME"
    return 0
  fi

  if command -v dify >/dev/null 2>&1; then
    printf '使用 dify plugin package 打包...\n'
    dify plugin package "$PLUGIN_DIR_NAME"
    return 0
  fi

  printf '未找到 dify-plugin 或 dify CLI，无法打包。\n' >&2
  return 1
}

run_packager

ARTIFACT_PATH=$(find_artifact || true)
if [ -z "$ARTIFACT_PATH" ]; then
  printf '打包命令已执行，但未定位到 .difypkg 产物。\n' >&2
  exit 1
fi

printf '已生成插件包: %s\n' "$ARTIFACT_PATH"
