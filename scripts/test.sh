#!/bin/sh

set -eu

uv run --project . python -m pytest "$@"
