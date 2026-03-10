#!/bin/sh

set -eu

uv run --project . pytest "$@"
