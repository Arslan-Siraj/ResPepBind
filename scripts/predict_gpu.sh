#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
exec respepbind --device cuda "$@"
