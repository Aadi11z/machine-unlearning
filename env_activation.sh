#!/usr/bin/env bash
# Run with: source env_activation.sh
# The project environment is created and locked by: uv sync --locked

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[ERROR] Source this file: source env_activation.sh" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SCRATCH:-}" ]]; then
    SCRATCH_PROJECT="${SCRATCH%/}/machine-unlearning"
else
    SCRATCH_PROJECT="$REPO_ROOT"
fi

export SCRATCH_PROJECT
export UV_PROJECT_ENVIRONMENT="$SCRATCH_PROJECT/.venv"
export UV_CACHE_DIR="$SCRATCH_PROJECT/.uv-cache"
export HF_HOME="$SCRATCH_PROJECT/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export UNML_DATA="$SCRATCH_PROJECT/data"
export UNML_OUTPUTS="$SCRATCH_PROJECT/outputs"
export PATH="$SCRATCH_PROJECT/.local/bin:$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is required but was not found on PATH" >&2
    echo "Expected: $SCRATCH_PROJECT/.local/bin/uv" >&2
    return 1
fi

if [[ ! -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]]; then
    echo "[ERROR] uv environment not found at $UV_PROJECT_ENVIRONMENT" >&2
    echo "Run: cd $REPO_ROOT && uv sync --locked" >&2
    return 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    conda deactivate
fi

source "$UV_PROJECT_ENVIRONMENT/bin/activate"

echo "uv environment activated"
