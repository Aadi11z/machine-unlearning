#!/usr/bin/env bash
# Run with: source env_activation.sh
# The project environment is created and locked by: uv sync --locked

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[ERROR] Source this file: source env_activation.sh" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is required but was not found on PATH" >&2
    return 1
fi

SCRATCH_PROJECT="${SCRATCH_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ENV="${UV_PROJECT_ENVIRONMENT:-$SCRATCH_PROJECT/.venv}"

if [[ ! -d "$PROJECT_ENV" ]]; then
    echo "[ERROR] uv environment not found at $PROJECT_ENV" >&2
    echo "Run: cd $SCRATCH_PROJECT && uv sync --locked" >&2
    return 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    conda deactivate
fi

# Activate the uv-managed environment for tools that require an active venv.
source "$PROJECT_ENV/bin/activate"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ENV"

# Keep caches and writable experiment artifacts on the project filesystem.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_PROJECT/.uv-cache}"
export HF_HOME="${HF_HOME:-$SCRATCH_PROJECT/huggingface_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

export UNML_DATA="${UNML_DATA:-$SCRATCH_PROJECT/data}"
export UNML_OUTPUTS="${UNML_OUTPUTS:-$SCRATCH_PROJECT/outputs}"
export UNML_SPLIT="${UNML_SPLIT:-$UNML_OUTPUTS/cifar100/canonical/development/splits/cifar100_canonical_development_v1.json}"
export UNML_BEST_CKPT="${UNML_BEST_CKPT:-$UNML_OUTPUTS/cifar100/canonical/cifar100_canonical_v1/checkpoints/finetuned_best.pt}"

echo "uv environment activated: $PROJECT_ENV"
echo "UV_CACHE_DIR: $UV_CACHE_DIR"
echo "HF_HOME: $HF_HOME"
echo ""
echo "Convenience vars:"
echo "  UNML_DATA = $UNML_DATA"
echo "  UNML_OUTPUTS = $UNML_OUTPUTS"
echo "  UNML_SPLIT = $UNML_SPLIT"
echo "  UNML_BEST_CKPT = $UNML_BEST_CKPT"
