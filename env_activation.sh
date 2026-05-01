#!/usr/bin/env bash
# Run with: source env_activation.sh
# Need to run before any session

SCRATCH_PROJECT=""

# Activate Venv
if [[ ! -d "$SCRATCH_PROJECT/unml-env" ]]; then
    echo "[ERROR] venv not found at $SCRATCH_PROJECT/unml-env"
    echo "Create it with: python3 -m venv $SCRATCH_PROJECT/unml-env"
    return 1
fi
source "$SCRATCH_PROJECT/unml-env/bin/activate"

# Set Cache dirs
export PIP_CACHE_DIR="$SCRATCH_PROJECT/.pip-cache"
export HF_HOME="$SCRATCH_PROJECT/huggingface_cache"

# Uncomment if you need to create
# mkdir -p "$SCRATCH_PROJECT/.pip-cache"
# mkdir -p "$SCRATCH_PROJECT/huggingface_cache"
# mkdir -p "$SCRATCH_PROJECT/data"
# mkdir -p "$SCRATCH_PROJECT/outputs"

# Set Dataset path, Outputs path, and Splits Location
export UNML_DATA="$SCRATCH_PROJECT/data"
export UNML_OUTPUTS="$SCRATCH_PROJECT/outputs"
export UNML_SPLIT="$SCRATCH_PROJECT/outputs/splits"
export UNML_BEST_CKPT="$SCRATCH_PROJECT/outputs/finetune/checkpoints/finetuned_best.pt"

echo "✓ venv Activated: $SCRATCH_PROJECT/unml-env"
echo "✓ PIP_CACHE_DIR: $PIP_CACHE_DIR"
echo "✓ HF_HOME: $HF_HOME"
echo ""
echo "Convenience vars:"
echo "  UNML_DATA = $UNML_DATA"
echo "  UNML_OUTPUTS = $UNML_OUTPUTS"
echo "  UNML_SPLIT = $UNML_SPLIT"
echo "  UNML_BEST_CKPT = $UNML_BEST_CKPT"
