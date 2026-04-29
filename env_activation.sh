#!/usr/bin/env bash
# Run with: source env_activation.sh
# Need to run before any session

PATH=""

# Activate Venv
if [[ ! -d "$PATH/unml-env" ]]; then
    echo "[ERROR] venv not found at $PATH/unml-env"
    echo "Create it with: python3 -m venv $PATH/unml-env"
    return 1
fi
source "$PATH/unml-env/bin/activate"

# Set Cache dirs
export PIP_CACHE_DIR="$PATH/.pip-cache"
export HF_HOME="$PATH/huggingface_cache"

# Uncomment if you need to create
# mkdir -p "$PATH/.pip-cache"
# mkdir -p "$PATH/huggingface_cache"
# mkdir -p "$PATH/data"
# mkdir -p "$PATH/outputs"

# Set Dataset path, Outputs path
export DATA="$PATH/data"
export OUTPUTS="$PATH/outputs"

echo "✓ venv Activated: $PATH/unml-env"
echo "✓ PIP_CACHE_DIR:  $PIP_CACHE_DIR"
echo "✓ HF_HOME:        $HF_HOME"
echo ""
echo "Convenience vars:"
echo "  DATA     = $DATA"
echo "  OUTPUTS  = $OUTPUTS"
