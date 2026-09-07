#!/bin/bash
#SBATCH --job-name=unml-cifar100-oracle
#SBATCH --partition=gpu_a100_8
#SBATCH --qos=qos_gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/machine-unlearning/logs/%x-%j.out
#SBATCH --error=/scratch/%u/machine-unlearning/logs/%x-%j.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[ERROR] Submit this file with sbatch; direct bash execution has no GPU allocation." >&2
  exit 2
fi

export SCRATCH="/scratch/${USER:?USER must identify the cluster account}"
UNML_ROOT="${UNML_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set UNML_ROOT}}"

: "${UNML_RUN_ID:?Set a new oracle run id, e.g. 2026-08-30-a100-oracle-01}"
: "${UNML_ORACLE_REQUEST:?Set UNML_ORACLE_REQUEST, e.g. flowers_superclass}"

# Both values become directory components. Reject traversal before any work.
for component in "$UNML_RUN_ID" "$UNML_ORACLE_REQUEST"; do
  if [[ ! "$component" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
    echo "[ERROR] Run id and request must be 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit." >&2
    exit 2
  fi
done

cd "$UNML_ROOT"
UNML_ROOT="$(pwd -P)"
# Slurm opens log files before running this script; create logs before sbatch.
spack load python@3.11.9
source "$UNML_ROOT/env_activation.sh"
UNML_DATA_DIR="$UNML_DATA"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EXPECTED_SCRATCH_PROJECT="/scratch/$USER/machine-unlearning"
if [[ "$SCRATCH_PROJECT" != "$EXPECTED_SCRATCH_PROJECT" ]]; then
  echo "[ERROR] Expected scratch project $EXPECTED_SCRATCH_PROJECT, got $SCRATCH_PROJECT" >&2
  exit 2
fi

FINAL="$UNML_OUTPUTS/cifar100/baseline"
ORACLE_DIR="$UNML_OUTPUTS/cifar100/oracle/$UNML_RUN_ID/$UNML_ORACLE_REQUEST"
ORACLE_SPLIT="$ORACLE_DIR/splits/${UNML_ORACLE_REQUEST}.json"
if [[ -e "$ORACLE_DIR" ]]; then
  echo "[ERROR] Oracle run destination already exists: $ORACLE_DIR" >&2
  exit 2
fi

echo "[preflight] host=$(hostname) job=$SLURM_JOB_ID scratch_project=$SCRATCH_PROJECT"
echo "[preflight] root=$UNML_ROOT data=$UNML_DATA_DIR cache=$HF_HOME"
echo "[preflight] venv=$UV_PROJECT_ENVIRONMENT outputs=$UNML_OUTPUTS"
echo "[preflight] canonical=$FINAL oracle=$ORACLE_DIR"

uv run --locked python -c \
  'import json, sys; from pathlib import Path; from unml.manifest import verify_manifest_artifacts; root = Path(sys.argv[1]); manifest_path = root / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); verify_manifest_artifacts(manifest, root=root); required = [root / "checkpoints" / "base_init.pt", root / "metrics" / "finetune_metrics.json"]; missing = [str(path) for path in required if not path.is_file()]; missing and (_ for _ in ()).throw(FileNotFoundError(f"Canonical oracle sources missing: {missing}")); print(f"[preflight] canonical manifest verified: {manifest_path}")' \
  "$FINAL"

uv run --locked python -c \
  'import os; from unml.preflight import validate_cache_manifest, validate_cuda; print({"cuda": validate_cuda(True, "bf16"), "cache_manifest": str(validate_cache_manifest(model_name="openai/clip-vit-base-patch16", dataset_name="cifar100", hf_home=os.environ["HF_HOME"])[0])})'

uv run --locked python scripts/generate_canonical_oracle_split.py \
  --data-dir "$UNML_DATA_DIR" --output-path "$ORACLE_SPLIT" \
  --request "$UNML_ORACLE_REQUEST" --download

uv run --locked python scripts/train_vlm.py --oracle --offline --device cuda \
  --data-dir "$UNML_DATA_DIR" --dataset cifar100 \
  --request "$UNML_ORACLE_REQUEST" --split-path "$ORACLE_SPLIT" \
  --output-dir "$ORACLE_DIR" \
  --initial-checkpoint "$FINAL/checkpoints/base_init.pt" \
  --source-metrics "$FINAL/metrics/finetune_metrics.json"

uv run --locked python scripts/evaluate_canonical_reference_pair.py \
  --canonical-dir "$FINAL" --oracle-dir "$ORACLE_DIR" \
  --data-dir "$UNML_DATA_DIR" --device cuda --offline

uv run --locked python scripts/write_retraining_oracle_manifest.py \
  --canonical-dir "$FINAL" --oracle-dir "$ORACLE_DIR"

echo "canonical=$FINAL"
echo "oracle=$ORACLE_DIR"
