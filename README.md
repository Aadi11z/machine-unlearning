# Lightweight VLM Machine Unlearning Pipeline

End-to-end project for machine unlearning on a lightweight vision-language model (VLM):
1. Pull CIFAR-10 and create retain/forget splits.
2. Fine-tune a lightweight CLIP adapter model.
3. Unlearn using multiple methods.
4. Run membership-inference attacks and compare model utility and forget quality.

## Why this is novel
This project includes a new unlearning objective:
- `counterfactual_rebind`: for forget samples, the model is pushed toward counterfactual class prompts while preserving retain behavior through KL-to-teacher regularization.

This creates a controllable forgetting mechanism that is stronger than retain-only fine-tuning while preserving utility better than unconstrained gradient ascent.

## Project structure
- `src/unml/data.py`: dataset pull + split creation + dataloaders
- `src/unml/model.py`: frozen CLIP backbone + lightweight low-rank adapters
- `src/unml/train.py`: finetuning pipeline
- `src/unml/unlearn.py`: unlearning methods
- `src/unml/attacks.py`: membership-inference attacks + tradeoff plot/report
- `scripts/*.py`: CLI entrypoints
- `scripts/run_pipeline.py`: full experiment orchestration


## Tests
```bash
pytest -q
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick run (full pipeline)
```bash
python scripts/run_pipeline.py \
  --data-dir data \
  --split-path outputs/splits/cifar10_split.json \
  --forget-classes 3,5 \
  --ft-epochs 3 \
  --ul-steps 200 \
  --batch-size 128 \
  --device auto
```

Outputs:
- `outputs/finetune/checkpoints/base_init.pt`
- `outputs/finetune/checkpoints/finetuned_best.pt`
- `outputs/unlearning/checkpoints/unlearn_<method>.pt`
- `outputs/comparison/comparison.csv`
- `outputs/comparison/comparison.md`
- `outputs/comparison/utility_vs_forget.png`

## Step-by-step commands

### 1) Pull dataset and create splits
```bash
python scripts/prepare_data.py \
  --data-dir data \
  --split-path outputs/splits/cifar10_split.json \
  --forget-classes 3,5 \
  --forget-fraction 1.0
```

### 2) Fine-tune lightweight VLM adapters
```bash
python scripts/train_vlm.py \
  --data-dir data \
  --split-path outputs/splits/cifar10_split.json \
  --output-dir outputs/finetune \
  --epochs 5 \
  --batch-size 128
```

### 3) Run unlearning methods
```bash
python scripts/run_unlearning.py \
  --data-dir data \
  --split-path outputs/splits/cifar10_split.json \
  --finetuned-checkpoint outputs/finetune/checkpoints/finetuned_best.pt \
  --output-dir outputs/unlearning \
  --method retain_only

python scripts/run_unlearning.py \
  --data-dir data \
  --split-path outputs/splits/cifar10_split.json \
  --finetuned-checkpoint outputs/finetune/checkpoints/finetuned_best.pt \
  --output-dir outputs/unlearning \
  --method ga_kl

python scripts/run_unlearning.py \
  --data-dir data \
  --split-path outputs/splits/cifar10_split.json \
  --finetuned-checkpoint outputs/finetune/checkpoints/finetuned_best.pt \
  --output-dir outputs/unlearning \
  --method counterfactual_rebind
```

### 4) Attack and compare utility vs forgetting
```bash
python scripts/evaluate_attacks.py \
  --data-dir data \
  --split-path outputs/splits/cifar10_split.json \
  --base-checkpoint outputs/finetune/checkpoints/base_init.pt \
  --candidate finetuned=outputs/finetune/checkpoints/finetuned_best.pt \
  --candidate retain_only=outputs/unlearning/checkpoints/unlearn_retain_only.pt \
  --candidate ga_kl=outputs/unlearning/checkpoints/unlearn_ga_kl.pt \
  --candidate counterfactual_rebind=outputs/unlearning/checkpoints/unlearn_counterfactual_rebind.pt
```

## Implemented attacks
- `Confidence MIA`: membership inference using true-label confidence.
- `Delta-to-Base MIA`: confidence shift from the base adapter model (`current_confidence - base_confidence`).

Forgetting quality combines:
- Forget-set accuracy drop.
- Resistance to both attacks (AUC close to 0.5 is better).

## Notes
- Backbone: `openai/clip-vit-base-patch32` (frozen).
- Trainable params: low-rank adapters on image/text embeddings + optional logit scale.
- This makes training lightweight and unlearning iterations fast.

# Novel Directions To Explore (this is needs research)

- Hard counterfactual rebind (novel extension): Instead of random y_cf, choose semantically closest competing class by embedding similarity.
> Hypothesis: more realistic confusion yields stronger and cleaner forgetting.
- Curriculum counterfactual rebind: Start with easy counterfactual classes, then gradually harder ones.
> Hypothesis: improves stability and utility retention.
- Uncertainty-aware rebind: Weight forget samples by confidence or margin. Focus updates on high-memorization points.
- Prototype-anchored rebind: Add class prototype alignment so forget samples move toward chosen counterfactual prototype.
- Distribution-preserving rebind: Regularize to preserve retain feature geometry while altering forget regions.

- Disentanglement-Based Unlearning
1. Split representation into shared and forget-sensitive components.
2. Train adversary to predict forget attribute from shared part.
3. Train encoder to remove forget signal from shared part (gradient reversal/adversarial objective).
4. Use retain supervision + utility constraints so task performance remains.
5. At unlearning time, damp or reset forget-sensitive branch and rebind through shared branch.
