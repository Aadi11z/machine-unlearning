# Lightweight VLM Machine Unlearning Pipeline

End-to-end project for machine unlearning on a lightweight vision-language model (VLM):
1. Pull CIFAR-10 or CIFAR-100 and create retain/forget splits.
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

## Config-Driven Run

Select the dataset by changing one field in `config/parameters.yaml`:

```yaml
data:
  dataset: cifar10  # change to cifar100
```

The corresponding profile supplies its forget classes and fraction. CIFAR-10
uses cat/dog (`3,5`). CIFAR-100 uses five flower classes
(`54,62,70,82,92`).

Inspect the resolved run without starting any work:

```bash
python scripts/run_pipeline.py --show-config
```

Run the complete configured pipeline:

```bash
python scripts/run_pipeline.py
```

Artifacts remain isolated:

```text
outputs/
  cifar10/
    splits/
    finetune/
    unlearning/
    comparison/
  cifar100/
    splits/
    finetune/
    unlearning/
    comparison/
```

On Sharanga, `env_activation.sh` supplies `UNML_DATA` and `UNML_OUTPUTS`.
The same command then reads data and writes all heavy artifacts under scratch.

CLI arguments remain available as temporary overrides, but normal experiments
should be defined in `config/parameters.yaml`.

## Implemented attacks
- `Confidence MIA`: membership inference using true-label confidence.
- `Delta-to-Base MIA`: confidence shift from the base adapter model (`current_confidence - base_confidence`).

Forgetting quality combines:
- Forget-set accuracy drop.
- Resistance to both attacks (AUC close to 0.5 is better).

## Notes
- Backbone: `openai/clip-vit-base-patch32` (frozen).
- Trainable params: low-rank adapters on image/text embeddings + optional logit scale.
- Split files record the dataset and class vocabulary used by downstream stages.
- Split/checkpoint dataset mismatches are rejected before an experiment runs.
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
