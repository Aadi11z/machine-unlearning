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
- `src/unml/model.py`: frozen CLIP backbone + post-projection adapters or
  internal vision LoRA
- `src/unml/train.py`: finetuning pipeline
- `src/unml/unlearn.py`: unlearning methods
- `src/unml/attacks.py`: membership-inference attacks + tradeoff plot/report
- `src/unml/probe.py`: sample- and class-level checkpoint inspection
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

The same switch selects the model architecture:

- CIFAR-10: CLIP ViT-B/32 with the historical image/text post-projection
  adapters.
- CIFAR-100: CLIP ViT-B/16 with rank-8 LoRA on `q_proj` and `v_proj` in all
  12 vision blocks (24 adapted projections).

The CIFAR-100 profile keeps the text encoder and original CLIP weights frozen,
uses BF16 and vision gradient checkpointing, and trains with batch size 64 plus
two-step accumulation for effective batch size 128.

For CIFAR-100, select the deletion request in the same profile:

```yaml
data:
  profiles:
    cifar100:
      request: flowers_superclass  # or rose_selective
```

- `flowers_superclass` forgets all five flower classes.
- `rose_selective` forgets rose and records the other four flowers as sibling
  retain classes.

Inspect the resolved run without starting any work:

```bash
python scripts/run_pipeline.py --show-config
```

Run the complete configured pipeline:

```bash
python scripts/run_pipeline.py
```

Before requesting a GPU on Sharanga, populate and verify the Hugging Face
cache on the login node:

```bash
python scripts/cache_model.py
```

Run the bounded infrastructure smoke test through a GPU batch job:

```bash
python scripts/train_vlm.py --smoke --offline --device cuda
```

Smoke settings are stored under `training.smoke` in
`config/parameters.yaml`. Smoke artifacts are isolated under
`phase2_benchmark`, marked as partial, and must not be used as research
results.

For CIFAR-100, the config also enables the optimized input pipeline:

```yaml
training:
  profiles:
    cifar100:
      persistent_workers: true
      non_blocking: true
  pin_memory: true
  prefetch_factor: 2
```

Pinned host buffers and nonblocking CUDA copies can overlap data transfer with
GPU work. Persistent workers are used only for reusable training loaders;
evaluation loaders are kept nonpersistent to avoid retaining many worker
processes. The same controls apply to unlearning and attack evaluation.
Throughput improvement must be measured on Sharanga before it is reported.

Run the controlled runtime matrix on one A100 after caching the model:

```bash
python scripts/benchmark_runtime.py \
  --dataset cifar100 \
  --request flowers_superclass \
  --repeats 3
```

The matrix compares the configured profile, disabled gradient checkpointing,
synchronous input transfer, and eight workers. Every run uses an isolated
smoke directory. Raw records and mean/standard-deviation summaries are written
under `runtime_benchmark/`. The summarizer rejects mixed Git commits, GPU
models, optimizer-step counts, or processed-example counts.

Artifacts remain isolated:

```text
outputs/
  cifar10/
    splits/
    finetune/
    unlearning/
    comparison/
  cifar100/
    flowers_superclass/
      splits/
      finetune/
      unlearning/
      comparison/
    rose_selective/
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

Evaluation is hierarchy-aware for both datasets. Each candidate is evaluated
once on the complete test set and once on forget-training samples. The same
outputs produce:

- target test micro/macro accuracy;
- sibling test micro/macro accuracy (`N/A` when no sibling group exists);
- unrelated-retain micro/macro accuracy;
- overall and retain accuracy;
- full per-class accuracy and confusion CSVs;
- a compact target/sibling/unrelated confusion CSV;
- confidence and delta-to-base MIA.

Delta-MIA scores are aligned by dataset index before subtraction, so shuffled
forget loaders cannot compare different samples. Base-model outputs are
computed once and reused across all candidate checkpoints.

Manually inspect selected examples across the fine-tuned reference and
unlearned checkpoints:

```bash
python scripts/probe_checkpoint.py --offline --device cuda
```

The default test probe selects examples from the configured target classes.
Use `--class-name rose`, `--class-id 70`, or repeatable `--index` arguments for
specific checks. With `--source train` and no selector, it samples the exact
`forget_indices` recorded in the split. Results are written as CSV, JSON,
Markdown, and optional source images under the request-specific `probes/`
directory. This is a per-example diagnostic; paper claims still require the
aggregate evaluation metrics above.

## Notes
- CIFAR-10 backbone: `openai/clip-vit-base-patch32` (frozen).
- CIFAR-100 backbone: `openai/clip-vit-base-patch16` with vision-only LoRA.
- Checkpoints save adapter/LoRA state and configuration, not frozen CLIP
  weights.
- CIFAR-100 fine-tuning metrics record runtime, throughput, and peak allocated
  GPU memory for the required Sharanga benchmark.
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
