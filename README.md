# UN-ML: CLIP Machine-Unlearning Research Platform

UN-ML is a research and demonstration platform for studying machine
unlearning in CLIP-based image classification. It supports CIFAR-10 and
CIFAR-100 experiments, parameter-efficient adapters, multiple unlearning
methods, privacy/utility evaluation, and a FastAPI interface backed by remote
Modal GPU jobs.

## Current status

Implemented:

- frozen CLIP backbones with historical post-projection adapters or internal
  vision LoRA;
- CIFAR-10/CIFAR-100 retain/forget experiment tooling;
- `ga_kl`, `counterfactual_rebind`, `entropy_rebind`, `h_tgsd`,
  and `h_tgsd_no_sibling_preservation`;
- hierarchy-aware metrics, membership-inference attacks, retraining-oracle and
  multi-seed tools, semantic-subspace analysis, and per-image probes;
- a FastAPI/Jinja2/HTMX interface for selecting any CIFAR-100 target, running a
  job, polling it, and comparing baseline/unlearned top-five predictions; and
- asynchronous HMAC-authenticated Modal execution with prepared data/model
  Volumes and validated safetensors transport.

Not yet complete:

- the target-neutral canonical CIFAR-100 baseline;
- the 18-template prompt contract and locked LoRA configuration;
- interruption-safe baseline training and immutable manifest promotion;
- canonical rose/tulip demonstrations and full comparative evidence;
- clipboard image paste; and
- a verified public deployment using the promoted baseline.

The active CIFAR-100 configuration is request-specific and must not be used as
the permanent baseline. Follow [`docs/PLAN.md`](docs/PLAN.md) before starting
baseline training.

## Research claim boundary

H-TGSD is designed and implemented. It uses teacher-derived text/image
directions to suppress target-specific semantic subspaces while preserving
shared/sibling and unrelated behavior. Its superiority remains a research
hypothesis until the canonical multi-seed experiments are complete. An image
probe can illustrate changed behavior; it does not prove deletion.

## Setup

The project uses uv and the checked-in lockfile:

```bash
uv sync --locked
uv run --locked pytest -q
```

Do not maintain a second pip/`venv/` environment alongside the uv-managed
`.venv`.

## Inspect the current experiment configuration

```bash
uv run --locked python scripts/run_pipeline.py --show-config
```

At present this resolves to the legacy/request-specific CIFAR-100
`flowers_superclass` workflow. It is suitable for integration checks and
historical experiments, not for creating `cifar100_canonical_v1`.

Prepare data and model assets:

```bash
uv run --locked python scripts/prepare_data.py --dataset cifar100
uv run --locked python helpers/cache_model.py --dataset cifar100
```

Run the current configured pipeline only after reviewing the warning in the
[runbook](docs/RUNBOOK.md):

```bash
uv run --locked python scripts/run_pipeline.py
```

## Local interface

```bash
uv run --env-file .env unml-interface \
  --offline \
  --device cpu \
  --baseline-checkpoint outputs/cifar100/legacy/rose_selective/baseline_2000/checkpoints/finetuned_best.pt
```

Open <http://127.0.0.1:8000>.

- With `UNML_MODAL_URL` and `UNML_JOB_SECRET`, jobs dispatch to Modal.
- Without them, non-hosted mode can run local subprocess jobs.
- Public deployments must use `--hosted`, which rejects missing remote
  credentials rather than falling back to CPU unlearning.
- The current probe accepts uploaded JPEG, PNG, and WebP images. Clipboard
  paste is planned but not implemented.

For the current Modal path, the explicit local baseline must exactly match the
checkpoint uploaded to the worker. Do not rely on catalog auto-selection: the
canonical id/hash binding that makes this automatic is still planned work.
The ignored checkpoint is not included in a clean clone; obtain or reproduce
the trusted legacy artifact before using this command.

Uploaded images are qualitative and may be outside the CIFAR-100 distribution.
“Relative confidence” is normalized across the fixed candidate labels; it is
not a calibrated probability.

## Architecture

```text
Browser (HTMX)
  -> FastAPI job manager
  -> HMAC-signed Modal endpoint
  -> detached GPU unlearning call
  -> validated safetensors adapter
  -> resident CLIP probe service
  -> baseline vs candidate predictions
```

See [`docs/flowchart.md`](docs/flowchart.md) for the current flow and planned
canonical manifest boundary.

Core modules:

- `src/unml/data.py`: datasets, class hierarchy, splits, and loaders;
- `src/unml/model.py`: CLIP wrapper, adapters, LoRA, and checkpoints;
- `src/unml/train.py`: current fine-tuning and retraining-oracle loop;
- `src/unml/unlearn.py`: unlearning execution;
- `src/unml/methods.py`: method registry and configuration;
- `src/unml/disentangle.py`: H-TGSD semantic bases and losses;
- `src/unml/attacks.py`: privacy, utility, and representation evaluation;
- `src/interface/`: web app, jobs, catalog, remote runner, and probe service;
- `worker/modal_app.py`: Modal asset preparation and detached GPU endpoint; and
- `scripts/`: CLI entry points and orchestration.

## Artifact boundaries

Current request-specific outputs live under paths such as:

```text
outputs/cifar100/legacy/flowers_superclass/
outputs/cifar100/legacy/rose_selective/
outputs/cifar100/legacy/jobs/
```

The planned immutable baseline will live under
`outputs/cifar100/canonical/`, with a versioned manifest and explicit id/hash.
Development runs, internal recovery checkpoints, expiring jobs, promoted
demonstrations, and legacy artifacts must remain distinguishable.

## Documentation

- [`docs/README.md`](docs/README.md): documentation map and authority rules;
- [`docs/PLAN.md`](docs/PLAN.md): current architecture decisions and phases;
- [`docs/PRD.md`](docs/PRD.md): requirements and acceptance evidence;
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md): verified commands and limitations;
- [`docs/IDEAS.md`](docs/IDEAS.md): unresolved research/product questions;
- [`deploy/README.md`](deploy/README.md): Modal/Hugging Face deployment status.

The local `research/` workspace remains intentionally untracked and is not a
dependency of the maintained documentation.

## Security and scientific communication

- Never commit Modal/Hugging Face credentials or HMAC values.
- Public inputs are allowlisted and bounded; remote adapters use safetensors.
- Do not deserialize externally influenced pickle checkpoints.
- Do not compare results whose baseline, split, prompt, adapter schema, or
  method version differs.
- Do not describe one changed prediction, target-score suppression, or an
  implemented hypothesis as proof of selective deletion or method superiority.
