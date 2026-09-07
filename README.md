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

- the Phase 1 validation-only pilot and formally locked LoRA configuration;
- complete baseline provenance, exact-resume acceptance, and portable release
  artifacts required by the PRD;
- canonical rose/tulip demonstrations and full comparative evidence;
- a verified public deployment using the promoted baseline.

The canonical split, 18-template prompt contract, A100-trained baseline, and
flowers retraining oracle now exist. They support continued development, but
the artifact is not yet the fully accepted research release in `docs/PLAN.md`.

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

At present this resolves to a CIFAR-100 `flowers_superclass` development
workflow under `outputs/cifar100/development/`. It is suitable for integration
checks and development runs, not for replacing the verified baseline package.

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
  --output-root outputs
```

Open <http://127.0.0.1:8000>.

- With `UNML_MODAL_URL` and `UNML_JOB_SECRET`, jobs dispatch to Modal.
- Without them, non-hosted mode can run local subprocess jobs.
- Public deployments must use `--hosted`, which rejects missing remote
  credentials rather than falling back to CPU unlearning.
- The probe accepts uploaded or pasted JPEG, PNG, and WebP images through the
  same bounded backend validation path.

The interface loads the manifest at `UNML_BASELINE_MANIFEST` (default:
`outputs/cifar100/baseline/manifest.json`), verifies its artifacts once at
startup, and reads the class vocabulary from that manifest. A retraining oracle appears in a
separate reference table only after both checkpoints have been evaluated on
the same target-specific test split. `--baseline-checkpoint` remains a legacy
compatibility override and does not establish canonical provenance.

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

Historical request-specific outputs are preserved under paths such as:

```text
outputs/cifar100/archive/legacy/flowers_superclass/
outputs/cifar100/archive/legacy/rose_selective/
```

The immutable baseline lives under `outputs/cifar100/baseline/` as one package
containing `manifest.json`, its checkpoint, and metrics. Set
`UNML_BASELINE_MANIFEST` to the mounted package manifest in each deployment;
the Modal worker must receive its own path because it has a separate filesystem.
Development runs live below `outputs/cifar100/development/<request>/`,
identity-bound interface jobs below `outputs/cifar100/jobs/`, and historical
artifacts below `outputs/cifar100/archive/legacy/`. The latter are explicitly
precomputed demonstrations, not fresh unlearning results.

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
