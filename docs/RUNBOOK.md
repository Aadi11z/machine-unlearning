# UN-ML Runbook

This runbook documents commands supported by the current checkout. It does not
turn planned requirements into executable features. Read [`PLAN.md`](PLAN.md)
before baseline work.

## 1. Critical baseline warning

The current `config/parameters.yaml` resolves to the request-specific
`flowers_superclass` experiment and writes under:

```text
outputs/cifar100/legacy/flowers_superclass/
```

It still uses a request-dependent development split, one prompt template, a
trainable replacement temperature, non-resumable fine-tuning, and directory-
based baseline discovery in some consumers. It is useful for historical
experiments and integration checks, but it is **not** the canonical CIFAR-100
baseline pipeline described in `PLAN.md`.

Do not start the permanent baseline until Phase 0 and Phase 1 are implemented
and their gates pass. There is intentionally no canonical promotion command in
this runbook yet.

The canonical development entry point is now explicit and isolated by run ID:

```bash
uv run --locked python scripts/generate_canonical_split.py \
  --data-dir data \
  --run-id 2026-08-29-gpu-a \
  --download

uv run --locked python scripts/train_canonical_baseline.py \
  --data-dir data \
  --split-path outputs/cifar100/canonical/development/2026-08-29-gpu-a/splits/cifar100_canonical_development_v1.json \
  --run-id 2026-08-29-gpu-a
```

The paired Sharanga job trains the final canonical baseline, creates a
target-specific oracle split, and trains the retraining oracle:

```bash
export UNML_ROOT="$PWD"
export UNML_DATA_DIR="$PWD/data"
export UNML_RUN_ID=2026-08-29-gpu-a
export UNML_ORACLE_REQUEST=flowers_superclass
export UNML_ORACLE_FORGET_CLASSES=54,62,70,82,92
sbatch src/hpc/submit_canonical_pair.slurm
```

It writes the canonical release to
`outputs/cifar100/canonical/cifar100_canonical_v1/` and the target-specific
oracle to `outputs/cifar100/canonical/retraining_oracle/<run-id>/<request>/`.

This command performs development training only (`evaluate_test=false`). It
does not create or overwrite the promoted `cifar100_canonical_v1` release;
promotion is a later, separately verified step.

## 2. Local setup

Use the checked-in uv lockfile and the project `.venv`:

```bash
uv sync --locked
uv run --locked pytest -q
```

Do not create a second `venv/`, install from `requirements.txt`, or maintain a
parallel pip environment. `uv sync --locked` is the dependency source of truth.

Useful read-only checks:

```bash
uv run --locked python scripts/run_pipeline.py --show-config
uv run --locked python scripts/train_vlm.py --help
uv run --locked python scripts/unlearn_job.py --help
uv run --locked unml-interface --help
```

If sandboxed tooling cannot write the normal uv cache, point `UV_CACHE_DIR` to
a writable temporary directory. That is an execution-environment workaround,
not a dependency change.

## 3. Data and model cache

Prepare the currently selected request split:

```bash
uv run --locked python scripts/prepare_data.py
```

Prepare CIFAR-100 and cache CLIP ViT-B/16 explicitly:

```bash
uv run --locked python scripts/prepare_data.py --dataset cifar100
uv run --locked python helpers/cache_model.py --dataset cifar100
```

The cache helper downloads the processor, tokenizer, and model when needed and
then verifies an offline reload. Use `--offline` on CLIs that expose it; for
other GPU commands set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` only
after that verification succeeds.

Current per-target Modal data/model preparation is documented in
[`../deploy/README.md`](../deploy/README.md).

## 4. Current request-specific experiment workflow

This section runs the existing pipeline; it does not create
`cifar100_canonical_v1`.

Select the dataset and request in `config/parameters.yaml`, then inspect the
resolved paths before running anything:

```bash
uv run --locked python scripts/run_pipeline.py --show-config
```

Current supported request examples include:

- `flowers_superclass`: forget all five CIFAR-100 flower classes; and
- `rose_selective`: forget rose while treating the other flowers as siblings.

Run individual stages:

```bash
uv run --locked python scripts/prepare_data.py
uv run --locked python scripts/train_vlm.py --offline --device cuda
uv run --locked python scripts/run_unlearning.py \
  --method counterfactual_rebind \
  --device cuda
uv run --locked python scripts/evaluate_attacks.py \
  --device cuda
```

Run the configured historical pipeline:

```bash
uv run --locked python scripts/run_pipeline.py
```

That command may run fine-tuning, the retained-data oracle, all configured
unlearning methods, and evaluation. Confirm the resolved stage list first.
Neither `--resume` at the pipeline level nor a surviving output directory
currently restores optimizer/scheduler/RNG state inside an interrupted
fine-tuning run.

## 5. Safe smoke checks

Run preflight on the actual GPU node before allocating a real experiment:

```bash
uv run --locked python helpers/preflight_run.py \
  --dataset cifar100 \
  --request flowers_superclass \
  --require-cuda \
  --require-venv \
  --require-offline-cache
```

Run the bounded legacy training smoke test:

```bash
uv run --locked python scripts/train_vlm.py \
  --dataset cifar100 \
  --request flowers_superclass \
  --smoke \
  --offline \
  --device cuda
```

Smoke artifacts are partial infrastructure evidence. Never promote them or use
their metrics as baseline/unlearning research results.

The planned Phase 0 GPU gate is stricter: it must also verify the canonical
split and prompt contract, frozen/trainable parameters, non-finite handling,
checkpoint/metric binding, strict reload, and recovery behavior.

## 6. One headless unlearning job

Resolve a job without touching data or models:

```bash
uv run --locked python scripts/unlearn_job.py \
  --forget-class tulip \
  --method ga_kl \
  --steps 50 \
  --dry-run
```

A real job requires a compatible baseline checkpoint:

```bash
uv run --locked python scripts/unlearn_job.py \
  --forget-class tulip \
  --method ga_kl \
  --steps 50 \
  --baseline-checkpoint /path/to/finetuned_best.pt \
  --device cuda
```

Until Phase 2, compatibility is not the same as canonical provenance. Record
the exact checkpoint path/hash and do not compare jobs produced from different
request-specific baselines as if they shared one baseline.

## 7. Local web interface

Cache model assets first, then run:

```bash
uv sync --locked
uv run --env-file .env unml-interface \
  --offline \
  --device cpu \
  --baseline-checkpoint outputs/cifar100/legacy/rose_selective/baseline_2000/checkpoints/finetuned_best.pt
```

Open <http://127.0.0.1:8000>.

- With both `UNML_MODAL_URL` and `UNML_JOB_SECRET`, new jobs dispatch to Modal.
- Without them, non-hosted local mode can run jobs as subprocesses.
- `--hosted` requires remote credentials and refuses local CPU unlearning.
- When using Modal, `--baseline-checkpoint` is mandatory in practice: it must
  be the exact local copy of the checkpoint uploaded to the worker. The example
  matches the legacy deployment recipe. If the worker Volume contains another
  checkpoint, change both sides together.

Do not run a Modal-backed probe when that alignment is unknown. The current
remote result validates adapter schema and shape but does not prove which
baseline hash produced it. Local catalog discovery can otherwise select a
different request-local baseline. PLAN Phase 2's id/hash enforcement is the
required permanent fix.

The example checkpoint lives under ignored `outputs/` and is not present in a
clean clone. Obtain the trusted legacy artifact from the project's archive or
reproduce it with the historical experiment workflow before starting the
interface; then verify that its SHA-256 is identical to the worker upload.

The current probe accepts uploaded JPEG, PNG, or WebP files. Clipboard paste is
planned, not implemented. Uploaded images are qualitative probes and may be
out-of-distribution for CIFAR-100.

## 8. Modal and public deployment

Use [`../deploy/README.md`](../deploy/README.md) for the current Modal commands,
secret names, and Volume setup.

The current deployment assets use the legacy `baseline_2000` directory. Do not
publish that as the canonical research release. PLAN Phase 2 must first:

1. promote `cifar100_canonical_v1`;
2. copy its adapter and manifest together;
3. remove first-match baseline selection;
4. require the same id/hash in the worker and interface; and
5. pass a remote load/job/reload/probe smoke test.

## 9. Artifact boundaries

Current request-specific layout:

```text
outputs/
  cifar10/
  cifar100/
    legacy/
      flowers_superclass/
        splits/
        finetune/
        retrain_oracle/
        unlearning/
        comparison/
      rose_selective/
      jobs/
```

Planned canonical layout:

```text
outputs/cifar100/canonical/
  development/
    <dated-run>/
  pilots/
    <dated-run>/
  cifar100_canonical_v1/
    adapter.safetensors
    manifest.json
    metrics/
    predictions/
```

The exact final filenames are part of Phase 0's manifest/schema work. Do not
create ad hoc filenames and treat this illustrative tree as implemented.

Keep these tiers distinct:

- trusted internal recovery checkpoints;
- development/pilot artifacts;
- immutable final-fit baseline artifacts;
- expiring exploratory jobs;
- immutable promoted demonstrations; and
- legacy request-specific artifacts.

The former `retain_only` unlearning method is archived with legacy outputs and
is no longer part of the supported unlearning-method registry. New studies use
the canonical baseline and target-specific retraining oracle as the two
retain-data reference runs; existing `retain_only` checkpoints remain readable
as historical artifacts but are not regenerated by the pipeline.

## 10. Current evaluation and probes

Evaluate configured candidates:

```bash
uv run --locked python scripts/evaluate_attacks.py \
  --device cuda
```

Inspect representative examples:

```bash
uv run --locked python helpers/probe_checkpoint.py \
  --class-name rose \
  --limit-per-class 10 \
  --offline \
  --device cuda
```

These tools provide hierarchy-aware accuracy, per-class/confusion outputs,
membership-inference evidence, semantic-subspace analysis when applicable, and
per-example diagnostics. Artifact presence proves that a computation ran; it
does not by itself establish selective unlearning or method superiority.

## 11. CIFAR-10 and legacy studies

CIFAR-10 remains supported by parts of the experiment code. The current
`run_study.py` workflow fine-tunes within request/seed cells and therefore does
not satisfy the new single-canonical-baseline contract. Treat it as legacy
until Phase 4 defines CIFAR-10's role and the study runner validates explicit
baseline manifests.

Do not delete legacy scripts or artifacts solely because the new plan
supersedes them; first determine whether they are required for reproducibility.

## 12. Sharanga and SLURM status

The checked-in `env_activation.sh` now activates the locked `.venv` created by
`uv sync --locked` and sets project-local uv/Hugging Face caches. Several
`src/hpc/*.slurm` files still contain CIFAR-10-era flat paths or
request-specific assumptions. Those wrappers are not the production
canonical-baseline entry point required by PLAN Phase 0B.

Before canonical cluster training:

- provide one tracked uv/lockfile-based submission path;
- set data, outputs, and Hugging Face cache explicitly;
- run the canonical CUDA/BF16/offline-cache preflight on the allocated node;
- record Git dirty state, dependency lock, CUDA, GPU, SLURM job, and resolved
  configuration; and
- verify recovery and collision behavior with a bounded job.

Generic cluster operations such as allocation commands, partitions, QoS,
scratch retention, and quotas are site/user-specific and should be confirmed
with the current Sharanga documentation rather than frozen into this project
runbook.

## 13. Troubleshooting

### uv cache permission errors

If `uv run` fails before Python starts because a sandbox cannot write the user
cache, rerun in the normal shell or use a writable temporary `UV_CACHE_DIR`.
Do not replace the lockfile workflow with pip.

### Slow first remote job

Check whether CIFAR-100, CLIP, the target split, and baseline artifact already
exist in their Modal Volumes. A normal prepared job should not download the
169 MB dataset or fine-tune a baseline.

### Offline model failure

Run `helpers/cache_model.py --dataset cifar100` with network access, then repeat
the offline reload. Do not allow a GPU job to silently download core assets.

### Interrupted fine-tuning

The current trainer cannot resume a scientifically identical run. Preserve
logs and partial artifacts for diagnosis, but restart only an exploratory run.
Implement and verify Phase 0B before the permanent baseline.

### Missing held-out metrics in the interface

Quick jobs may not have promotion-quality evaluation. Render the values as not
evaluated; do not infer them from training metrics or an uploaded image.

## 14. Security and artifact policy

- Never commit secret values, personal tokens, or absolute user paths.
- Remote artifacts must remain safetensors and pass complete identity/key/
  shape/dtype validation before persistence or activation.
- Trusted internal `.pt` recovery files must never enter an externally
  influenced public deserialization path.
- Do not retain uploaded images by default.
- Treat scratch and Modal Volumes as working storage, not the sole archival
  copy of promoted research artifacts.
