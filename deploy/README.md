# Deploy notes

## Status boundary

The commands below deploy the current legacy `baseline_2000` demonstration
path. They are useful for integration testing but do not satisfy the canonical
baseline requirements in `docs/PLAN.md`: the worker still discovers a
`baseline_*` directory, and the Docker image copies request-local artifacts.

Do not present this deployment as the canonical research release. PLAN Phase 2
must first promote `cifar100_canonical_v1`, require its manifest id/hash in the
worker and interface, and update the Docker/Volume artifact paths together.

## Hugging Face Space (web app, free CPU tier)

1. Create a Docker Space (public), e.g. `<user>/unml-interface`.
2. Push the repository, using `deploy/Dockerfile` as the Space Dockerfile
   (copy its contents to the Space root `Dockerfile`, or set the repo layout
   to match). The current image bakes in the legacy `baseline_2000` checkpoint,
   request split metadata, and a historical comparison CSV.
3. Required Space secrets:
   - `UNML_MODAL_URL`, `UNML_JOB_SECRET`: dispatch unlearning jobs to the
     Modal worker. The Docker entry point refuses to start without both, so a
     public CPU Space never accepts impractical local training jobs.
   - `UNML_JOBS_PER_HOUR` (default 6), `UNML_PROBES_PER_MINUTE` (default 60).
   - `UNML_MAX_REMOTE_JOBS_PER_MONTH` (default 300), `UNML_JOB_TTL_DAYS`
     (default 7).

The probe runs CLIP ViT-B/16 on CPU; first request loads weights (~10 s),
subsequent probes are a few seconds.

## Modal worker (GPU jobs)

```bash
uv sync --locked --group modal
modal setup
modal volume create unml-data
modal volume create unml-hf
modal volume create unml-artifacts
modal secret create unml-secret UNML_SECRET_KEY="$UNML_JOB_SECRET"
# Legacy integration artifact only; replace after canonical promotion.
modal volume put unml-artifacts outputs/cifar100/rose_selective/baseline_2000 baseline_2000
modal run worker/modal_app.py::prepare_assets
modal deploy worker/modal_app.py
```

`prepare_assets` is a one-time CPU setup step. It downloads and extracts
CIFAR-100 once, generates deterministic splits for all 100 target classes,
caches CLIP ViT-B/16, and explicitly commits both Modal Volumes. GPU jobs run
offline and fail fast if these assets are missing; they never download data or
model weights. Run the preparation command again only after changing the
dataset, model, split configuration, or seed.

Those per-target splits are unlearning inputs. They are not the target-neutral
45k/5k development split used to select the future canonical baseline.

Generate `UNML_JOB_SECRET` once and export it locally before creating the Modal
secret. Modal injects that value into the worker as `UNML_SECRET_KEY`; the web
app keeps using `UNML_JOB_SECRET`. Do not commit either value.

The HTTP endpoint detaches each GPU invocation and immediately returns a Modal
call ID. The interface polls that ID with short signed requests, so closing a
browser or losing one long-lived HTTP connection does not cancel the GPU job.

Then run the interface with `UNML_MODAL_URL=<endpoint>` and
`UNML_JOB_SECRET=<secret>`. New classes dispatch to Modal; results land under
`outputs/cifar100/jobs/<request>_<method>_<steps>/` and are served from disk
afterwards. Each result records wall-clock time; consult the Modal dashboard
for current GPU usage and pricing.

## Security posture

- Job specs are allowlisted (`class_id` 0-99, fixed methods, steps <= 500)
  and HMAC-signed between app and worker.
- Complete remote candidate adapter states travel as safetensors and are
  validated against the baseline configuration before being written or loaded;
  no pickle deserialization of remote content.
- Adapter loading validates complete keys, shapes, and dtypes before any tensor
  is copied.
- Probe requests and image files are capped at 10 MiB before decoding, then
  checked and re-encoded through PIL.
