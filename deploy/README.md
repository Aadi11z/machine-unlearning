# Deploy notes

## Hugging Face Space (web app, free CPU tier)

1. Create a Docker Space (public), e.g. `<user>/unml-interface`.
2. Push the repository, using `deploy/Dockerfile` as the Space Dockerfile
   (copy its contents to the Space root `Dockerfile`, or set the repo layout
   to match). The image bakes in the 1.2 MB baseline checkpoint, split
   metadata, and comparison CSV.
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
uv pip install modal
modal secret create unml-job-secret UNML_JOB_SECRET=<openssl rand -hex 32>
modal volume put unml-artifacts outputs/cifar100/rose_selective/baseline_2000 baseline_2000
modal deploy worker/modal_app.py
```

Then run the interface with `UNML_MODAL_URL=<endpoint>` and
`UNML_JOB_SECRET=<secret>`. New classes dispatch to Modal; results land under
`outputs/cifar100/jobs/<request>_<method>_<steps>/` and are served from disk
afterwards. Each result records wall-clock time; cost is approximately
`wall_time x $0.59/hr` on T4.

## Security posture

- Job specs are allowlisted (`class_id` 0-99, fixed methods, steps <= 500)
  and HMAC-signed between app and worker.
- Remote deltas travel as safetensors and are validated against the baseline
  configuration before being written or loaded; no pickle deserialization of
  remote content.
- Adapter loading rejects unexpected keys before any tensor is copied.
- Probe requests and image files are capped at 10 MiB before decoding, then
  checked and re-encoded through PIL.
