# UN-ML Product Requirements Document

## 1. Product definition

UN-ML is a machine-unlearning research and demonstration platform for
CLIP-based image classification. It combines reproducible experiment tooling
with a public workflow in which a visitor chooses a CIFAR-100 class, runs an
approved unlearning job remotely, and compares a fine-tuned baseline with the
unlearned model on an image.

This document defines required behavior and acceptance evidence.
[`PLAN.md`](PLAN.md) defines delivery order and architecture decisions;
[`RUNBOOK.md`](RUNBOOK.md) documents what can be run in the current checkout;
[`IDEAS.md`](IDEAS.md) contains unresolved questions.

## 2. Current implementation boundary

Implemented foundation:

- request-specific CIFAR-10 and CIFAR-100 experiment pipelines;
- frozen CLIP backbones with post-projection adapters or vision LoRA;
- six unlearning method identifiers, hierarchy-aware evaluation,
  membership-inference attacks, per-image probes, and multi-seed study tools;
- a FastAPI/Jinja2/HTMX interface with HTML and JSON endpoints;
- asynchronous HMAC-signed Modal submission and polling, prepared data/model
  Volumes, complete adapter-state safetensors transport, rate limits, budgets,
  and artifact TTL;
- bounded image uploads and atomic candidate-adapter activation; and
- historical rose outputs and a completed exploratory tulip job.

Not yet implemented:

- the target-neutral canonical CIFAR-100 development split and final-fit
  baseline pipeline;
- the 18-template prompt contract across training, unlearning, and serving;
- the locked LoRA configuration selected by bounded pilots;
- interruption-safe fine-tuning resume and immutable baseline promotion;
- explicit canonical baseline id/hash enforcement in every consumer;
- canonical rose/tulip demonstrations with complete evaluation;
- clipboard image paste; and
- a verified public deployment using the promoted canonical baseline.

The active CIFAR-100 configuration remains request-specific, uses one prompt
template, and trains a replacement temperature scalar. It is valid for legacy
experiments but must not be described as the permanent canonical pipeline.

## 3. Users and journeys

### Researcher

Builds or selects an immutable baseline family, runs compatible unlearning
methods under versioned configurations, evaluates multiple seeds, and exports
records suitable for analysis and publication.

### Demo visitor

Selects a CIFAR-100 target, registered method, and bounded step count, follows
remote progress, uploads an image, and compares baseline and unlearned
predictions with honest evidence labels. Clipboard paste is planned but is not
part of the current journey.

### Maintainer

Prepares reusable remote assets, promotes verified checkpoints, deploys the
web app, manages quotas and retention, and diagnoses failures without exposing
secrets or silently changing the baseline.

## 4. Goals

- Make every canonical CIFAR-100 unlearning result traceable to one fixed,
  target-neutral vision-LoRA baseline.
- Select the baseline configuration with bounded validation-only pilots before
  performing one all-50k final fit.
- Preserve model, prompt, data, environment, configuration, seed, checkpoint,
  and metric identity.
- Separate exploratory jobs, qualitative probes, promotion evaluation, and
  comparative research claims.
- Keep rose and tulip as stable demonstrations after compatibility validation
  or rerunning against the canonical contract.
- Retain CIFAR-10 until its benchmark role is decided.
- Keep the public job and probe workflow safe, bounded, and understandable.

## 5. Non-goals for the canonical-baseline release

- Claiming certified deletion, literal erasure of frozen-backbone knowledge, or
  proven superiority before matched multi-seed experiments exist.
- Treating one image or changed prediction as proof of unlearning.
- Making full fine-tuning, linear probing, LP-FT, or WiSE-FT share the product
  checkpoint contract; these are separate research families.
- Implementing full-model unlearning without a separately approved scope.
- Turning CLIP into free-form VQA or caption generation.
- Accepting arbitrary models, checkpoints, URLs, paths, code, methods, or
  hyperparameters from public users.
- Adding accounts, billing, a general database, or CIFAR-10 web jobs before
  they solve a defined requirement.

## 6. Requirement status

- **Exists:** verified in the current implementation and must remain working.
- **Required:** committed work that is not necessarily implemented yet.
- **Research:** requires a scoped experiment or decision before implementation.

## 7. Functional requirements

### 7.1 Canonical baseline integrity

| ID | Status | Requirement | Acceptance evidence |
|---|---|---|---|
| BASE-01 | Required | Generate a target-neutral, seed-42, stratified 45k/5k CIFAR-100 development split; keep the official 10k test set untouched. | Ordered indices, labels, vocabulary, generator version, and digest reproduce and do not change with a later forget request. |
| BASE-02 | Required | Define one revision-pinned ViT-B/16 vision-LoRA family with frozen backbone, text tower, projections, and biases. Freeze temperature by default and treat training it as a separate pilot candidate. | A trainable-parameter audit and manifest distinguish every baseline family. |
| BASE-03 | Required | Implement the 18 official CIFAR-100 templates as one versioned normalized ensemble across training, evaluation, unlearning, and serving. | Prototype tensors and prompt digest match in every component; single-template artifacts are rejected as incompatible. |
| BASE-04 | Required | Add strict, atomic checkpoint validation and bind `best`, `last`, and their metrics to exact hashes and epochs. | Missing, extra, wrong-shape, wrong-dtype, and earlier-best regression tests pass without partial mutation. |
| BASE-05 | Required | Add atomic epoch-boundary recovery containing optimizer, scheduler, scaler when applicable, RNG, epoch/global-step, and supported DataLoader/sampler state. | An interrupted run resumes with the same next-epoch data order and LR state in the pinned environment. |
| BASE-06 | Required | Provide one tracked baseline entry point with config validation, CUDA/BF16 preflight, collision protection, and non-finite checks. | Incompatible precision, request-specific split, dirty output collision, or non-finite state fails before promotion. |
| BASE-07 | Required | Run a bounded validation-only LoRA pilot over the candidates defined in PLAN Phase 1, then lock one schema and schedule. | Machine-readable report contains all configs/seeds and no CIFAR-100 test metric. |
| BASE-08 | Required | Produce a distinct final-fit artifact by training the locked configuration once on all 50k training examples, then evaluate the untouched test set once. | Manifest distinguishes development and final-fit artifacts and contains aggregate, per-class, superclass, calibration, logits, and required feature records. |
| BASE-09 | Required | Promote `cifar100_canonical_v1` with immutable checkpoint and artifact digests. | Clean-process reload reproduces recorded predictions and manifest verification detects any changed input or artifact. |
| BASE-10 | Required | Require the promoted baseline id/hash in CLI, worker, catalog, registry, jobs, probes, demos, and deployment. | No consumer selects a baseline by first-match glob or request-local directory inference. |
| BASE-11 | Required | Record zero-shot CLIP under the same vocabulary, prompt, and preprocessing contract before adaptation. | Zero-shot results are stored as a comparator, not an acceptance threshold copied from literature. |

### 7.2 Job execution and identity

| ID | Status | Requirement | Acceptance evidence |
|---|---|---|---|
| JOB-01 | Exists | Accept any CIFAR-100 target and derive its class and sibling metadata server-side. | Invalid class ids or names fail before dispatch. |
| JOB-02 | Exists | Submit Modal work asynchronously and poll a detached call id. | Closing the browser does not cancel the GPU call; polling reaches a terminal state. |
| JOB-03 | Required | Include dataset, target, baseline id/hash, method id/version, seed, preset, and effective-config digest in job identity. | Behavior-changing configurations cannot collide or overwrite artifacts. |
| JOB-04 | Required | Expose only approved method/preset combinations publicly while retaining explicit research CLI controls. | UI, API, and worker reject the same unsupported inputs. |
| JOB-05 | Required | Return safe, actionable terminal states for timeout, quota, missing asset, invalid response, cancellation, and GPU failure. | Tests cover every failure without leaking secrets or leaving a false running state. |

### 7.3 Artifacts and demonstrations

| ID | Status | Requirement | Acceptance evidence |
|---|---|---|---|
| ART-01 | Exists | Transport the complete candidate adapter state as safetensors and validate identity, complete keys, shape, and dtype before persistence or activation. | Malformed payload tests leave the resident model unchanged. |
| ART-02 | Required | Separate expiring jobs, development artifacts, final-fit baselines, and immutable demonstrations. | TTL cleanup cannot delete promoted artifacts and the catalog does not infer tier from a name alone. |
| ART-03 | Required | Preserve historical rose/tulip outputs as explicitly legacy until exact compatibility is proven. | Legacy records identify their request-specific split and single-template contract. |
| ART-04 | Required | Promote canonical rose and tulip demonstrations by rerunning or by a documented exact-contract migration. | Each records baseline, target, method version, seed, config and artifact digests, runtime, and complete evaluation. |
| ART-05 | Required | Keep trusted internal recovery state separate from public portable artifacts. | Public paths never deserialize externally influenced pickle data. |

### 7.4 Evaluation and benchmarks

| ID | Status | Requirement | Acceptance evidence |
|---|---|---|---|
| EVAL-01 | Required | Version exploratory and promotion evaluation as different tiers. | UI and result schemas never imply quick metrics are complete. |
| EVAL-02 | Required | Promotion evaluation reports held-out target, sibling, unrelated, retain, all-class and per-class utility; privacy/MIA; runtime; peak memory; and artifact size. | Missing values are explicit `not_evaluated` or justified `not_applicable`. |
| EVAL-03 | Required | Aggregate comparative claims over recorded seeds with uncertainty. | Incompatible baseline, split, prompt, schema, method version, or config records are rejected. |
| EVAL-04 | Required | Evaluate the baseline test set only after pilot selection and evaluate unlearning on fixed held-out and forget-train contracts. | Test metrics do not appear in pilot ranking inputs. |
| EVAL-05 | Required | Define CIFAR-10 as a full benchmark, regression dataset, or legacy evidence before removing or expanding it. | Written matrix gives its hypothesis, methods, metrics, and comparability limits. |
| EVAL-06 | Research | Select established unlearning comparators, retraining-oracle coverage, and any compositional/OOD datasets needed by the paper. | Each addition answers a predeclared question and has a bounded compute plan. |

### 7.5 Method governance

| ID | Status | Requirement | Acceptance evidence |
|---|---|---|---|
| METHOD-01 | Required | Classify `retain_only`, `ga_kl`, `counterfactual_rebind`, `entropy_rebind`, `h_tgsd`, and `h_tgsd_no_sibling_preservation` as contribution, comparator, ablation, or exploratory. | Every retained method answers a distinct question. |
| METHOD-02 | Required | Give each method a versioned schema, one research default, and a small justified search space. | Unknown or irrelevant fields fail and resolved settings are stored. |
| METHOD-03 | Required | Publish fewer tested presets than research configurations. | UI, API, worker, and documentation expose the same presets. |
| METHOD-04 | Required | Archive or remove a method only after checking scripts, tests, reports, and artifact reproducibility. | Removal includes a migration note and leaves no unexplained broken references. |

### 7.6 Probe and interface

| ID | Status | Requirement | Acceptance evidence |
|---|---|---|---|
| PROBE-01 | Exists | Upload JPEG, PNG, or WebP and compare baseline/unlearned top-five scores over the fixed CIFAR-100 vocabulary. | HTML and JSON routes enforce request-byte, decode, and pixel limits. |
| PROBE-02 | Required | Paste an image through the same backend validation path. | Paste and upload produce equivalent validated multipart requests. |
| PROBE-03 | Required | Show target score/rank changes, raw top-five tables, candidate identity, evidence tier, and available recorded metrics. | Qualitative, quick, missing, and promotion evidence remain distinguishable. |
| PROBE-04 | Required | Keep the forgotten target in the candidate vocabulary and label external images as qualitative/OOD probes. | The interface cannot manufacture forgetting by removing the target prompt. |
| PROBE-05 | Required | Remove, constrain, or clearly explain the retained-subject field. | It cannot be mistaken for a free-form VQA prompt. |
| PROBE-06 | Research | Design controlled open-vocabulary or multi-concept scoring on suitable data. | Candidate prompts, controls, retained/forgotten concepts, and metrics are defined before UI work. |
| UX-01 | Required | Present select, run, poll, and probe as one clear responsive journey. | Desktop/mobile browser and keyboard checks pass. |
| UX-02 | Required | Report only backend-supported queued, preparing, running, evaluating, completed, failed, and expired states. | Copy never labels asset preparation as optimizer progress. |
| UX-03 | Required | Improve hierarchy, typography, spacing, tables, contrast, focus, and status announcements after the feature contract stabilizes. | Accessibility and interaction checks pass without hiding raw evidence. |

### 7.7 Deployment and operations

| ID | Status | Requirement | Acceptance evidence |
|---|---|---|---|
| OPS-01 | Exists | Preserve HMAC authentication, allowlisted inputs, rate limits, monthly budget, TTL, bounded uploads, and hosted remote-only execution. | Focused security/configuration tests pass. |
| OPS-02 | Exists | Reuse prepared CIFAR-100 data, per-target splits, and CLIP cache from Modal Volumes. | A normal current job does not redownload core data/model assets. |
| OPS-03 | Required | Replace the current legacy baseline Volume artifact with the promoted canonical baseline and manifest. | Remote execution verifies the same baseline id/hash as the web deployment. |
| OPS-04 | Required | Deploy the web app only after canonical manifest enforcement. | Health, job, poll, persistence, restart/reload, and probe checks pass at the public URL. |
| OPS-05 | Required | Record wall time and resource settings and benchmark training/evaluation batch choices. | Performance claims cite repeatable measurements. |
| OPS-06 | Required | Document secret names without values and fail fast on missing/mismatched configuration. | Source, artifacts, responses, and logs expose no secret value. |

### 7.8 Research comparison and extensions

| ID | Status | Requirement | Acceptance evidence |
|---|---|---|---|
| COMP-01 | Research | Implement zero-shot, linear-probe, full-FT/LLRD, LP-FT, and WiSE-FT only as separately identified comparator families. | Results state architecture, trainable parameters, checkpoint schema, and valid comparison limits. |
| COMP-02 | Research | Define adapter-delta interpolation separately from full-model WiSE-FT. | The operation is mathematically specified and validated before robustness claims. |
| RES-01 | Research | Test semantic disentanglement and localization with matched random, low-ranked, gradient-based, and unrestricted controls. | Forget and allowed behavior are evaluated separately under matched budgets. |
| RES-02 | Research | Relate every extension to H-TGSD and isolate the new contribution. | A predeclared ablation distinguishes reused and new components. |
| RES-03 | Research | Describe localization as influential-component evidence, not literal storage/deletion proof. | Papers, reports, and interface copy preserve the boundary. |

## 8. Non-functional requirements

### Reproducibility

- Pin model, processor/tokenizer, prompt, vocabulary, data, code, dependency,
  configuration, seed, hardware, checkpoint, and metric identities.
- Target exact epoch-boundary resume in one pinned environment and statistical
  reproducibility across recorded seeds; do not promise cross-platform bitwise
  identity.
- Never weaken validation to make historical artifacts appear compatible.

### Security and privacy

- Never accept arbitrary remote code, paths, URLs, pickle payloads, model
  identifiers, or unrestricted hyperparameters.
- Do not retain uploaded images by default.
- Never expose HMAC keys, Modal credentials, Hugging Face credentials, or secret
  values.

### Reliability and cost

- Reset the resident model to the verified baseline after every probe,
  including failure paths.
- Persist completed jobs so they remain probeable after process restart.
- Bound timeouts, quotas, polling, uploads, and artifact retention.
- Reuse prepared assets and the resident backbone; measure latency and cost
  before publishing targets.

### Scientific communication

- Call fixed-vocabulary normalized scores “relative confidence,” not calibrated
  probability.
- Keep image probes, quick metrics, held-out evaluation, privacy results, and
  conclusions visibly distinct.
- Do not present implemented hypotheses as demonstrated improvements.

## 9. Delivery alignment

| PLAN phase | Required outcome |
|---|---|
| Phase 0A | Canonical development split, prompt/model contract, strict checkpoints, metric binding, and real-GPU integrity smoke. |
| Phase 0B | Atomic recovery, manifest generation, production preflight, and collision-safe submission. |
| Phase 1 | Validation-only bounded LoRA pilot and locked configuration. |
| Phase 2 | All-50k final fit, one test evaluation, immutable promotion, explicit consumer enforcement, and remote copy verification. |
| Phase 3 | Separately identified zero-shot, linear-probe, full-FT, LP-FT, and WiSE-FT research families. |
| Phase 4 | Versioned unlearning evaluation, method governance, CIFAR-10 decision, and canonical rose/tulip demos. |
| Phase 5 | Clipboard input, evidence-complete interface, canonical public deployment, and later UX refinement. |
| Phase 6 | Semantic disentanglement, localization, multi-concept, and robustness research extensions. |

## 10. Canonical release acceptance

The first canonical research demo is accepted only when:

1. Phase 0 and Phase 1 gates pass before the final-fit job starts.
2. `cifar100_canonical_v1` verifies from a clean process and is selected by
   explicit id/hash everywhere.
3. No test result contributed to pilot selection.
4. Every promoted unlearning checkpoint has compatible provenance and complete
   or explicitly unavailable evaluation fields.
5. Canonical rose and tulip demonstrations survive normal job TTL cleanup.
6. UI, API, worker, CLI, and documentation expose the same approved presets.
7. Focused, full, GPU, remote, reload, and probe checks pass without weakening
   assertions.
8. Hosted mode cannot train locally on CPU, redownload core assets per job,
   deserialize untrusted pickle, or expose a secret.
9. The interface never treats a qualitative prediction change as proof of
   deletion.

## 11. Decisions still requiring authority or evidence

- The winning Phase 1 LoRA schema, optimizer schedule, transform, dropout, and
  training duration.
- Strict deterministic mode versus faster statistical reproducibility.
- Method classifications, public presets, and promotion thresholds.
- CIFAR-10, OOD/corruption, compositional-dataset, and full-model-unlearning
  scope.
- Whether historical rose/tulip checkpoints pass exact compatibility or must
  be rerun.
- Public hosting owner/name, artifact distribution, quotas, and retention.
