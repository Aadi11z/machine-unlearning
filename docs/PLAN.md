# UN-ML Development Plan

## Objective

Build a reproducible machine-unlearning research platform around a canonical
CIFAR-100 CLIP baseline, scientifically comparable unlearning experiments, and
an honest public demonstration. `PRD.md` remains the feature contract;
`IDEAS.md` remains the backlog for unsettled research questions.

## Recommended route

Use a **target-neutral vision-LoRA baseline as the first canonical baseline**.
Keep full fine-tuning, linear probing, LP-FT, WiSE-FT, and other adaptation
strategies as research comparators rather than blocking the first baseline.

This is the smallest route consistent with the existing product:

- the interface and Modal worker keep one frozen CLIP backbone resident and
  hot-swap small vision-LoRA states;
- the current unlearning methods update the same adapter family;
- rose and tulip can remain legacy evidence, while canonical demonstrations
  are rerun unless an exact compatibility check proves migration is valid; and
- pilot runs can select a defensible LoRA configuration without repeatedly
  training a permanent baseline.

A full-fine-tuned primary baseline would require a different serving and
unlearning contract: a large fine-tuned backbone, a new adapter initialization
scheme, different checkpoint compatibility rules, and methods capable of
unlearning full-model parameters. It may be valuable evidence later, but it is
not a drop-in improvement to the current platform.

## Decisions for the canonical baseline

The following define the intended baseline family. Exact LoRA and optimization
choices remain provisional until the pilot locks them:

| Concern | Decision |
|---|---|
| Dataset | CIFAR-100, all 100 classes; no forget request influences baseline data or selection. |
| Backbone | Revision-pinned `openai/clip-vit-base-patch16`. |
| Adaptation family | Vision-attention LoRA with the CLIP backbone, text tower, projections, and biases frozen. Freeze temperature by default; treat training it as an explicit pilot ablation. |
| Classifier | CLIP image-to-text similarity; do not add a trainable 100-way head to the canonical product model. |
| Resolution | 224x224, the backbone's native resolution. Do not add positional interpolation unless a separate non-224 experiment is approved. |
| Prompt contract | Use the 18 official OpenAI CIFAR-100 templates as one versioned, cached ensemble across training, evaluation, unlearning, and serving. |
| Objective | Cross-entropy over the 100 CLIP similarity logits. This is a supervised classification objective, not a claim to reproduce CLIP pretraining. |
| Precision | BF16 on a verified supported GPU; FP32 fallback for tests. FP16 requires `GradScaler`, unscale-before-clip, and resume state. |
| Test use | Never use CIFAR-100 test results for configuration selection. Evaluate the locked canonical run on the test set once. |
| Compatibility | A different backbone, prompt contract, LoRA rank, LoRA target-module set, or layer set defines a different baseline family and cannot be silently hot-swapped. The later forgotten class does not. |

## Corrections to the proposed redesign

- Upscaling CIFAR-100 from 32x32 does not recover detail, but 224x224 is the
  expected CLIP ViT-B/16 input contract. It is not a positional-embedding bug.
- OpenAI publishes 18 CIFAR-100 templates, not the 80-template ImageNet set.
- FT-CLIP's learning rates, layer decay, weight decay, warmup, augmentation,
  batch size, and epoch count were established for ImageNet full fine-tuning.
  They are useful pilot hypotheses, not validated CIFAR-100 LoRA defaults.
- Layer-wise learning-rate decay and a 10x head learning rate do not apply
  directly to the current model because its backbone is frozen and it has no
  trainable classifier head.
- A linear probe is a useful research baseline, but it is a different
  classifier architecture and is not interchangeable with text-prototype LoRA
  checkpoints in the interface.
- Exact bitwise reproducibility across PyTorch/CUDA versions and different
  hardware is not guaranteed. The project will target exact resume in one
  pinned environment and statistically reproducible results across seeds.
- WiSE-FT is a later robustness experiment, not a prerequisite for the first
  canonical LoRA baseline. For LoRA, interpolation should be defined and tested
  at the effective adapter-delta level rather than assumed equivalent to
  full-model weight interpolation.
- RandAugment, MixUp, and CutMix are optional ablations. They should not be the
  default until they improve validation results and their implications for
  example-level forgetting are documented.

## Phase 0 - Make baseline training safe

Do not launch the permanent baseline until this phase passes.

Phase 0 has two gates. Training integrity is required before pilots. Recovery
and provenance are required before the expensive final-fit run. Enforcement in
public consumers is completed when the artifact is promoted in Phase 2.

### Gate A: training integrity

#### Data contract

- Add a dedicated canonical CIFAR-100 split command that accepts no forget
  request.
- Generate a fixed, stratified 45,000/5,000 development train/validation split
  from the official 50,000 training examples using seed 42.
- Store ordered indices, labels, dataset identity, class-vocabulary hash, seed,
  generation version, and SHA-256 digest.
- Prove in tests that later rose, tulip, or arbitrary-class requests cannot
  change the canonical development split.

#### Model and evaluation contract

- Implement the 18-template CIFAR-100 ensemble; define normalization,
  aggregation, caching, and digest rules, and use the identical result in
  training, evaluation, unlearning, and serving. The current single-template
  path is incompatible and must fail rather than mix contracts.
- Change the current `train_logit_scale: true` default for canonical pilots to
  frozen. If trainable temperature is tested, record it as a separate candidate
  and include the scalar in checkpoint compatibility and manifest validation.
- Evaluate only validation metrics during pilot epochs; keep the test set out
  of the tuning loop.
- Reject missing, extra, shape-incompatible, or dtype-incompatible adapter
  tensors before any model mutation.
- Bind `best` and `last` metrics to the exact checkpoint hashes and epochs they
  describe.

#### Gate A acceptance

- Canonical split hashes reproduce from a clean checkout and remain unchanged
  across arbitrary forget requests.
- Frozen/trainable parameters and prompt prototypes match the resolved
  configuration.
- Save/load produces identical logits on a fixed batch, and an incomplete
  adapter fails before changing the model.
- Earlier-best-epoch tests prove metrics refer to the promoted checkpoint.
- One short real-GPU BF16 run passes with finite gradients and gradient
  checkpointing enabled.

### Gate B: recovery and provenance

#### Checkpoint and resume contract

- Save atomic periodic recovery checkpoints containing model, optimizer,
  scheduler, scaler when applicable, epoch, global step, sampler/DataLoader
  state needed for the supported resume boundary, and Python/NumPy/Torch/CUDA
  RNG states.
- Resume at a documented boundary. Epoch-boundary exact resume is sufficient
  for v1; do not claim exact mid-batch replay unless it is tested.
- Keep trusted internal recovery state separate from portable public artifacts.
  Public adapters remain safetensors and never require pickle deserialization.
- Save `best` and `last` separately during pilots and an explicit final-fit
  artifact for the all-50k run.

#### Provenance and submission

- Define a versioned baseline manifest containing baseline id, checkpoint hash,
  architecture and trainable-parameter schema, model/processor/tokenizer
  revisions, prompt digest, class-vocabulary digest, split digest, resolved
  config, dependency-lock digest, Git revision and dirty state, seed,
  determinism mode, hardware, and evaluation artifacts.
- Provide one tracked production baseline entry point with config validation,
  CUDA/BF16 preflight, collision protection, non-finite loss/gradient checks,
  and clear resume/promote commands. Retire or repair stale CIFAR-10-era SLURM
  and environment defaults.

#### Gate B acceptance

- Manifest generation and artifact hashes reproduce from the resolved inputs.
- An interrupted epoch-boundary run resumes with the same next-epoch order and
  learning-rate state in the pinned environment.
- The production entry point fails fast on an incompatible GPU/precision,
  output collision, stale request-specific split, or non-finite training state.

## Phase 1 - Select the LoRA configuration with bounded pilots

Do not call any configuration "final" before this pilot. Do not inspect the
CIFAR-100 test set during it.

### Stage A: adapter schema screen

Use the fixed 45k/5k development split, identical preprocessing, one seed, and
a short equal-step budget:

- Q/V LoRA with ranks 4, 8, and 16;
- Q/K/V LoRA with rank 8;
- `alpha = rank` so nominal LoRA scaling is held constant; and
- all 12 vision layers, with the text tower and temperature frozen.

Only add output-projection or MLP LoRA after these candidates fail an explicit
quality target; doing so expands the hot-swap schema and artifact size.

### Stage B: optimization and preprocessing screen

- Carry the best two schemas forward.
- Compare peak learning rates `3e-4`, `1e-3`, and the existing `3e-3`.
- Add linear warmup plus step-based cosine decay; tune warmup as a fraction of
  the actual optimizer-step budget rather than copying ImageNet epoch counts.
- Compare weight decay `1e-4`, `1e-2`, and `5e-2`; do not assume the full-model
  FT-CLIP value is optimal for LoRA matrices.
- Compare deterministic CLIP preprocessing with conservative
  random-resized-crop plus horizontal flip.
- Add a versioned LoRA-dropout field and implementation, then compare dropout
  0 and 0.05 for the leading candidate. The current `LoRALinear` has no dropout
  path, so this comparison cannot run until that bounded change is tested.
- Keep gradient clipping configurable and log pre/post-clip norms.

Use successive halving or another fixed budget so obviously poor candidates
stop early. Implement no general sweep framework beyond what this experiment
needs.

### Stage C: confirmation

- Run the leading one or two configurations with seeds 42, 123, and 456.
- Select using validation top-1, macro per-class accuracy, worst-class or lower
  quantile accuracy, run-to-run variance, stability, and adapter size.
- Prefer the smaller adapter when it is within a predeclared tolerance (for
  example 0.5 percentage points) of the best mean validation score.
- Lock the adapter schema, optimization config, prompt contract, transforms,
  and training duration before the canonical run.

### Phase 1 gate

- A machine-readable pilot report records every resolved configuration and
  seed, ranks candidates without test metrics, and names one locked winner.
- The chosen configuration meets a predeclared validation threshold and has no
  unexplained class collapse or non-finite training event.

## Phase 2 - Train and promote the canonical CIFAR-100 baseline

- Start once from the pinned pretrained CLIP revision.
- Treat the 45k/5k runs strictly as development artifacts. Then create a
  distinct final-fit artifact by training the locked configuration once on all
  50,000 CIFAR-100 training examples for the duration selected in Phase 1. Its
  manifest must record that it has no development validation partition. Do not
  use early stopping on the test set.
- Record the zero-shot model under the same preprocessing, vocabulary, and
  prompt-ensemble contract before adaptation.
- Evaluate the final locked checkpoint once on the untouched 10,000-example
  test set.
- Store top-1, top-5, loss, per-class and superclass metrics, confusion matrix,
  calibration summary, per-example labels/logits, and baseline image features
  needed by the approved unlearning evaluations. Treat embeddings as research
  data with a versioned schema, not a UI payload.
- Promote only after a clean-process reload reproduces recorded predictions
  and the manifest verifies every artifact digest.
- Require the promoted baseline id and hash in the CLI, unlearning jobs, Modal
  worker, interface catalog, registry, probe records, and demonstrations.
  Remove first-match/glob baseline selection from all consumers.
- Copy the verified baseline and manifest to Modal/HF deployment storage, then
  run one remote no-op/load smoke check before accepting unlearning jobs.

### Phase 2 gate

- One immutable `cifar100_canonical_v1` release exists under
  `outputs/cifar100/canonical/` with verified manifests and portable
  safetensors.
- Every local and remote consumer rejects another baseline id or altered hash.
- Recorded metrics are reproducible from the promoted artifact without
  depending on a request-specific split.

## Phase 3 - Establish research comparison families

These baselines answer research questions; they do not replace the production
LoRA baseline or have to share its serving format.

1. **Zero-shot CLIP:** lower-adaptation reference using the same prompt and
   preprocessing contract.
2. **Linear probe:** frozen CLIP image features plus a trained 100-way head.
   Define its optimizer and selection protocol experimentally; do not encode an
   unverified expected accuracy as an acceptance criterion.
3. **Full FT with layer-wise LR decay:** research upper comparator using a
   separately tuned CIFAR-100 recipe. FT-CLIP ImageNet settings seed the search,
   not the final values.
4. **LP-FT:** evaluate only after the linear-probe and full-FT families are
   correct and the head-to-backbone transition is specified.
5. **WiSE-FT:** assess ID and chosen distribution-shift benchmarks for the
   full-FT family. Separately define adapter-delta interpolation for LoRA.

Each family gets its own baseline id, checkpoint schema, manifests, unlearning
compatibility statement, and compute budget. Full-model unlearning is a later
method family, not silently compared as if it updated the same parameters as
LoRA-only unlearning.

### Phase 3 gate

- Each retained comparator answers a distinct hypothesis and is evaluated with
  the same data and reporting policy where comparison is valid.
- Results explicitly state when architectures, trainable parameter sets, or
  unlearning capabilities differ.

## Phase 4 - Unlearning evaluation and curated demonstrations

- Separate quick exploratory jobs from promotion-quality evaluation.
- Define held-out target, sibling, unrelated, retain, and all-class utility;
  per-class results; privacy/MIA metrics; runtime; peak memory; and artifact
  size.
- Add a retain-only retraining oracle where computationally feasible and other
  established unlearning baselines selected for the paper's research question.
- Run comparative claims over multiple seeds with uncertainty and reject
  records whose baseline, split, prompt, adapter schema, or method version
  differs.
- Classify each existing method as contribution, comparison baseline,
  ablation, or exploratory. Expose only tested method/step presets publicly.
- Re-run or migrate rose and tulip against the canonical baseline, perform full
  promotion evaluation, and store them outside expiring job artifacts.
- Retain CIFAR-10 as a secondary benchmark until a written experiment matrix
  shows it adds no useful evidence. Do not mix CIFAR-10 and CIFAR-100 artifacts.

### Phase 4 gate

- A promoted checkpoint has a complete versioned evaluation record and can be
  compared only with compatible runs.
- Rose and tulip are reproducible demonstrations, not evidence inferred from a
  single uploaded image.

## Phase 5 - Complete the product workflow

- Implement clipboard paste through the same bounded upload/decode/pixel
  validation path.
- Clarify or replace the retained-subject control; do not imply free-form VQA.
- Show quick metrics, full recorded evaluation, unavailable evidence, and
  external-image probes as visibly different evidence tiers.
- Keep the target in the candidate vocabulary and label arbitrary web images
  as qualitative, potentially out-of-distribution probes.
- Verify one complete public path: class selection, asynchronous Modal job,
  polling, validated artifact persistence, restart/reload, and image probe.
- Preserve HMAC authentication, rate limits, monthly budget, artifact TTL,
  prepared reusable Modal assets, and hosted rejection of CPU unlearning.
- Improve typography, spacing, responsiveness, accessibility, and comparison
  visuals only after the feature/evidence contract is stable.

### Phase 5 gate

- Upload and clipboard paths produce equivalent validated probes.
- A real remote job completes without per-job model/data downloads and reloads
  after a web-process restart.
- The UI never presents a qualitative score change as proof of deletion.

## Phase 6 - Research extensions

- Study semantic disentanglement and knowledge localization with explicit
  forget/allowed behaviors, matched random and gradient-based controls, and
  causal interventions on ranked parameter groups.
- Treat localization as evidence about parameters influential for behavior,
  not literal proof of where a concept is stored.
- Design multi-concept/open-vocabulary evaluation on a suitable multi-label or
  compositional dataset before adding prompt-like UI input.
- Add corruption/distribution-shift benchmarks only when they answer a stated
  robustness question.
- Preregister hypotheses, metrics, and ablations before implementing broad new
  method families.

## Verification checklist before the expensive baseline

- Canonical split balance, disjointness, digest, and target neutrality.
- Exact model, processor, tokenizer, prompts, vocabulary, and environment
  identity.
- Frozen/trainable parameter names and counts match the selected LoRA schema.
- Forward/backward BF16 smoke test with finite gradients and gradient
  checkpointing.
- Tiny-set overfit and short loss-decrease checks.
- Gradient accumulation produces the declared optimizer-step count and
  effective batch size.
- Warmup/cosine trace matches the resolved schedule.
- Save/load prediction equivalence, malformed-checkpoint rejection, and atomic
  resume.
- Same-environment deterministic smoke rerun and three-seed pilot stability.
- Zero-shot result recorded before training; no test metric used for selection.
- Promoted manifest verifies after copying to remote storage.

## Deferred decisions requiring evidence

- Exact LoRA targets, rank, alpha, dropout, learning rate, weight decay,
  warmup, augmentation, and duration: resolve in Phase 1.
- Strict deterministic mode versus faster statistical reproducibility: record
  both policy and hardware constraints before the pilot.
- Which OOD/corruption and machine-unlearning benchmarks support the intended
  paper claim: resolve before Phase 4 implementation.
- Whether full-model unlearning is in the paper scope: deciding yes authorizes
  a separate architecture and compute plan.
- Whether historical rose/tulip checkpoints can be migrated or must be rerun:
  decide after manifest compatibility inspection.

## Primary references

- [CLIP paper](https://proceedings.mlr.press/v139/radford21a.html) and
  [OpenAI CIFAR-100 prompt templates](https://github.com/openai/CLIP/blob/main/data/prompts.md)
- [FT-CLIP paper](https://arxiv.org/abs/2212.06138) and
  [official implementation](https://github.com/LightDXY/FT-CLIP)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [LP-FT](https://arxiv.org/abs/2202.10054)
- [WiSE-FT](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html)
- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html),
  [AMP guidance](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html),
  and [checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

## Research-release criteria

- explicit canonical baseline identity and complete provenance for every
  checkpoint;
- no test-set tuning and multi-seed uncertainty for comparative claims;
- complete held-out utility, privacy, efficiency, and per-class evaluation;
- immutable, reproducible demonstration artifacts;
- passing focused, full, real-GPU, remote-job, reload, and probe checks;
- clear separation of qualitative probes from experimental evidence; and
- no secrets, unsafe remote deserialization, first-match baseline selection,
  or hosted CPU-training fallback.
