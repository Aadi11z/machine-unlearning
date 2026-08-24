# Ideas and Open Questions

This file contains unsettled hypotheses and product questions. The canonical
baseline safety work and LoRA pilot are committed requirements in `PLAN.md` and
`PRD.md`, not backlog ideas.

## Adaptation and comparison families

- How much scientific value do full fine-tuning, linear probing, LP-FT, and
  WiSE-FT add beyond the production vision-LoRA baseline?
- Should the paper include full-model unlearning, or explicitly scope its
  unlearning claims to the adapter family?
- Which distribution-shift data would make WiSE-FT or adapter-delta
  interpolation meaningful rather than decorative?
- Which comparator families justify their compute and maintenance cost?

### Retraining oracle comparison

- Add a retain-only retraining oracle: restart from the same adapter
  initialization and retrain on the retain data without the forgotten class.
- Compare each unlearning method with that oracle on forgotten-class accuracy,
  retained and sibling utility, membership-inference resistance, prediction or
  embedding distance, wall-clock time, GPU-hours, optimization steps, memory,
  and artifact size.
- Optionally add a full-model retain-only retraining oracle as a stronger but
  less directly matched reference. Keep it separate from the matched LoRA
  oracle because it changes the number of trainable parameters and compute.
- Present the oracle as a later benchmark and evidence target, not as a claim
  that unlearning is faster or equivalent until the measurements are complete.

These models require separate baseline identities and checkpoint contracts.
They should not be presented as hot-swappable alternatives to the canonical
vision-LoRA baseline.

## Working demonstration artifacts

- Preserve historical rose and tulip artifacts as legacy evidence.
- Decide whether exact compatibility checks can migrate either artifact, or
  whether both must be rerun against the canonical baseline and 18-template
  prompt contract.
- Prefer canonical demonstrations produced by the proposed method, while
  retaining appropriate comparison methods and complete held-out metrics.
- Keep an individual image probe qualitative. Selective forgetting claims must
  come from aggregate held-out, privacy, and retained-utility evaluation.

## Dataset scope

- Keep CIFAR-10 until its research role is explicitly decided.
- Decide whether CIFAR-10 is a full secondary benchmark, a cheap regression
  dataset, or legacy-only evidence.
- Identify whether a compositional or multi-label dataset is needed for
  semantic selectivity and hybrid-scene evaluation that CIFAR cannot support.
- Choose datasets because they test a distinct hypothesis, not merely to add
  tables.

## Benchmarks and evaluation

- Select established unlearning baselines and retraining-oracle coverage that
  match the paper's claims and compute budget.
- Decide which privacy attacks complement the implemented confidence- and
  delta-based membership inference evaluation.
- Define corruption or out-of-distribution benchmarks only after stating the
  robustness question.
- Decide promotion thresholds for forgetting, sibling/unrelated utility,
  privacy, runtime, memory, checkpoint size, and multi-seed stability.

## Semantic disentanglement and knowledge localization

- Can target-specific directions be separated from shared or sibling semantic
  directions more reliably than the current H-TGSD construction?
- Which localization unit is most meaningful: parameter groups, LoRA
  directions, activations, gradients, influence approximations, or learned
  representation subspaces?
- Compare high-ranked interventions with low-ranked, random, gradient-based,
  and unrestricted controls under the same update budget.
- Measure unwanted and allowed behavior separately. Treat localization as
  evidence about components influential for behavior, not proof that a concept
  is literally stored or deleted there.
- Require every extension to state what it reuses from H-TGSD and which
  ablation isolates the new contribution.

## Method consolidation

- Classify every current method as contribution, comparator, ablation, or
  exploratory.
- Retain a method only when it answers a distinct experimental question or is
  required to reproduce an existing artifact.
- Choose one documented research default and a small justified search space
  for every retained method.
- Expose fewer tested public presets than research configurations.

Possible hypotheses, pending that audit:

- semantically nearest rather than random counterfactual rebinding;
- curriculum counterfactual selection;
- confidence- or margin-weighted forget samples;
- prototype-anchored rebinding;
- retained-geometry constraints; and
- adversarial separation of shared and forget-sensitive representations.

## Interface and probe questions

- Clipboard paste is a committed Phase 5 feature, not an open design question.
  The unresolved question is how to communicate paste/upload validation and
  failure states clearly.
- Should the retained-subject field be removed, restricted to a candidate
  vocabulary, or replaced by an explicit multi-concept experiment?
- CLIP scores image/text similarity; it does not answer free-form visual
  questions. A prompt box therefore needs a defined candidate set, controls,
  and metrics rather than VQA-style copy.
- CIFAR-100 includes `woman` and `rose`, but not `dog`, and its images are
  single-label and low-resolution. Hybrid-scene claims require different data.
- After the feature contract is stable, how should the interface improve
  hierarchy, spacing, typography, progress, responsive behavior, and
  accessibility without hiding raw evidence?

## Brainstorming order

1. Define the benchmark matrix and paper claim.
2. Classify and consolidate methods and configurations.
3. Decide CIFAR-10 and any additional dataset roles.
4. Frame semantic disentanglement and localization hypotheses and ablations.
5. Design controlled multi-concept evaluation.
6. Refine public presets, interface explanations, and visual design.
