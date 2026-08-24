# UN-ML Documentation

## Sources of truth

- [`PLAN.md`](PLAN.md): architecture decisions, delivery order, phase gates,
  and the route to the canonical CIFAR-100 baseline.
- [`PRD.md`](PRD.md): required product, research, operational, and acceptance
  behavior. A requirement marked **Required** is not an implementation fact.
- [`RUNBOOK.md`](RUNBOOK.md): commands that are executable in the current
  checkout, operational limitations, and troubleshooting.
- [`IDEAS.md`](IDEAS.md): unresolved hypotheses and product questions only.
- [`flowchart.md`](flowchart.md): current interface/Modal flow and the planned
  canonical-baseline identity boundary.

When these files appear to conflict, current source and tests establish what
exists, the PRD establishes what must exist, and the plan establishes when it
will be built.

## Current boundary

The FastAPI interface, Modal job path, request-specific CIFAR-10/CIFAR-100
experiment pipeline, unlearning methods, and evaluation tools exist. The
target-neutral canonical CIFAR-100 baseline pipeline described in the plan does
not yet exist. In particular, the active configuration still uses a
request-specific split, one prompt template, and a trainable replacement
temperature. Do not use that path to create the permanent baseline.

## Other documentation

- [`../README.md`](../README.md): project overview and quick-start commands.
- [`../deploy/README.md`](../deploy/README.md): Modal and Hugging Face deployment
  status and setup.

The local `research/` workspace contains historical notes, reports, and
publication material, but it is intentionally ignored and is not a maintained
dependency of these committed docs. Where local research notes describe
request-specific baseline training, the newer `docs/PLAN.md` supersedes them.
