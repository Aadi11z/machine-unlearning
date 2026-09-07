# Sharanga oracle submission

`submit_canonical_oracle.slurm` runs the current development/integration oracle
workflow. It is not an accepted final-fit run. Other local wrappers remain
ignored and may use historical paths.

On the login node, use a checkout containing this wrapper and the reference
evaluation/manifest scripts. Prepare the locked environment, CIFAR-100 data,
CLIP cache, and baseline package before submitting. `env_activation.sh` expects
these under `/scratch/$USER/machine-unlearning/`; the baseline must include
`outputs/cifar100/baseline/manifest.json`, `checkpoints/base_init.pt`, and
`metrics/finetune_metrics.json`.

From the repository root:

```bash
export UNML_ROOT="$PWD"
export UNML_RUN_ID=oracle-dev-01  # Choose a new ID for every submission.
export UNML_ORACLE_REQUEST=flowers_superclass

# Slurm opens these files before the wrapper executes.
mkdir -p "/scratch/$USER/machine-unlearning/logs"
bash -n src/hpc/submit_canonical_oracle.slurm
sbatch --test-only src/hpc/submit_canonical_oracle.slurm
sbatch src/hpc/submit_canonical_oracle.slurm
```

The wrapper retains the existing A100 partition/QoS and Python Spack settings;
check their availability for your account before submission. It refuses direct
execution without a Slurm job ID, unsafe run/request names, and an existing
oracle destination. On the allocated node, its preflight prints resolved
repository, scratch, data, cache, environment, baseline, and oracle paths, then
verifies baseline artifacts and CUDA/BF16/cache readiness before training.
Inspect those lines in the job log before treating the cluster setup as verified.

Results go to
`/scratch/$USER/machine-unlearning/outputs/cifar100/oracle/<run>/<request>/`.
Local wrapper tests stub the cluster commands; they do not establish GPU or
scheduler availability.
