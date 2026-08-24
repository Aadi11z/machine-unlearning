"""Command-line entrypoint for the UN-ML web interface."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UN-ML web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--offline", action="store_true", help="Require locally cached CLIP assets"
    )
    parser.add_argument(
        "--hosted",
        action="store_true",
        help="Require Modal credentials instead of allowing local CPU unlearning",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    return parser.parse_args()


def _runner_mode(
    *, hosted: bool, modal_url: str | None, modal_secret: str | None
) -> str:
    """Choose remote execution explicitly; public hosts must never train on CPU."""
    if bool(modal_url) != bool(modal_secret):
        raise SystemExit(
            "UNML_MODAL_URL and UNML_JOB_SECRET must be configured together."
        )
    if hosted and not modal_url:
        raise SystemExit(
            "Hosted mode requires UNML_MODAL_URL and UNML_JOB_SECRET; "
            "refusing to run CPU unlearning publicly."
        )
    return "modal" if modal_url else "subprocess"


def main() -> None:
    import uvicorn

    from interface.catalog import ArtifactCatalog
    from interface.jobs import JobManager, SubprocessJobRunner
    from interface.webapp import ProbeService, create_app
    from unml.utils import transformers_offline

    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    modal_url = os.environ.get("UNML_MODAL_URL")
    modal_secret = os.environ.get("UNML_JOB_SECRET")
    runner_mode = _runner_mode(
        hosted=args.hosted, modal_url=modal_url, modal_secret=modal_secret
    )

    output_root = args.output_root or (REPO_ROOT / "outputs")
    catalog = ArtifactCatalog(
        output_root=output_root,
        baseline_checkpoint_path=args.baseline_checkpoint,
    )
    baseline = catalog.baseline_checkpoint()
    if baseline is None:
        raise SystemExit(
            "No fine-tuned baseline checkpoint found under "
            f"{output_root}. Pass --baseline-checkpoint explicitly."
        )

    def registry_loader():
        from interface.registry import ModelRegistry
        from unml.data import load_split_metadata

        _, _, class_names = load_split_metadata(
            str(_split_for_baseline(catalog, baseline)), "cifar100"
        )
        return ModelRegistry(
            baseline_checkpoint_path=str(baseline),
            baseline_name="Fine-tuned baseline",
            dataset_name="cifar100",
            class_names=class_names,
            prompt_template="a photo of a {}",
            device_name=args.device,
            local_files_only=args.offline or transformers_offline(),
        )

    data_dir = args.data_dir or (REPO_ROOT / "data")
    if runner_mode == "modal":
        from interface.jobs import ModalJobRunner

        runner = ModalJobRunner(
            endpoint_url=modal_url,
            secret=modal_secret,
            output_root=output_root,
        )
        print(f"[interface] dispatching new jobs to Modal worker: {modal_url}")
    else:
        runner = SubprocessJobRunner(
            repo_root=REPO_ROOT,
            baseline_checkpoint=baseline,
            data_dir=data_dir,
            output_root=output_root,
            device=args.device,
        )
    app = create_app(
        catalog=catalog,
        job_manager=JobManager(
            catalog,
            runner=runner,
            usage_tracker=_usage_tracker(output_root),
            output_root=output_root,
            ttl_days=float(os.environ.get("UNML_JOB_TTL_DAYS", "7")),
        ),
        probe_service=ProbeService(registry_loader),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _usage_tracker(output_root: Path):
    from interface.guards import UsageTracker

    return UsageTracker(
        output_root / "interface_usage.json",
        monthly_budget=int(os.environ.get("UNML_MAX_REMOTE_JOBS_PER_MONTH", "300")),
    )


def _split_for_baseline(catalog: ArtifactCatalog, baseline: Path) -> Path:
    request_dir = baseline.parent.parent.parent
    split_path = request_dir / "splits" / f"{request_dir.name}_split.json"
    if not split_path.is_file():
        raise SystemExit(
            f"Baseline split metadata not found at {split_path}; the interface "
            "needs it for the class vocabulary."
        )
    return split_path
