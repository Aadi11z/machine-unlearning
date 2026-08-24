from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from interface.catalog import ArtifactCatalog, validate_job_spec
from interface.jobs import (
    JobManager,
    JobRecord,
    JobStatus,
    ModalJobRunner,
    SubprocessJobRunner,
)
from interface.remote import (
    SIGNATURE_HEADER,
    decode_checkpoint,
    encode_checkpoint,
    parse_job_response,
    sign_payload,
    verify_signature,
    write_job_artifacts,
)


def test_signature_round_trip() -> None:
    body = b'{"class_id": 70}'
    signature = sign_payload("s3cret", body)
    assert verify_signature("s3cret", body, signature.upper())
    assert not verify_signature("s3cret", body + b" ", signature)
    assert not verify_signature("wrong", body, signature)
    assert not verify_signature("s3cret", body, None)


def test_subprocess_runner_uses_the_configured_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "custom-artifacts"
    seen: dict[str, list[str]] = {}

    def fake_run(command, **_kwargs):
        seen["command"] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "job_result.json").write_text(
            json.dumps({"result": {"checkpoint": str(output_dir / "model.safetensors")}})
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("interface.jobs.subprocess.run", fake_run)
    runner = SubprocessJobRunner(
        repo_root=tmp_path / "repo",
        baseline_checkpoint=tmp_path / "baseline.safetensors",
        data_dir=tmp_path / "data",
        output_root=output_root,
        device="cpu",
    )
    job = JobRecord(
        job_id="job1",
        class_id=70,
        class_name="rose",
        request_name="rose_selective",
        superclass="flowers",
        sibling_classes=[54],
        method="ga_kl",
        steps=200,
        status=JobStatus.RUNNING,
    )

    runner(job)

    expected = output_root / "cifar100" / "jobs" / "rose_selective_ga_kl_200"
    assert Path(seen["command"][seen["command"].index("--output-dir") + 1]) == expected
    assert job.checkpoint_path == str(expected / "model.safetensors")


def test_parse_job_response_validates_fields() -> None:
    good = {
        "checkpoint_b64": encode_checkpoint(b"ckpt"),
        "class_id": 70,
        "class_name": "rose",
        "request_name": "rose_selective",
        "superclass": "flowers",
        "sibling_classes": [54, 62, 82, 92],
        "method": "ga_kl",
        "steps": 200,
    }
    parsed = parse_job_response(json.dumps(good).encode())
    assert decode_checkpoint(parsed["checkpoint_b64"]) == b"ckpt"

    bad = copy.deepcopy(good)
    del bad["method"]
    with pytest.raises(ValueError, match="missing fields"):
        parse_job_response(json.dumps(bad).encode())

    bad2 = copy.deepcopy(good)
    bad2["checkpoint_b64"] = ""
    with pytest.raises(ValueError, match="empty checkpoint"):
        parse_job_response(json.dumps(bad2).encode())


def test_modal_runner_allows_long_running_endpoint(tmp_path: Path) -> None:
    runner = ModalJobRunner(
        endpoint_url="https://worker.invalid",
        secret="test-secret",
        output_root=tmp_path,
    )

    assert runner.timeout_s == 4000
    assert runner.request_timeout_s == 30


def _tiny_clip():
    from transformers import CLIPVisionConfig, CLIPTextConfig, CLIPConfig

    vision = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, image_size=32, patch_size=16, projection_dim=16,
    )
    text = CLIPTextConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=16, projection_dim=16,
    )
    return CLIPModel(
        CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                   projection_dim=16)
    )


@pytest.fixture
def artifact_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        "unml.model.CLIPModel.from_pretrained",
        lambda *_args, **_kwargs: _tiny_clip(),
    )
    from unml.model import LightweightVLM, ModelConfig, save_checkpoint

    model = LightweightVLM(
        ModelConfig(
            model_name="tiny",
            adapter_type="vision_lora",
            lora_rank=4,
            lora_alpha=4.0,
            lora_layers="all",
            lora_targets=("q_proj", "v_proj"),
            train_logit_scale=False,
        ),
    )
    baseline_dir = tmp_path / "outputs" / "cifar100" / "rose_selective" / "baseline_2000"
    save_checkpoint(str(baseline_dir / "checkpoints" / "finetuned_best.pt"), model)

    from unml.model import export_checkpoint_safetensors

    adapter_state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    adapter_state["logit_scale"] = model.logit_scale.detach().clone()
    checkpoint_bytes = export_checkpoint_safetensors(
        {
            "model_config": dict(model.cfg.__dict__),
            "adapter_state_dict": adapter_state,
        }
    )
    return SimpleNamespace(
        output_root=tmp_path / "outputs",
        checkpoint_bytes=checkpoint_bytes,
    )


def test_write_job_artifacts_lands_in_catalog_layout(artifact_env) -> None:
    env = artifact_env
    path = write_job_artifacts(
        output_root=env.output_root,
        class_id=70,
        class_name="rose",
        request_name="rose_selective",
        superclass="flowers",
        sibling_classes=[54, 62, 82, 92],
        method="h_tgsd",
        steps=150,
        checkpoint_bytes=env.checkpoint_bytes,
        metrics={"forget_acc": 0.01},
        wall_time_s=12.5,
    )
    assert path.is_file()
    catalog = ArtifactCatalog(output_root=env.output_root)
    candidate = catalog.precomputed_candidate(class_id=70, method="h_tgsd", steps=150)
    assert candidate is not None
    assert candidate.source == "job"
    assert candidate.comparison_model == "rose_selective_h_tgsd_150"

    with pytest.raises(Exception):
        write_job_artifacts(
            output_root=env.output_root,
            class_id=70,
            class_name="rose",
            request_name="rose_selective",
            superclass="flowers",
            sibling_classes=[54, 62, 82, 92],
            method="h_tgsd",
            steps=151,
            checkpoint_bytes=b"garbage-not-a-checkpoint",
            metrics={},
            wall_time_s=1.0,
        )


def test_write_job_artifacts_validates_without_constructing_clip(
    artifact_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = artifact_env

    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("remote validation must not construct LightweightVLM")

    monkeypatch.setattr(
        "unml.model.LightweightVLM.from_config", fail_if_constructed
    )

    path = write_job_artifacts(
        output_root=env.output_root,
        class_id=70,
        class_name="rose",
        request_name="rose_selective",
        superclass="flowers",
        sibling_classes=[54, 62, 82, 92],
        method="h_tgsd",
        steps=152,
        checkpoint_bytes=env.checkpoint_bytes,
        metrics={},
        wall_time_s=1.0,
    )

    assert path.is_file()


def test_write_job_artifacts_rejects_invalid_adapter_before_persisting(artifact_env) -> None:
    env = artifact_env
    from safetensors.torch import load as st_load
    from unml.model import export_checkpoint_safetensors, read_safetensors_metadata

    tensors = st_load(env.checkpoint_bytes)
    bad_key = next(key for key in tensors if key != "logit_scale")
    tensors[bad_key] = tensors[bad_key][:1]
    metadata = read_safetensors_metadata(env.checkpoint_bytes)
    checkpoint_bytes = export_checkpoint_safetensors(
        {
            "model_config": json.loads(metadata["model_config"]),
            "adapter_state_dict": tensors,
        }
    )

    with pytest.raises(ValueError, match="incompatible tensors"):
        write_job_artifacts(
            output_root=env.output_root,
            class_id=70,
            class_name="rose",
            request_name="rose_selective",
            superclass="flowers",
            sibling_classes=[54, 62, 82, 92],
            method="h_tgsd",
            steps=151,
            checkpoint_bytes=checkpoint_bytes,
            metrics={},
            wall_time_s=1.0,
        )
    assert not (
        env.output_root
        / "cifar100"
        / "jobs"
        / "rose_selective_h_tgsd_151"
    ).exists()


def test_modal_runner_round_trip_against_stub(artifact_env, tmp_path) -> None:
    env = artifact_env
    secret = "test-secret"
    spec_body = json.dumps({"class_id": 70, "method": "ga_kl", "steps": 120}).encode()

    def make_handler(response_payload: bytes, seen: dict):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                seen["signature"] = self.headers.get(SIGNATURE_HEADER)
                seen["body"] = body
                assert verify_signature(secret, body, seen["signature"])
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response_payload)

            def log_message(self, *args):  # silence
                pass

        return Handler

    response_payload = json.dumps(
        {
            "checkpoint_b64": encode_checkpoint(env.checkpoint_bytes),
            "class_id": 70,
            "class_name": "rose",
            "request_name": "rose_selective",
            "superclass": "flowers",
            "sibling_classes": [54, 62, 82, 92],
            "method": "ga_kl",
            "steps": 120,
            "wall_time_s": 42.0,
            "metrics": {"forget_acc": 0.0},
        }
    ).encode()

    seen: dict = {}
    server = HTTPServer(("127.0.0.1", 0), make_handler(response_payload, seen))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        runner = ModalJobRunner(
            endpoint_url=f"http://127.0.0.1:{server.server_port}/",
            secret=secret,
            output_root=env.output_root,
        )
        job = JobRecord(
            job_id="t1",
            class_id=70,
            class_name="rose",
            request_name="rose_selective",
            superclass="flowers",
            sibling_classes=[54, 62, 82, 92],
            method="ga_kl",
            steps=120,
            status=JobStatus.RUNNING,
        )
        runner(job)
    finally:
        server.shutdown()

    assert job.status is JobStatus.RUNNING  # manager sets DONE, not the runner
    assert job.source == "modal"
    assert job.candidate_id == "rose_selective_ga_kl_120"
    assert Path(job.checkpoint_path).is_file()
    assert seen["body"] == spec_body


def test_modal_runner_polls_detached_function_call(artifact_env) -> None:
    env = artifact_env
    secret = "test-secret"
    responses = [
        (202, {"status": "accepted", "call_id": "fc-test"}),
        (202, {"status": "running", "call_id": "fc-test"}),
        (
            200,
            {
                "checkpoint_b64": encode_checkpoint(env.checkpoint_bytes),
                "class_id": 70,
                "class_name": "rose",
                "request_name": "rose_selective",
                "superclass": "flowers",
                "sibling_classes": [54, 62, 82, 92],
                "method": "ga_kl",
                "steps": 120,
                "wall_time_s": 42.0,
                "metrics": {"forget_acc": 0.0},
            },
        ),
    ]
    seen_bodies: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            assert verify_signature(secret, body, self.headers.get(SIGNATURE_HEADER))
            seen_bodies.append(json.loads(body))
            status, payload = responses.pop(0)
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        runner = ModalJobRunner(
            endpoint_url=f"http://127.0.0.1:{server.server_port}/",
            secret=secret,
            output_root=env.output_root,
            poll_interval_s=0,
        )
        job = JobRecord(
            job_id="t1",
            class_id=70,
            class_name="rose",
            request_name="rose_selective",
            superclass="flowers",
            sibling_classes=[54, 62, 82, 92],
            method="ga_kl",
            steps=120,
            status=JobStatus.RUNNING,
        )
        runner(job)
    finally:
        server.shutdown()

    assert seen_bodies == [
        {"class_id": 70, "method": "ga_kl", "steps": 120},
        {"call_id": "fc-test"},
        {"call_id": "fc-test"},
    ]
    assert job.source == "modal"
    assert Path(job.checkpoint_path).is_file()


def test_modal_runner_rejects_mismatched_response_before_persisting(artifact_env) -> None:
    env = artifact_env
    secret = "test-secret"
    response_payload = json.dumps(
        {
            "checkpoint_b64": encode_checkpoint(env.checkpoint_bytes),
            "class_id": 69,
            "class_name": "rocket",
            "request_name": "rocket_selective",
            "superclass": "vehicles_2",
            "sibling_classes": [47, 55, 72, 95],
            "method": "ga_kl",
            "steps": 120,
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            assert verify_signature(secret, body, self.headers.get(SIGNATURE_HEADER))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response_payload)

        def log_message(self, *args):  # silence
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        runner = ModalJobRunner(
            endpoint_url=f"http://127.0.0.1:{server.server_port}/",
            secret=secret,
            output_root=env.output_root,
        )
        job = JobRecord(
            job_id="t1",
            class_id=70,
            class_name="rose",
            request_name="rose_selective",
            superclass="flowers",
            sibling_classes=[54, 62, 82, 92],
            method="ga_kl",
            steps=120,
            status=JobStatus.RUNNING,
        )
        with pytest.raises(RuntimeError, match="class_id"):
            runner(job)
    finally:
        server.shutdown()

    assert not (env.output_root / "cifar100" / "jobs").exists()


def test_catalog_translates_fresh_job_metrics_without_inventing_breakdowns(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    job_dir = output_root / "cifar100" / "jobs" / "rose_selective_ga_kl_120"
    checkpoint = job_dir / "checkpoints" / "unlearn_ga_kl.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    (job_dir / "job_result.json").write_text(
        json.dumps(
            {
                "request_name": "rose_selective",
                "result": {
                    "metrics": {
                        "forget_acc": 0.91,
                        "target_test_acc": 0.02,
                        "test_retain_acc": 0.81,
                        "test_all_acc": 0.80,
                    }
                },
            }
        )
    )

    catalog = ArtifactCatalog(output_root=output_root)
    candidate = catalog.precomputed_candidate(class_id=70, method="ga_kl", steps=120)
    assert candidate is not None
    assert candidate.source == "job"
    assert candidate.comparison_model == "rose_selective_ga_kl_120"
    metric_row = next(
        row
        for row in catalog.comparison_rows("rose_selective")
        if row["model"] == candidate.comparison_model
    )
    assert metric_row == {
        "model": "rose_selective_ga_kl_120",
        "target_test_acc": "0.02",
        "retained_test_acc": "0.81",
        "utility_test_all": "0.8",
    }
    assert "sibling_test_acc" not in metric_row
    assert "unrelated_test_acc" not in metric_row


def test_catalog_discovers_job_metrics_added_after_an_initial_lookup(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    catalog = ArtifactCatalog(output_root=output_root)
    assert catalog.comparison_rows("rose_selective") == []

    job_dir = output_root / "cifar100" / "jobs" / "rose_selective_ga_kl_120"
    job_dir.mkdir(parents=True)
    (job_dir / "job_result.json").write_text(
        json.dumps(
            {
                "candidate_id": "rose_selective_ga_kl_120",
                "request_name": "rose_selective",
                "result": {"metrics": {"target_test_acc": 0.02}},
            }
        )
    )

    assert catalog.comparison_rows("rose_selective") == [
        {"model": "rose_selective_ga_kl_120", "target_test_acc": "0.02"}
    ]


def test_catalog_reads_subprocess_job_result_metrics(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    job_dir = output_root / "cifar100" / "jobs" / "rose_selective_ga_kl_120"
    job_dir.mkdir(parents=True)
    (job_dir / "job_result.json").write_text(
        json.dumps(
            {
                "request_name": "rose_selective",
                "result": {
                    "checkpoint": "/ignored/checkpoint.pt",
                    "forget_acc": 0.91,
                    "target_test_acc": 0.02,
                    "test_retain_acc": 0.81,
                    "test_all_acc": 0.80,
                },
            }
        )
    )

    assert ArtifactCatalog(output_root=output_root).comparison_rows("rose_selective") == [
        {
            "model": "rose_selective_ga_kl_120",
            "target_test_acc": "0.02",
            "retained_test_acc": "0.81",
            "utility_test_all": "0.8",
        }
    ]


def test_validate_job_spec_bounds() -> None:
    assert validate_job_spec(class_id=99, method="h_tgsd", steps=500) == (99, "h_tgsd", 500)
    with pytest.raises(ValueError):
        validate_job_spec(class_id=100, method="ga_kl", steps=200)
    with pytest.raises(ValueError):
        validate_job_spec(class_id=70, method="ga_kl", steps=5)
