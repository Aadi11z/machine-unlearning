from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from interface.catalog import ArtifactCatalog, CandidateArtifact
from interface.jobs import JobManager
from interface.webapp import (
    UploadTooLargeError,
    ProbeService,
    create_app,
    read_upload_limited,
)


CLASS_NAMES = ("rose", "tulip", "woman")


def _tiny_clip() -> CLIPModel:
    vision = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        projection_dim=16,
    )
    text = CLIPTextConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        projection_dim=16,
    )
    return CLIPModel(
        CLIPConfig(
            text_config=text.to_dict(),
            vision_config=vision.to_dict(),
            projection_dim=16,
        )
    )


class _FakeProcessor:
    def __call__(self, images, return_tensors=None):
        return {"pixel_values": torch.zeros(1, 3, 32, 32)}


@pytest.fixture
def client_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    baseline_path = tmp_path / "finetuned_best.pt"
    save_checkpoint(str(baseline_path), model)
    candidate_dir = tmp_path / "rose_selective" / "unlearn_ga_kl_200" / "checkpoints"
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "unlearn_ga_kl.pt"
    save_checkpoint(str(candidate_path), model)

    comparison_csv = tmp_path / "rose_selective" / "eval_compare_x" / "comparison.csv"
    comparison_csv.parent.mkdir(parents=True)
    comparison_csv.write_text(
        "model,target_test_acc,sibling_test_acc,unrelated_test_acc,utility_test_all\n"
        "ga_kl,0.05,0.80,0.85,0.84\n"
    )

    output_root = tmp_path / "outputs"
    output_root.mkdir()
    (output_root / "cifar100").symlink_to(tmp_path, target_is_directory=True)

    catalog = ArtifactCatalog(
        output_root=output_root,
        baseline_checkpoint_path=baseline_path,
    )
    text_features = torch.nn.functional.normalize(
        torch.randn(3, 16, generator=torch.Generator().manual_seed(0)), dim=-1
    )

    def fake_loader(**kwargs):
        assert kwargs["checkpoint_path"] == str(baseline_path)
        return SimpleNamespace(
            model=model,
            metadata={},
            class_names=CLASS_NAMES,
            image_processor=_FakeProcessor(),
            class_text_features=text_features,
            device=torch.device("cpu"),
        )

    def registry_loader():
        from interface.registry import ModelRegistry

        return ModelRegistry(
            baseline_checkpoint_path=str(baseline_path),
            baseline_name="Fine-tuned baseline",
            dataset_name="cifar100",
            class_names=list(CLASS_NAMES),
            prompt_template="a photo of a {}",
            device_name="cpu",
            predictor_loader=fake_loader,
        )

    app = create_app(
        catalog=catalog,
        job_manager=JobManager(catalog),
        probe_service=ProbeService(registry_loader),
    )
    png = _png_bytes()
    return SimpleNamespace(client=TestClient(app), png=png, catalog=catalog)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(90, 140, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_index_renders_groups_and_strip(client_env) -> None:
    response = client_env.client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "CIFAR-100" in body
    assert "flowers" in body
    assert "rose" in body
    assert "Forget it" in body


def test_precomputed_job_completes_and_probe_round_trips(client_env) -> None:
    client = client_env.client
    submit = client.post(
        "/api/unlearn",
        data={"class_ref": "rose", "method": "ga_kl", "steps": 200},
    )
    assert submit.status_code == 202
    job = submit.json()
    assert job["status"] == "done"
    assert job["source"] == "precomputed"
    assert job["candidate_id"] == "rose_selective_ga_kl_200"

    probe = client.post(
        "/api/probe",
        files={"image": ("probe.png", client_env.png, "image/png")},
        data={"candidate_id": job["candidate_id"], "retained_label": "woman"},
    )
    assert probe.status_code == 200
    payload = probe.json()
    assert payload["target_class_name"] == "rose"
    assert payload["verdict"]["verdict"]
    assert len(payload["baseline_top_k"]) >= 1
    assert len(payload["candidate_top_k"]) >= 1
    assert payload["metrics_table"][0]["Value"] == "0.0500"


def test_probe_html_fragment_flow(client_env) -> None:
    client = client_env.client
    submit = client.post(
        "/unlearn",
        data={"class_id": "70", "method": "ga_kl", "steps": "200"},
    )
    assert submit.status_code == 200
    assert "Done" in submit.text or "queued" in submit.text
    assert 'data-candidate-id="rose_selective_ga_kl_200"' in submit.text

    probe = client.post(
        "/probe",
        files={"image": ("probe.png", client_env.png, "image/png")},
        data={"candidate_id": "rose_selective_ga_kl_200"},
    )
    assert probe.status_code == 200
    assert "result-panel" in probe.text


def test_unlearn_html_reports_unavailable_execution(client_env) -> None:
    class UnavailableJobManager:
        def submit(self, **_kwargs):
            raise RuntimeError("No remote job executor is configured")

    app = create_app(
        catalog=client_env.catalog,
        job_manager=UnavailableJobManager(),
    )
    response = TestClient(app).post(
        "/unlearn",
        data={"class_id": "70", "method": "ga_kl", "steps": "200"},
    )

    assert response.status_code == 503
    assert "No remote job executor is configured" in response.text


def test_invalid_job_specs_are_rejected(client_env) -> None:
    client = client_env.client
    cases = [
        {"class_ref": "rose", "method": "made_up", "steps": 200},
        {"class_ref": "rose", "method": "ga_kl", "steps": 100000},
        {"class_ref": "not_a_class", "method": "ga_kl", "steps": 200},
    ]
    for case in cases:
        response = client.post("/api/unlearn", data=case)
        assert response.status_code == 400, case


def test_probe_without_candidate_returns_404(client_env) -> None:
    response = client_env.client.post(
        "/api/probe",
        files={"image": ("probe.png", client_env.png, "image/png")},
        data={"candidate_id": "never_ran"},
    )
    assert response.status_code == 404


def test_corrupt_image_upload_is_rejected(client_env) -> None:
    client_env.client.post(
        "/api/unlearn",
        data={"class_ref": "rose", "method": "ga_kl", "steps": 200},
    )
    response = client_env.client.post(
        "/api/probe",
        files={"image": ("junk.png", b"not-an-image", "image/png")},
        data={"candidate_id": "rose_selective_ga_kl_200"},
    )
    assert response.status_code == 400


def test_probe_routes_reject_oversized_request_bodies_before_parsing(client_env) -> None:
    app = create_app(
        catalog=client_env.catalog,
        job_manager=JobManager(client_env.catalog),
        max_probe_upload_bytes=1024,
    )
    client = TestClient(app)
    oversized = b"x" * 2048
    for path in ("/probe", "/api/probe"):
        response = client.post(
            path,
            files={"image": ("oversized.bin", oversized, "application/octet-stream")},
            data={"candidate_id": "rose_selective_ga_kl_200"},
        )
        assert response.status_code == 413


def test_bounded_upload_reader_stops_at_the_limit() -> None:
    from starlette.datastructures import UploadFile as StarletteUploadFile

    upload = StarletteUploadFile(
        filename="large.bin", file=io.BytesIO(b"x" * 4096)
    )
    with pytest.raises(UploadTooLargeError):
        asyncio.run(read_upload_limited(upload, max_bytes=1024))
    assert upload.file.tell() == 1025


def test_healthz_reports_model_state(client_env) -> None:
    response = client_env.client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "model_loaded": False}
