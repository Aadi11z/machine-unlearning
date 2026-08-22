# UN-ML platform web app on Hugging Face Spaces (free CPU tier).
# Build context must be the repository root:
#   docker build -f deploy/Dockerfile -t unml-interface .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1.0" "torchvision>=0.16.0"

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY helpers/cache_model.py ./helpers/cache_model.py
COPY config ./config

# Small trusted artifacts: class-agnostic baseline, split metadata,
# and the recorded comparison metrics used by the results panel.
COPY outputs/cifar100/rose_selective/baseline_2000/checkpoints/finetuned_best.pt \
     /app/outputs/cifar100/rose_selective/baseline_2000/checkpoints/finetuned_best.pt
COPY outputs/cifar100/rose_selective/splits/rose_selective_split.json \
     /app/outputs/cifar100/rose_selective/splits/rose_selective_split.json
COPY outputs/cifar100/rose_selective/eval_compare_h_tgsd_ablation_ga_kl_200/comparison.csv \
     /app/outputs/cifar100/rose_selective/eval_compare_h_tgsd_ablation_ga_kl_200/comparison.csv

RUN pip install --no-cache-dir \
    "transformers>=4.40.0,<5.0" "tqdm>=4.66.0" "PyYAML>=6.0" \
    "numpy>=1.26.0,<2.0" "pandas>=2.2.0" "tabulate>=0.9.0" \
    "scikit-learn>=1.4.0" "matplotlib>=3.8.0" "Pillow>=10.0" \
    "safetensors>=0.4.0" "fastapi>=0.115.0" "uvicorn>=0.30.0" \
    "jinja2>=3.1.0" "python-multipart>=0.0.9"

# The entrypoint deliberately runs offline, so bake the frozen CLIP assets
# into the image while network access is still available at build time.
RUN python helpers/cache_model.py --dataset cifar100

EXPOSE 7860
CMD ["python", "scripts/run_interface.py", "--host", "0.0.0.0", "--port", "7860", "--device", "cpu", "--offline", "--hosted"]
