FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy environment and install requirements
COPY environment.yaml /app/environment.yaml
# We can install requirements using conda or pip. Since we want standard PyTorch+MONAI:
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    monai[all]==1.3.0 \
    hydra-core==1.3.2 \
    wandb \
    dvc \
    pytest \
    ruff \
    black \
    nibabel \
    click \
    pandas \
    openpyxl \
    scikit-learn \
    scipy

# Copy source code
COPY . /app

CMD ["python", "scripts/preprocess_all.py", "--help"]
