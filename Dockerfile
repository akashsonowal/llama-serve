FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    apt-utils \
    wget \
    curl \
    git \
    unzip \
    ca-certificates \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PATH="/app:${PATH}"

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "src/llama_serve/app.py"]