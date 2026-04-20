FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    python3 \
    python3-matplotlib \
    python3-pip \
    python3-venv \
    python3-yaml \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash autoresearch

WORKDIR /workspace

COPY docker/entrypoint.sh /usr/local/bin/autoresearch-entrypoint
RUN chmod +x /usr/local/bin/autoresearch-entrypoint

COPY --chown=autoresearch:autoresearch . /workspace

USER autoresearch

ENTRYPOINT ["autoresearch-entrypoint"]
CMD []
