FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /workspace

RUN python -m pip install --upgrade pip

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install -e ".[entities]"

CMD ["python", "-m", "osu_chatbot.app.cli", "--help"]
