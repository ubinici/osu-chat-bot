FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY eval ./eval

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir .

COPY config.toml ./

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/artifacts /home/app/.cache/huggingface \
    && chown -R app:app /app/artifacts /home/app

USER app

EXPOSE 8000

CMD ["osu-bot", "serve", "--host", "0.0.0.0", "--port", "8000"]
