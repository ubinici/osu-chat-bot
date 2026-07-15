# CPU Droplet deployment

This is a single-host MVP stack: one chatbot process, one Qdrant node, and one CPU-only Ollama process. It is meant for real-world testing, not high availability.

## Droplet

Start with at least 8 GB RAM. The documented `qwen3:4b` Ollama artifact is about 2.5 GB, and the embedding model, Qdrant, operating system, and indexing process need additional headroom. Prefer a dedicated-CPU plan if sustained generation latency matters; resize after measuring rather than guessing.

Use an Ubuntu CPU Droplet with SSH keys and DigitalOcean monitoring enabled. Apply a Cloud Firewall that permits SSH only. The Compose file binds the API to `127.0.0.1`, and it does not publish Qdrant or Ollama.

## First deployment

Install Docker Engine with its Compose plugin, then clone this repository on the Droplet.

```bash
cd osu-chat-bot
cp deploy/.env.example deploy/.env
chmod 600 deploy/.env
git clone --depth 1 https://github.com/ppy/osu-wiki.git database/osu-wiki
mkdir -p artifacts
sudo chown -R 1000:1000 artifacts
docker compose -f deploy/compose.yaml build app
docker compose -f deploy/compose.yaml up -d qdrant ollama
docker compose -f deploy/compose.yaml exec ollama ollama pull qwen3:4b
```

Build the new dense index remotely:

```bash
docker compose -f deploy/compose.yaml run --rm app osu-bot ingest
docker compose -f deploy/compose.yaml run --rm app osu-bot validate
docker compose -f deploy/compose.yaml run --rm app osu-bot index --batch-size 32
docker compose -f deploy/compose.yaml run --rm app osu-bot eval eval/osu_seed.jsonl
```

If indexing is interrupted, resume it:

```bash
docker compose -f deploy/compose.yaml run --rm app osu-bot index --resume --batch-size 32
```

Start the API and verify it:

```bash
docker compose -f deploy/compose.yaml up -d app
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does approach rate change?"}'
```

For private testing from your computer, use an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 your-user@your-droplet
```

Then call `http://127.0.0.1:8000` locally. Add an authenticated TLS reverse proxy before exposing the API publicly.

## Provider switch

To call Ollama Cloud directly, change `deploy/.env`:

```dotenv
OSU_BOT_GENERATION_PROVIDER=ollama
OSU_BOT_GENERATION_URL=https://ollama.com
OSU_BOT_GENERATION_MODEL=gpt-oss:20b
OSU_BOT_GENERATION_API_KEY=your-secret
OSU_BOT_GENERATION_THINK=low
```

The HTTPS scheme is required. GPT-OSS accepts `low`, `medium`, or `high` for
thinking; other Ollama models can normally use `false`.

To use a separate OpenAI-compatible endpoint instead, change `deploy/.env`:

```dotenv
OSU_BOT_GENERATION_PROVIDER=openai-compatible
OSU_BOT_GENERATION_URL=https://your-provider.example/v1
OSU_BOT_GENERATION_MODEL=your-model
OSU_BOT_GENERATION_API_KEY=your-secret
```

Restart only the app after changing generation settings:

```bash
docker compose -f deploy/compose.yaml up -d --force-recreate app
```

Third-party inference charges are separate from the eligible CPU infrastructure described in the DigitalOcean student-credit notice.

## Observe and leave cleanly

Record the Droplet plan, peak RAM, first-token/total latency, retrieval evaluation score, model name, and container image digests. Keep the API private until authentication and TLS exist.

Before the credits expire, export the Qdrant collection, copy evaluation reports and deployment settings off the Droplet, and review all remaining Droplets, volumes, and snapshots before deleting anything.

## References

- [DigitalOcean CPU Droplet plan guidance](https://docs.digitalocean.com/products/droplets/concepts/choosing-a-plan/)
- [DigitalOcean Cloud Firewalls](https://docs.digitalocean.com/products/networking/firewalls/how-to/create/)
- [DigitalOcean monitoring agent](https://docs.digitalocean.com/products/monitoring/how-to/install-metrics-agent/)
- [Ollama CPU-only Docker setup](https://docs.ollama.com/docker)
- [Ollama qwen3:4b model details](https://ollama.com/library/qwen3:4b)
- [Qdrant installation and storage guidance](https://qdrant.tech/documentation/installation/)
- [Qdrant network security guidance](https://qdrant.tech/documentation/security/)
