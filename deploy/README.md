# CPU Droplet deployment

This is a single-host closed-beta stack: one FastAPI RAG process, one lightweight Discord process, and one Qdrant node. Generation runs through GPT-OSS on Ollama Cloud. It is meant for real-world testing, not high availability.

## Droplet

Size the Droplet for the embedding model, Qdrant, and indexing workload; model generation no longer consumes local RAM because it runs on Ollama Cloud. Resize after measuring peak memory and request latency rather than guessing.

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
docker compose -f deploy/compose.yaml up -d qdrant
```

Before starting the API, set the Ollama Cloud key in the ignored `deploy/.env` file:

```dotenv
OSU_BOT_GENERATION_PROVIDER=ollama
OSU_BOT_GENERATION_URL=https://ollama.com
OSU_BOT_GENERATION_MODEL=gpt-oss:20b
OSU_BOT_GENERATION_API_KEY=your-secret
OSU_BOT_GENERATION_THINK=low
OSU_BOT_MAX_CONCURRENT_REQUESTS=4
```

Build the new dense index remotely:

```bash
docker compose -f deploy/compose.yaml run --rm app osu-bot ingest
docker compose -f deploy/compose.yaml run --rm app osu-bot links
docker compose -f deploy/compose.yaml run --rm app osu-bot aliases
docker compose -f deploy/compose.yaml run --rm app osu-bot validate
docker compose -f deploy/compose.yaml run --rm app osu-bot index --batch-size 32
docker compose -f deploy/compose.yaml run --rm app osu-bot eval eval/osu_seed.jsonl
```

`aliases` is CPU-light and does not change embeddings. After a code-only upgrade to
the query analyzer, run `links` and `aliases`, rebuild/recreate the app container,
and evaluate the existing collection before deciding whether to re-index.

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

## Discord closed beta

Create an application in the [Discord Developer Portal](https://discord.com/developers/applications),
add a bot user, and install it into one test server with the `bot` and
`applications.commands` scopes. Give it permission to view the test channel and send
messages. The integration only uses slash commands, buttons, and modals, so the privileged
Message Content intent is not required.

Turn on Developer Mode in Discord and copy the test server ID, then add these values to
the ignored `deploy/.env` file:

```dotenv
OSU_BOT_DISCORD_TOKEN=your-bot-token
OSU_BOT_DISCORD_GUILD_ID=your-test-server-id
OSU_BOT_DISCORD_EPHEMERAL=false
OSU_BOT_ANSWER_VERSION=gpt-oss-discord-v1
```

Using a guild ID makes `/ask` sync directly to the test server. If the guild ID is omitted,
the bot registers the command globally and Discord may take longer to propagate it.

Start both serving processes and inspect their logs:

```bash
docker compose -f deploy/compose.yaml up -d app discord
docker compose -f deploy/compose.yaml logs -f app discord
```

The Discord container calls `http://app:8000/v1/chat`; it does not load a second embedding
model. Answers include wiki links and feedback controls. Feedback is written to
`artifacts/feedback/events.jsonl` and contains the query, retrieval metadata, rating, and
optional correction, but no Discord user ID. Rotate the token immediately if it is ever
printed, pasted into chat, or committed.

## Response style profile

Build the style profile offline from a JSONL export with one message object per line and a
`content` field. Use chat data you are permitted to process and do not copy the raw export
onto the server:

```bash
osu-bot build-style-profile private/chat.jsonl \
  --output artifacts/style/osu_chat_style.json \
  --minimum-messages 100
```

The output stores aggregate ratios only—no quotes, usernames, IDs, or extracted n-grams.
Copy that small JSON artifact to the server, then set:

```dotenv
OSU_BOT_STYLE_PROFILE_PATH=/app/artifacts/style/osu_chat_style.json
```

Recreate the app container after changing the profile. The profile adjusts soft tone
tendencies while the grounding and citation rules remain authoritative.

## Hugging Face authentication

The public embedding model can be downloaded anonymously, but authenticated Hub
requests have higher rate limits. Create a dedicated fine-grained read token and
store it only in the ignored `deploy/.env` file:

```dotenv
HF_TOKEN=hf_your_token
```

Recreate the app container after adding or rotating the token. Compose also passes
it to one-off `app` commands used for indexing. Never commit the token.

## Provider configuration

The HTTPS scheme is required for Ollama Cloud. GPT-OSS accepts `low`, `medium`, or
`high` for thinking. `OSU_BOT_MAX_CONCURRENT_REQUESTS` bounds concurrent calls made
by the async FastAPI route; start at 4 and adjust from observed latency and provider limits.

To use a separate OpenAI-compatible endpoint instead, change `deploy/.env`:

```dotenv
OSU_BOT_GENERATION_PROVIDER=openai-compatible
OSU_BOT_GENERATION_URL=https://your-provider.example/v1
OSU_BOT_GENERATION_MODEL=your-model
OSU_BOT_GENERATION_API_KEY=your-secret
```

Restart the app after changing generation settings:

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
- [Discord interactions overview](https://docs.discord.com/developers/interactions/overview)
- [Discord application commands](https://docs.discord.com/developers/interactions/application-commands)
- [discord.py interactions API](https://discordpy.readthedocs.io/en/stable/interactions/api.html)
- [Qdrant installation and storage guidance](https://qdrant.tech/documentation/installation/)
- [Qdrant network security guidance](https://qdrant.tech/documentation/security/)
