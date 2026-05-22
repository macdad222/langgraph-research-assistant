# LiteLLM Linux Template

This folder contains a sanitized LiteLLM + Postgres template for running a model gateway in front of a vLLM server.

1. Copy `.env.example` to `.env`.
2. Edit `config.example.yaml` for your vLLM host/model.
3. Copy `config.example.yaml` to `config.yaml`.
4. Start with `docker compose up -d`.

The Linux template runs LiteLLM with `network_mode: host` and listens on host port `4010`.

LiteLLM Postgres stays containerized, but is published only on host loopback:

```text
127.0.0.1:5433 -> litellm-db:5432
```

This keeps the LiteLLM UI/API database persistent while letting upstream callers use the real host/LAN network path.

Do not commit `.env`, `config.yaml` if it contains secrets, `ui-credentials.txt`, or Postgres data.
