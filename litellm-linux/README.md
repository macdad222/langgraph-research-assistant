# LiteLLM Linux Template

This folder contains a sanitized LiteLLM + Postgres template for running a model gateway in front of a vLLM server.

1. Copy `.env.example` to `.env`.
2. Edit `config.example.yaml` for your vLLM host/model.
3. Copy `config.example.yaml` to `config.yaml`.
4. Start with `docker-compose up -d`.

Do not commit `.env`, `config.yaml` if it contains secrets, `ui-credentials.txt`, or Postgres data.
