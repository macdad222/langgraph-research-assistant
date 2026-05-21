# Sanitization Notes

This public repo was created from a working local lab environment by staging only safe files.

## Included

- Agent source code
- Frontend source code
- Docker Compose templates
- `.env.example` templates
- Documentation
- Helper scripts

## Excluded

- Live `.env` files
- Generated credentials
- Research run JSON archive
- Redis, Neo4j, Postgres, ClickHouse, and MinIO volumes
- Docker credential/config folders
- LiteLLM UI credentials

## Template Rules

Use `.env.example` files as documentation. Copy them to `.env` locally and fill in real values outside git.
