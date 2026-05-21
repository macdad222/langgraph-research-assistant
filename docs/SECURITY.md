# Security Notes

This repository is a sanitized public template. It should not contain live secrets.

## Never Commit

- `.env` files
- LiteLLM master keys
- vLLM API keys
- Tavily API keys
- Langfuse secret keys
- Neo4j production passwords
- Generated credential files
- Database volumes
- Research run data containing private questions or source material

## Before Publishing Changes

Run checks like:

```bash
git status --short
rg -n "(sk-|tvly-|password|secret|api[_-]?key|master[_-]?key|BEGIN RSA|BEGIN OPENSSH)" .
```

Then inspect every match manually. Example placeholders such as `change-me` are expected.

## Public Deployment Warning

The included Compose files are intended as local-development templates. Do not expose these services to the internet without adding authentication, TLS, network policy, proper secret management, and backup procedures.
