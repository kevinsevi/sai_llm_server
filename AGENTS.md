# Repository Guidelines

## Project Structure & Module Organization

All source code lives at the repository root. There are no subdirectories for source files.

| File | Responsibility |
|---|---|
| `sai_gateway.py` | FastAPI app; entry point for all HTTP endpoints |
| `sai_handler.py` | LiteLLM CustomLLM provider; makes the actual HTTP call to SAI |
| `sai_converter.py` | Translates OpenAI request/response format to/from SAI native format |
| `sai_extractor.py` | Recursive text extraction from mixed content structures |
| `sai_builder.py` | Builds SSE events for streaming responses (Responses API format) |
| `sai_chat_completions.py` | Handler logic for /v1/chat/completions |
| `sai_completions.py` | Handler logic for /v1/completions |
| `sai_responses.py` | Handler logic for /v1/responses |
| `sai_models.py` | Static model catalogue returned by /v1/models |
| `sai_system_prompt.py` | Loads per-endpoint system prompt files (system_prompt_*.txt) |
| `config.yaml` | LiteLLM configuration (model list, custom provider map) |
| `logs/` | Rotating log files written by sai_handler.py |

System prompt overrides go in `system_prompt_responses.txt` and
`system_prompt_chat_completions.txt` at the root.

## Build, Test, and Development Commands

```bash
# Start locally with Docker Compose (development)
docker-compose up -d

# Tail logs
docker-compose logs -f

# Verify the server is healthy
curl http://localhost:4000/health

# Rebuild image after code changes
docker-compose up -d --build

# Production deploy (Docker Swarm)
docker stack deploy -c docker-stack.yml sai_llm
```

Required environment variables (`.env` file or shell exports):

```
SAI_KEY=<system api key>
SAI_COOKIE=<system cookie>
SAI_TEMPLATE_ID=<template id>
SAI_URL=<sai base url>
VERBOSE_LOGGING=false
REQUEST_TIMEOUT=600
MAX_RETRIES=3
```

## Coding Style & Naming Conventions

- **Language:** Python 3.11+.
- **Indentation:** 4 spaces; no tabs.
- **Classes:** PascalCase (e.g., `OpenAiSAIConverter`, `ResponseEventBuilder`).
- **Functions/variables:** snake_case.
- **Module names:** `sai_<domain>.py` prefix for every new module (e.g., `sai_auth.py`).
- **Logging:** use the shared `logger` imported from `sai_handler`; prefix log
  messages with an emoji and the request_id for traceability
  (e.g., `logger.info(f"[{request_id}] message")`).
- No external formatter is configured; follow PEP 8 manually.

## Testing Guidelines

There is no automated test suite. Validation is done manually:

1. Start the server with `docker-compose up -d`.
2. Send a request to the relevant endpoint and inspect the response.
3. Set `VERBOSE_LOGGING=true` to see full payloads in `logs/sai_handler.log`.
4. Search logs by the 8-character `request_id` printed at the start of every request.

If you add a test suite, place tests in a `tests/` directory and use
`test_<module>.py` naming.

## Commit & Pull Request Guidelines

Commit messages follow **imperative, sentence-case Spanish** (matching the existing history):

```
Anade soporte para streaming en /v1/completions
Corrige extraccion de tool_calls con arguments no-JSON
Elimina modulo de excepciones personalizadas
```

- One logical change per commit.
- PR descriptions must include: what changed, why, and how to test it.
- Link any related issue in the PR body.
- Do not commit `.env` files, secrets, or credentials.

## Security & Configuration Notes

- Secrets (`SAI_KEY`, `SAI_COOKIE`) must be injected via environment variables
  or Docker Swarm secrets -- never hardcoded.
- In Swarm mode, use `docker secret create` and reference secrets in `docker-stack.yml`.
- The server supports dual authentication: per-user API keys forwarded in
  `Authorization` / `x-api-key` headers, with automatic fallback to system credentials.
