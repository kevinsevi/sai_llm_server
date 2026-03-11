---
apply: always
---

# guidelines.md

# Resumen de Archivos del Proyecto OpenAI SAI Gateway

sai_extractor.py: Extrae texto recursivamente de estructuras complejas (str, list, dict) con prioridades para campos específicos.
sai_converter.py: Convierte bidireccionalmente entre formatos OpenAI y LiteLLM, procesando mensajes, system prompts, tools y metadata.
sai_handler.py: Implementa CustomLLM para comunicación con SAI, soporta completion síncrono/asíncrono/streaming, extrae tool_calls, gestiona autenticación y maneja errores HTTP.
sai_builder.py: Construye eventos SSE en formato OpenAI Responses API (response.created, response.in_progress, response.output_text.delta, response.output_item.done, response.completed, etc.) con soporte para function_calls y custom_tool_calls.
sai_completions.py: Handler para endpoints /v1/completions (legacy OpenAI format) con soporte streaming y no streaming.
sai_chat_completions.py: Handler para endpoints /v1/chat/completions con soporte streaming SSE, tool_calls y formato OpenAI Chat Completions API.
sai_responses.py: Handler para endpoints /v1/responses con eventos SSE canónicos, soporta function_calls, custom_tool_calls (apply_patch) y formato OpenAI Responses API.
sai_models.py: Define modelos disponibles (claude-sonnet-4-5-20250929, gpt-5.2-2025-12-11) y funciones para listar/buscar/validar modelos.
sai_system_prompt.py: Gestiona system prompts adicionales por endpoint (responses/chat_completions) desde archivos externos configurables vía variables de entorno, con caché inteligente basado en timestamps de modificación y helpers para concatenación automática.
sai_gateway.py: Aplicación FastAPI que expone endpoints compatibles con OpenAI (Completions, Chat Completions, Messages, Responses, Models) y coordina handlers.
sai_exceptions.py: Excepciones de dominio para SAI (SAIAPIError, SAIAuthenticationError, SAIRateLimitError, SAIPromptTooLongError, SAIServerError, SAIConnectionError, SAITimeoutError); usadas en sai_handler y capturadas en sai_gateway/handlers para mapear a HTTP.