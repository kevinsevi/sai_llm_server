"""
Gateway para exponer SAI con API compatible con OpenAI.

Este servidor proporciona:
- /v1/messages (sin streaming) - OpenaAI Messages API format
- /v1/responses (con streaming SSE) - OpenAI Responses API format
- /v1/chat/completions (OpenAI compatible) - Chat Completions API format

FIX v1.0.4: El verdadero problema era el nombre del evento final. OpenAI
            Responses API usa 'response.done', no 'response.completed'.
            Codex CLI seguía la especificación correctamente.
"""

import json
import time
import uuid
from typing import AsyncIterator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import logging

# Importar tu clase SAILLM existente
from sai_handler import sai_llm, logger, VERBOSE_LOGGING
from sai_models import get_models_list, get_model_by_id

app = FastAPI(title="OpenAI SAI Gateway", version="1.0.4")

def _normalize_bool(value) -> bool:
    """
    Normaliza un valor a booleano, manejando strings "true"/"false".
    
    Args:
        value: Puede ser bool, str, int, None
        
    Returns:
        bool: Valor normalizado
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if isinstance(value, int):
        return value != 0
    return False

# Logger específico del gateway

class OpenAiSAIConverter:
    """Convierte entre formatos de API de OpenAI y SAI."""

    @staticmethod
    def _extract_text_recursive(content) -> str:
        """
        Extrae texto de estructuras anidadas recursivamente.

        Maneja:
        - Strings simples: "hola"
        - Listas de content blocks: [{"type": "text", "text": "hola"}]
        - Listas anidadas complejas (Codex CLI): [{"type": "message", "content": [...]}]

        Args:
            content: Puede ser str, list, dict o cualquier combinación anidada

        Returns:
            str: Texto extraído y concatenado
        """
        # Caso 1: Ya es un string
        if isinstance(content, str):
            return content

        # Caso 2: Es una lista
        if isinstance(content, list):
            texts = []
            for item in content:
                # Recursión para cada elemento de la lista
                extracted = OpenAiSAIConverter._extract_text_recursive(item)
                if extracted:
                    texts.append(extracted)
            return " ".join(texts)

        # Caso 3: Es un diccionario
        if isinstance(content, dict):
            # Prioridad 1: Si tiene "text", usarlo directamente
            if "text" in content:
                return str(content["text"])

            # Prioridad 2: Si tiene "content", recursión
            if "content" in content:
                return OpenAiSAIConverter._extract_text_recursive(content["content"])

            # Prioridad 3: Si tiene "input_text" (formato Codex CLI)
            if "input_text" in content:
                return str(content["input_text"])

            # Si no tiene ninguno de los campos esperados, retornar vacío
            return ""

        # Caso 4: Otro tipo (None, int, etc.)
        return ""

    @staticmethod
    def openai_to_litellm(openai_request: dict) -> tuple[list, dict]:
        """
        Convierte request de OpenAI Messages API a formato LiteLLM.

        Args:
            openai_request: Request en formato OpenAI

        Returns:
            tuple: (messages, kwargs_for_litellm)
        """
        messages = []

        # Extraer system prompt si existe
        system_prompt = openai_request.get("system", "")
        if system_prompt:
            system_text = OpenAiSAIConverter._extract_text_recursive(system_prompt)
            if system_text:
                messages.append({
                    "role": "system",
                    "content": system_text
                })

        # Convertir mensajes
        for msg in openai_request.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")

            # Usar extracción recursiva para manejar cualquier nivel de anidación
            text_content = OpenAiSAIConverter._extract_text_recursive(content)

            # Solo agregar si hay contenido real
            if text_content:
                messages.append({
                    "role": role,
                    "content": text_content
                })

        # Preparar kwargs para LiteLLM
        kwargs = {
            "model": openai_request.get("model", "claude-sonnet-4-5-20250929"),
            "temperature": openai_request.get("temperature"),
            "max_tokens": openai_request.get("max_tokens", 4096),
            "top_p": openai_request.get("top_p"),
            "top_k": openai_request.get("top_k"),
            "stop_sequences": openai_request.get("stop_sequences"),
        }

        # Filtrar None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Extraer metadata adicional si existe
        metadata = openai_request.get("metadata", {})
        if metadata:
            kwargs["litellm_params"] = {
                "metadata": metadata
            }

        return messages, kwargs

    @staticmethod
    def litellm_to_openai_response(litellm_response, model: str, request_id: str) -> dict:
        """
        Convierte respuesta de LiteLLM a formato OpenAI Messages.

        Args:
            litellm_response: Respuesta de sai_llm.acompletion()
            model: Modelo usado
            request_id: ID de la solicitud

        Returns:
            dict: Respuesta en formato OpenAI
        """
        # Extraer texto de la respuesta
        text = ""
        if hasattr(litellm_response, 'text') and litellm_response.text:
            text = litellm_response.text
        elif hasattr(litellm_response, 'choices') and litellm_response.choices:
            choice = litellm_response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                text = choice.message.content or ""

        # Extraer usage
        usage_dict = {}
        if hasattr(litellm_response, 'usage'):
            usage_obj = litellm_response.usage
            if isinstance(usage_obj, dict):
                usage_dict = usage_obj
            else:
                usage_dict = usage_obj.__dict__ if hasattr(usage_obj, '__dict__') else {}

        # Extraer finish_reason
        finish_reason = "end_turn"
        if hasattr(litellm_response, 'choices') and litellm_response.choices:
            choice = litellm_response.choices[0]
            if hasattr(choice, 'finish_reason'):
                litellm_finish = choice.finish_reason
                # Mapear finish_reason de LiteLLM a OpenAI
                finish_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "error": "error"
                }
                finish_reason = finish_reason_map.get(litellm_finish, "end_turn")

        return {
            "id": f"msg_{request_id}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ],
            "model": model,
            "stop_reason": finish_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage_dict.get("prompt_tokens", 0),
                "output_tokens": usage_dict.get("completion_tokens", 0)
            }
        }


@app.post("/v1/completions")
@app.post("/completions")
async def completions_endpoint(request: Request):
    """
    Endpoint compatible con OpenAI Completions API (legacy).
    Reutiliza la lógica de sai_handler.py para mantener consistencia.

    Este endpoint soporta:
    - Formato de prompt simple (string)
    - Streaming y no streaming
    - Autenticación vía headers
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        body = await request.json()
        logger.info(f"📨 [{request_id}] /v1/completions request recibido")
        logger.debug(f"📋 [{request_id}] Body completo: {json.dumps(body, indent=2)}")

        logger.debug(f"🔑 [{request_id}] Headers recibidos:")
        logger.debug(f"   - Authorization: {request.headers.get('authorization', 'NO PRESENTE')}")
        logger.debug(f"   - x-api-key: {request.headers.get('x-api-key', 'NO PRESENTE')}")

        # Extraer parámetros
        prompt = body.get("prompt", "")
        stream = _normalize_bool(body.get("stream", False))
        model = body.get("model", "claude-sonnet-4-5-20250929")
        max_tokens = body.get("max_tokens", 4096)
        temperature = body.get("temperature")
        top_p = body.get("top_p")
        stop = body.get("stop")

        logger.info(
            f"📊 [{request_id}] Model: {model} | "
            f"Prompt length: {len(prompt)} chars | "
            f"Stream: {stream}"
        )

        # Validar prompt
        if not prompt:
            logger.error(f"❌ [{request_id}] Request sin 'prompt'")
            raise HTTPException(
                status_code=400,
                detail="Se requiere 'prompt'"
            )

        # Convertir prompt a formato de mensajes para sai_handler
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Preparar kwargs para sai_llm
        kwargs = {
            "model": model,
            "max_tokens": max_tokens
        }

        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop is not None:
            kwargs["stop"] = stop

        # Extraer user_api_key si viene en headers
        user_api_key = request.headers.get("authorization", "").replace("Bearer ", "").strip()
        if not user_api_key:
            user_api_key = request.headers.get("x-api-key")

        if user_api_key:
            kwargs["headers"] = {"user_api_key": user_api_key}
            logger.info(f"🔑 [{request_id}] user_api_key detectada")

        # Decidir si es streaming o no
        if stream:
            logger.info(f"🌊 [{request_id}] Modo streaming activado")
            return await _completions_streaming(request_id, messages, kwargs, model)
        else:
            logger.info(f"📄 [{request_id}] Modo sin streaming")
            return await _completions_non_streaming(request_id, messages, kwargs, model)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Error en /v1/completions: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _completions_non_streaming(request_id: str, messages: list, kwargs: dict, model: str):
    """Maneja completions sin streaming (reutiliza sai_handler.py)."""
    logger.info(f"🚀 [{request_id}] Llamando a sai_llm.acompletion()...")
    litellm_response = await sai_llm.acompletion(messages=messages, **kwargs)

    # Extraer texto
    text = ""
    if hasattr(litellm_response, 'text') and litellm_response.text:
        text = litellm_response.text
    elif hasattr(litellm_response, 'choices') and litellm_response.choices:
        choice = litellm_response.choices[0]
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            text = choice.message.content or ""

    # Extraer usage
    usage_dict = {}
    if hasattr(litellm_response, 'usage'):
        usage_obj = litellm_response.usage
        if isinstance(usage_obj, dict):
            usage_dict = usage_obj
        else:
            usage_dict = usage_obj.__dict__ if hasattr(usage_obj, '__dict__') else {}

    # Extraer finish_reason
    finish_reason = "stop"
    if hasattr(litellm_response, 'choices') and litellm_response.choices:
        choice = litellm_response.choices[0]
        if hasattr(choice, 'finish_reason'):
            finish_reason = choice.finish_reason or "stop"

    # Construir respuesta en formato OpenAI Completions
    openai_response = {
        "id": f"cmpl-{request_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": usage_dict.get("prompt_tokens", 0),
            "completion_tokens": usage_dict.get("completion_tokens", 0),
            "total_tokens": usage_dict.get("total_tokens", 0)
        }
    }

    logger.info(
        f"✅ [{request_id}] Respuesta lista | "
        f"Output tokens: {usage_dict.get('completion_tokens', 0)}"
    )

    return JSONResponse(content=openai_response)


async def _completions_streaming(request_id: str, messages: list, kwargs: dict, model: str):
    """Maneja completions con streaming (reutiliza sai_handler.py)."""
    async def event_generator():
        try:
            logger.info(f"🌊 [{request_id}] Iniciando streaming...")

            # Llamar a SAI streaming (reutiliza sai_handler.py)
            logger.info(f"🚀 [{request_id}] Llamando a sai_llm.astreaming()...")

            chunk_count = 0

            async for chunk in sai_llm.astreaming(messages=messages, **kwargs):
                chunk_count += 1

                if chunk_count == 1:
                    logger.warning(f"⏱️ [{request_id}] PRIMER CHUNK recibido")

                # Extraer texto del chunk
                chunk_text = chunk.get('text') if isinstance(chunk, dict) else getattr(chunk, 'text', None)

                if chunk_text:
                    # Chunk de contenido en formato Completions
                    content_chunk = {
                        "id": f"cmpl-{request_id}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "text": chunk_text,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"

                # Verificar si es el último chunk
                is_finished = chunk.get('is_finished') if isinstance(chunk, dict) else getattr(chunk, 'is_finished', False)
                if is_finished:
                    finish_reason = chunk.get('finish_reason') if isinstance(chunk, dict) else getattr(chunk, 'finish_reason', 'stop')

                    # Chunk final
                    final_chunk = {
                        "id": f"cmpl-{request_id}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "text": "",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": finish_reason or "stop"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"

            # Enviar [DONE]
            yield "data: [DONE]\n\n"

            logger.info(f"✅ [{request_id}] Streaming completado | Chunks: {chunk_count}")

        except Exception as e:
            logger.error(f"❌ [{request_id}] Error en streaming: {type(e).__name__}: {str(e)}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked"
        }
    )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions_endpoint(request: Request):
    """
    Endpoint compatible con OpenAI Chat Completions API.
    Reutiliza la lógica de messages_endpoint y responses_endpoint.
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        body = await request.json()
        logger.info(f"📨 [{request_id}] /v1/chat/completions request recibido")
        logger.debug(f"📋 [{request_id}] Body completo: {json.dumps(body, indent=2)}")

        logger.debug(f"🔑 [{request_id}] Headers recibidos:")
        logger.debug(f"   - Authorization: {request.headers.get('authorization', 'NO PRESENTE')}")
        logger.debug(f"   - x-api-key: {request.headers.get('x-api-key', 'NO PRESENTE')}")

        # Extraer parámetros de OpenAI format
        stream = _normalize_bool(body.get("stream", False))
        model = body.get("model", "claude-sonnet-4-5-20250929")
        messages_input = body.get("messages", [])

        logger.info(
            f"📊 [{request_id}] Model: {model} | "
            f"Messages: {len(messages_input)} | "
            f"Stream: {stream}"
        )

        # Validar messages
        if not messages_input:
            logger.error(f"❌ [{request_id}] Request sin 'messages'")
            raise HTTPException(
                status_code=400,
                detail="Se requiere 'messages'"
            )

        # Convertir a formato interno
        openai_body = {
            "model": model,
            "messages": messages_input,
            "max_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "stop": body.get("stop")
        }

        messages, kwargs = OpenAiSAIConverter.openai_to_litellm(openai_body)

        # Extraer user_api_key si viene en headers
        user_api_key = request.headers.get("authorization", "").replace("Bearer ", "").strip()
        if not user_api_key:
            user_api_key = request.headers.get("x-api-key")

        if user_api_key:
            kwargs["headers"] = {"user_api_key": user_api_key}
            logger.info(f"🔑 [{request_id}] user_api_key detectada")

        # Decidir si es streaming o no
        if stream:
            logger.info(f"🌊 [{request_id}] Modo streaming activado")
            return await _chat_completions_streaming(request_id, messages, kwargs, model)
        else:
            logger.info(f"📄 [{request_id}] Modo sin streaming")
            return await _chat_completions_non_streaming(request_id, messages, kwargs, model)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Error en /v1/chat/completions: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _chat_completions_non_streaming(request_id: str, messages: list, kwargs: dict, model: str):
    """Maneja chat completions sin streaming."""
    logger.info(f"🚀 [{request_id}] Llamando a sai_llm.acompletion()...")
    litellm_response = await sai_llm.acompletion(messages=messages, **kwargs)

    # Extraer texto
    text = ""
    if hasattr(litellm_response, 'text') and litellm_response.text:
        text = litellm_response.text
    elif hasattr(litellm_response, 'choices') and litellm_response.choices:
        choice = litellm_response.choices[0]
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            text = choice.message.content or ""

    # Extraer usage
    usage_dict = {}
    if hasattr(litellm_response, 'usage'):
        usage_obj = litellm_response.usage
        if isinstance(usage_obj, dict):
            usage_dict = usage_obj
        else:
            usage_dict = usage_obj.__dict__ if hasattr(usage_obj, '__dict__') else {}

    # Extraer finish_reason
    finish_reason = "stop"
    if hasattr(litellm_response, 'choices') and litellm_response.choices:
        choice = litellm_response.choices[0]
        if hasattr(choice, 'finish_reason'):
            finish_reason = choice.finish_reason or "stop"

    # Construir respuesta en formato OpenAI
    openai_response = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": usage_dict.get("prompt_tokens", 0),
            "completion_tokens": usage_dict.get("completion_tokens", 0),
            "total_tokens": usage_dict.get("total_tokens", 0)
        }
    }

    logger.info(
        f"✅ [{request_id}] Respuesta lista | "
        f"Output tokens: {usage_dict.get('completion_tokens', 0)}"
    )

    return JSONResponse(content=openai_response)


async def _chat_completions_streaming(request_id: str, messages: list, kwargs: dict, model: str):
    """Maneja chat completions con streaming."""
    async def event_generator():
        try:
            logger.info(f"🌊 [{request_id}] Iniciando streaming...")

            # Chunk inicial
            initial_chunk = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            # Llamar a SAI streaming
            logger.info(f"🚀 [{request_id}] Llamando a sai_llm.astreaming()...")

            chunk_count = 0
            total_tokens = 0
            input_tokens = 0
            output_tokens = 0

            async for chunk in sai_llm.astreaming(messages=messages, **kwargs):
                chunk_count += 1

                if chunk_count == 1:
                    logger.warning(f"⏱️ [{request_id}] PRIMER CHUNK recibido")

                # Extraer texto del chunk
                chunk_text = chunk.get('text') if isinstance(chunk, dict) else getattr(chunk, 'text', None)

                if chunk_text:
                    # Chunk de contenido
                    content_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_text},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"

                # Extraer usage
                usage = chunk.get('usage') if isinstance(chunk, dict) else getattr(chunk, 'usage', None)
                if usage:
                    if isinstance(usage, dict):
                        input_tokens = usage.get("prompt_tokens", 0) or input_tokens
                        output_tokens = usage.get("completion_tokens", 0) or output_tokens
                        total_tokens = usage.get("total_tokens", 0) or total_tokens

                # Verificar si es el último chunk
                is_finished = chunk.get('is_finished') if isinstance(chunk, dict) else getattr(chunk, 'is_finished', False)
                if is_finished:
                    finish_reason = chunk.get('finish_reason') if isinstance(chunk, dict) else getattr(chunk, 'finish_reason', 'stop')

                    # Chunk final
                    final_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason or "stop"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"

            # Enviar [DONE]
            yield "data: [DONE]\n\n"

            logger.info(f"✅ [{request_id}] Streaming completado | Chunks: {chunk_count}")

        except Exception as e:
            logger.error(f"❌ [{request_id}] Error en streaming: {type(e).__name__}: {str(e)}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked"
        }
    )


@app.post("/v1/messages")
@app.post("/messages")
async def messages_endpoint(request: Request):
    """
    Endpoint compatible con OpenAI Messages API (sin streaming).
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        body = await request.json()
        logger.info(f"📨 [{request_id}] /v1/messages request recibido")
        logger.debug(f"📋 [{request_id}] Body completo: {json.dumps(body, indent=2)}")

        logger.debug(f"🔑 [{request_id}] Headers recibidos:")
        logger.debug(f"   - Authorization: {request.headers.get('authorization', 'NO PRESENTE')}")
        logger.debug(f"   - x-api-key: {request.headers.get('x-api-key', 'NO PRESENTE')}")

        # Validar que no esté pidiendo streaming
        if _normalize_bool(body.get("stream", False)):
            raise HTTPException(
                status_code=400,
                detail="Para streaming, use el endpoint /v1/responses"
            )

        # Normalizar el formato
        if "input" in body and "messages" not in body:
            logger.info(f"🔄 [{request_id}] Formato OpenAI SDK detectado (input) - Convirtiendo")
            body["messages"] = [
                {
                    "role": "user",
                    "content": body["input"]
                }
            ]

        # Validar que tengamos messages
        if not body.get("messages"):
            logger.error(f"❌ [{request_id}] Request sin 'messages' ni 'input'")
            raise HTTPException(
                status_code=400,
                detail="Se requiere 'messages' (OpenAI format) o 'input' (OpenAI format)"
            )

        logger.info(f"📊 [{request_id}] Model: {body.get('model', 'N/A')} | Messages: {len(body.get('messages', []))}")

        # Convertir request de OpenAI a LiteLLM
        messages, kwargs = OpenAiSAIConverter.openai_to_litellm(body)

        logger.info(
            f"🔄 [{request_id}] Convertido a formato LiteLLM | "
            f"Messages: {len(messages)} | "
            f"System: {'✓' if messages and messages[0]['role'] == 'system' else '✗'}"
        )

        # Extraer user_api_key
        user_api_key = request.headers.get("x-api-key")
        if not user_api_key:
            auth_header = request.headers.get("authorization", "")
            user_api_key = auth_header.replace("Bearer ", "").strip()

        if user_api_key:
            kwargs["headers"] = {"user_api_key": user_api_key}
            logger.info(f"🔑 [{request_id}] user_api_key detectada")

        # Llamar a SAI
        logger.info(f"🚀 [{request_id}] Llamando a sai_llm.acompletion()...")
        litellm_response = await sai_llm.acompletion(messages=messages, **kwargs)

        # Convertir respuesta a formato OpenAI
        model = body.get("model")
        openai_response = OpenAiSAIConverter.litellm_to_openai_response(
            litellm_response, model, request_id
        )

        logger.info(
            f"✅ [{request_id}] Respuesta lista | "
            f"Output tokens: {openai_response['usage']['output_tokens']} | "
            f"Stop reason: {openai_response['stop_reason']}"
        )

        return JSONResponse(content=openai_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Error en /v1/messages: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/responses")
@app.post("/responses")
async def responses_endpoint(request: Request):
    """
    Endpoint compatible con OpenAI Responses API (con streaming SSE).

    Este endpoint implementa el formato oficial de OpenAI Responses API
    que es utilizado por Codex CLI. Eventos clave:
    - response.created: Inicio del response
    - response.output_text.delta: Deltas de texto
    - response.output_item.done: Item completado
    - response.done: Response finalizado (NOT response.completed)

    FIX v1.0.4: El evento final correcto según la especificación oficial
    de OpenAI es 'response.done', no 'response.completed'. Este era el
    problema que causaba los reintentos.
    """
    request_id = str(uuid.uuid4())[:8]
    response_id = f"msg_{request_id}"
    output_item_id = f"item_{request_id}"

    try:
        body = await request.json()

        logger.info(f"📨 [{request_id}] /v1/responses request recibido")
        logger.debug(f"📋 [{request_id}] Body completo: {json.dumps(body, indent=2)}")

        logger.debug(f"🔑 [{request_id}] Headers recibidos:")
        logger.debug(f"   - Authorization: {request.headers.get('authorization', 'NO PRESENTE')}")
        logger.debug(f"   - x-api-key: {request.headers.get('x-api-key', 'NO PRESENTE')}")

        # Extraer parámetro stream
        stream = _normalize_bool(body.get("stream", False))
        logger.info(f"🔀 [{request_id}] Modo: {'Streaming' if stream else 'No streaming'}")

        # Normalizar formato
        if "input" in body and "messages" not in body:
            logger.info(f"🔄 [{request_id}] Formato OpenAI SDK detectado (input) - Convirtiendo")
            body["messages"] = [
                {
                    "role": "user",
                    "content": body["input"]
                }
            ]

        # Validar messages
        if not body.get("messages"):
            logger.error(f"❌ [{request_id}] Request sin 'messages' ni 'input'")
            raise HTTPException(
                status_code=400,
                detail="Se requiere 'messages' (OpenAI format) o 'input' (OpenAI format)"
            )

        logger.info(f"📊 [{request_id}] Model: {body.get('model', 'N/A')} | Messages: {len(body.get('messages', []))}")

        # Convertir request
        messages, kwargs = OpenAiSAIConverter.openai_to_litellm(body)

        logger.info(
            f"🔄 [{request_id}] Convertido a formato LiteLLM | "
            f"Messages: {len(messages)} | "
            f"System: {'✓' if messages and messages[0]['role'] == 'system' else '✗'}"
        )

        # Extraer user_api_key
        user_api_key = request.headers.get("x-api-key")
        if not user_api_key:
            auth_header = request.headers.get("authorization", "")
            user_api_key = auth_header.replace("Bearer ", "").strip()

        if user_api_key:
            kwargs["headers"] = {"user_api_key": user_api_key}
            logger.info(f"🔑 [{request_id}] user_api_key detectada")

        model = body.get("model")

        # Decidir entre streaming y no streaming
        if stream:
            logger.info(f"🌊 [{request_id}] Modo streaming activado")
            return await _responses_streaming(request_id, response_id, output_item_id, messages, kwargs, model)
        else:
            logger.info(f"📄 [{request_id}] Modo sin streaming")
            return await _responses_non_streaming(request_id, response_id, output_item_id, messages, kwargs, model)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Error en /v1/responses: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _responses_non_streaming(request_id: str, response_id: str, output_item_id: str,
                                   messages: list, kwargs: dict, model: str):
    """
    Maneja responses sin streaming (respuesta JSON completa).
    Compatible con OpenAI Responses API format.
    """
    logger.info(f"🚀 [{request_id}] Llamando a sai_llm.acompletion()...")
    litellm_response = await sai_llm.acompletion(messages=messages, **kwargs)

    # Extraer texto
    text = ""
    if hasattr(litellm_response, 'text') and litellm_response.text:
        text = litellm_response.text
    elif hasattr(litellm_response, 'choices') and litellm_response.choices:
        choice = litellm_response.choices[0]
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            text = choice.message.content or ""

    # Extraer usage
    usage_dict = {}
    if hasattr(litellm_response, 'usage'):
        usage_obj = litellm_response.usage
        if isinstance(usage_obj, dict):
            usage_dict = usage_obj
        else:
            usage_dict = usage_obj.__dict__ if hasattr(usage_obj, '__dict__') else {}

    # Extraer finish_reason
    finish_reason = "end_turn"
    if hasattr(litellm_response, 'choices') and litellm_response.choices:
        choice = litellm_response.choices[0]
        if hasattr(choice, 'finish_reason'):
            litellm_finish = choice.finish_reason
            finish_reason_map = {
                "stop": "end_turn",
                "length": "max_tokens",
                "error": "error"
            }
            finish_reason = finish_reason_map.get(litellm_finish, "end_turn")

    # Construir respuesta en formato OpenAI Responses API
    response = {
        "id": response_id,
        "type": "response",
        "model": model,
        "created": int(time.time()),
        "output": [
            {
                "id": output_item_id,
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ],
                "stop_reason": finish_reason
            }
        ],
        "usage": {
            "input_tokens": usage_dict.get("prompt_tokens", 0),
            "cached_input_tokens": 0,
            "output_tokens": usage_dict.get("completion_tokens", 0),
            "reasoning_output_tokens": 0,
            "total_tokens": usage_dict.get("total_tokens", 0)
        }
    }

    logger.info(
        f"✅ [{request_id}] Respuesta lista | "
        f"Output tokens: {usage_dict.get('completion_tokens', 0)} | "
        f"Stop reason: {finish_reason}"
    )

    return JSONResponse(content=response)


async def _responses_streaming(request_id: str, response_id: str, output_item_id: str,
                               messages: list, kwargs: dict, model: str):
    """
    Maneja responses con streaming SSE.
    Compatible con OpenAI Responses API format y Codex CLI.
    """
    # Generar stream de eventos SSE
    async def event_generator() -> AsyncIterator[str]:
        try:
            # 🔥 EVENTO CANÓNICO: response.created
            created_event = {
                "type": "response.created",
                "response_id": response_id,
                "model": model
            }
            yield "event: response.created\n"
            yield f"data: {json.dumps(created_event)}\n\n"

            logger.info(f"🌊 [{request_id}] Iniciando streaming SSE...")

            # Variables para tracking
            chunk_count = 0
            total_text = ""
            input_tokens = 0
            output_tokens = 0
            finish_reason = "end_turn"
            first_chunk_received = False

            # Llamar a SAI streaming
            logger.info(f"🚀 [{request_id}] Llamando a sai_llm.astreaming()...")

            output_item_added = {
                "type": "response.output_item.added",
                "item": {
                    "id": output_item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": ""
                        }
                    ]
                }
            }
            yield "event: response.output_item.added\n"
            yield f"data: {json.dumps(output_item_added)}\n\n"

            try:
                async for chunk in sai_llm.astreaming(messages=messages, **kwargs):
                    chunk_count += 1

                    if chunk_count == 1:
                        logger.warning(f"⏱️ [{request_id}] PRIMER CHUNK recibido")

                    if not first_chunk_received:
                        logger.info(f"📦 [{request_id}] Primer chunk recibido de SAI")
                        first_chunk_received = True

                    # Extraer texto del chunk
                    chunk_text = chunk.get('text') if isinstance(chunk, dict) else getattr(chunk, 'text', None)

                    if VERBOSE_LOGGING:
                        logger.debug(
                            f"[{request_id}] Chunk #{chunk_count} | "
                            f"Type: {type(chunk).__name__} | "
                            f"Text length: {len(chunk_text) if chunk_text else 0}"
                        )

                    # Emitir delta de texto si hay contenido
                    if chunk_text:
                        total_text += chunk_text

                        # 🔥 FORMATO CODEX CLI: OutputTextDelta
                        text_delta_event = {
                            "type": "response.output_text.delta",
                            "item_id": output_item_id,
                            "delta": chunk_text
                        }
                        yield f"event: response.output_text.delta\n"
                        yield f"data: {json.dumps(text_delta_event)}\n\n"

                    # Extraer usage del chunk
                    usage = chunk.get('usage') if isinstance(chunk, dict) else getattr(chunk, 'usage', None)
                    if usage:
                        if isinstance(usage, dict):
                            if usage.get("prompt_tokens", 0) > 0:
                                input_tokens = usage.get("prompt_tokens", 0)
                            if usage.get("completion_tokens", 0) > 0:
                                output_tokens = usage.get("completion_tokens", 0)
                        else:
                            if getattr(usage, "prompt_tokens", 0) > 0:
                                input_tokens = getattr(usage, "prompt_tokens", 0)
                            if getattr(usage, "completion_tokens", 0) > 0:
                                output_tokens = getattr(usage, "completion_tokens", 0)

                    # Extraer finish_reason
                    chunk_finish_reason = chunk.get('finish_reason') if isinstance(chunk, dict) else getattr(chunk, 'finish_reason', None)
                    if chunk_finish_reason:
                        finish_reason_map = {
                            "stop": "end_turn",
                            "length": "max_tokens",
                            "error": "error"
                        }
                        finish_reason = finish_reason_map.get(chunk_finish_reason, "end_turn")

            except Exception as stream_error:
                logger.error(
                    f"❌ [{request_id}] Error durante streaming: {type(stream_error).__name__}: {str(stream_error)}"
                )
                # Enviar evento de error
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "internal_error",
                        "message": str(stream_error)
                    }
                }
                yield f"event: error\n"
                yield f"data: {json.dumps(error_event)}\n\n"
                return

            logger.info(
                f"📦 [{request_id}] Streaming completado | "
                f"Chunks: {chunk_count} | "
                f"Total chars: {len(total_text)}"
            )

            # 🔥 EVENTO: output_item.done
            output_item_done = {
                "type": "response.output_item.done",
                "response_id": response_id,
                "item_id": output_item_id
            }
            yield "event: response.output_item.done\n"
            yield f"data: {json.dumps(output_item_done)}\n\n"

            # 🔥 EVENTO FINAL OBLIGATORIO: response.done
            done_event = {
                "type": "response.done",
                "response_id": response_id,
                "token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": 0,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": 0,
                    "total_tokens": input_tokens + output_tokens
                }
            }
            yield "event: response.done\n"
            yield f"data: {json.dumps(done_event)}\n\n"

            logger.info(
                f"✅ [{request_id}] Stream finalizando | "
                f"Input tokens: {input_tokens} | "
                f"Output tokens: {output_tokens}"
            )

            return

        except Exception as e:
            logger.error(f"❌ [{request_id}] Error en streaming: {type(e).__name__}: {str(e)}")
            # Enviar evento de error
            error_event = {
                "type": "error",
                "error": {
                    "type": "internal_error",
                    "message": str(e)
                }
            }
            yield f"event: error\n"
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8"
        }
    )

@app.get("/v1/models")
@app.get("/models")
async def models_endpoint():
    """
    Endpoint compatible con OpenAI Models API.
    Lista los modelos disponibles en formato OpenAI.

    Este endpoint no requiere autenticación y retorna una lista
    de modelos compatibles con SAI (importada desde sai_models.py).
    """
    logger.info("📋 GET /v1/models request recibido")
    
    models_data = get_models_list()
    
    logger.info(f"✅ Retornando {len(models_data['data'])} modelos disponibles")
    return JSONResponse(content=models_data)


@app.get("/v1/models/{model_id}")
@app.get("/models/{model_id}")
async def model_detail_endpoint(model_id: str):
    """
    Endpoint compatible con OpenAI Models API.
    Retorna detalles de un modelo específico.

    Args:
        model_id: ID del modelo a consultar
    """
    logger.info(f"📋 GET /v1/models/{model_id} request recibido")
    
    model_info = get_model_by_id(model_id)
    
    if model_info:
        logger.info(f"✅ Modelo '{model_id}' encontrado")
        return JSONResponse(content=model_info)
    else:
        logger.warning(f"⚠️ Modelo '{model_id}' no encontrado")
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"The model '{model_id}' does not exist",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "model_not_found"
                }
            }
        )


@app.get("/")
async def root():
    """Documentación del gateway."""
    return {
        "service": "OpenAI SAI Gateway",
        "version": "1.0.4",
        "description": "Gateway que expone SAI con APIs compatibles con OpenAI",
        "changelog": {
            "1.0.4": "CORRECT FIX: Changed final event from response.completed to response.done (per OpenAI Responses API spec)",
            "1.0.3": "Close stream immediately after completed (still wrong event name)",
            "1.0.2": "Attempted fix with keepalives (incorrect)",
            "1.0.1": "Attempted fix with delay (incorrect)",
            "1.0.0": "Initial release"
        },
        "endpoints": {
            "completions": {
                "paths": ["/v1/completions", "/completions"],
                "method": "POST",
                "format": "OpenAI Completions (legacy)",
                "streaming": True,
                "description": "Endpoint compatible con OpenAI Completions API (reutiliza sai_handler.py)"
            },
            "chat_completions": {
                "paths": ["/v1/chat/completions", "/chat/completions"],
                "method": "POST",
                "format": "OpenAI Chat Completions",
                "streaming": True,
                "description": "Endpoint compatible con OpenAI SDK"
            },
            "messages": {
                "paths": ["/v1/messages", "/messages"],
                "method": "POST",
                "format": "OpenAI Messages",
                "streaming": False,
                "description": "Endpoint compatible con OpenAI SDK (sin streaming)"
            },
            "responses": {
                "paths": ["/v1/responses", "/responses"],
                "method": "POST",
                "format": "Codex CLI Compatible",
                "streaming": True,
                "description": "Endpoint compatible con Codex CLI (emite eventos 'output_text_delta', 'output_item_done' y 'completed')"
            },
            "health": {
                "paths": ["/health"],
                "method": "GET",
                "description": "Health check endpoint"
            }
        },
        "notes": [
            "Todos los endpoints funcionan con o sin el prefijo /v1",
            "Soporta autenticación vía Authorization header o x-api-key header",
            "El endpoint /v1/responses es el ÚNICO compatible con Codex CLI",
            "El endpoint /v1/completions reutiliza la lógica de sai_handler.py",
            "Codex CLI REQUIERE los eventos: output_text_delta, output_item_done, completed",
            "v1.0.1: Fixed timing issue where stream closed before completed event was processed"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "openai-sai-gateway",
        "version": "1.0.4"
    }


if __name__ == "__main__":
    import signal
    import sys

    print("\n" + "="*80)
    print("🚀 OPENAI SAI GATEWAY v1.0.4")
    print("="*80)
    logger.info("📍 Endpoints disponibles:")
    logger.info("   - POST /v1/completions  (o /completions)")
    logger.info("           OpenAI Completions format (legacy) - streaming y no streaming")
    logger.info("   - POST /v1/chat/completions  (o /chat/completions)")
    logger.info("           OpenAI format - streaming y no streaming")
    logger.info("   - POST /v1/messages  (o /messages)")
    logger.info("           OpenAI format - sin streaming")
    logger.info("   - POST /v1/responses  (o /responses)")
    logger.info("           OpenAI Responses API format - con streaming SSE")
    logger.info("   - GET /health")
    logger.info("")
    logger.info("💡 Todos los endpoints funcionan con o sin el prefijo /v1")
    logger.info("🔧 FIX v1.0.4: Evento final correcto: response.done (no completed)")
    logger.info("🌐 Servidor iniciando en http://0.0.0.0:8000")
    logger.info("💡 Presiona Ctrl+C para detener")
    print("="*80 + "\n")

    def signal_handler(sig, frame):
        """Manejar Ctrl+C limpiamente."""
        print("\n" + "="*80)
        logger.info("👋 Deteniendo gateway... (Ctrl+C recibido)")
        logger.info("✅ Gateway detenido exitosamente")
        print("="*80 + "\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            timeout_keep_alive=75,
            timeout_graceful_shutdown=30
        )
    except KeyboardInterrupt:
        print("\n" + "="*80)
        logger.info("👋 Gateway detenido limpiamente")
        print("="*80 + "\n")
        sys.exit(0)