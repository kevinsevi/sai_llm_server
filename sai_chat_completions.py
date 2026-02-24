# sai_chat_completions.py

import time
import json

from sai_handler import sai_llm, logger
from starlette.responses import Response, StreamingResponse

class ChatCompletionsHandler:
    """Handler para endpoints de chat completions en formato OpenAI."""

    async def chat_completions_non_streaming(self, request_id: str, messages: list, kwargs: dict, model: str):
        """Maneja requests de chat completions sin streaming."""
        logger.info(f"🚀 [{request_id}] Llamando a sai_llm.acompletion()...")
        litellm_response = await sai_llm.acompletion(request_id, messages=messages, **kwargs)

        # Extraer texto y tool_calls
        text = ""
        tool_calls = None

        if hasattr(litellm_response, 'choices') and litellm_response.choices:
            choice = litellm_response.choices[0]

            # Extraer tool_calls si existen
            if hasattr(choice, 'message'):
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    tool_calls = choice.message.tool_calls
                # Siempre extraer content (puede coexistir con tool_calls)
                if hasattr(choice.message, 'content'):
                    text = choice.message.content or ""
        elif hasattr(litellm_response, 'text') and litellm_response.text:
            text = litellm_response.text

        # Extraer usage
        usage_dict = {}
        if hasattr(litellm_response, 'usage'):
            usage_obj = litellm_response.usage
            if isinstance(usage_obj, dict):
                usage_dict = usage_obj
            else:
                usage_dict = usage_obj.__dict__ if hasattr(usage_obj, '__dict__') else {}

        # Extraer finish_reason
        finish_reason = "tool_calls" if tool_calls else "stop"
        if hasattr(litellm_response, 'choices') and litellm_response.choices:
            choice = litellm_response.choices[0]
            if hasattr(choice, 'finish_reason') and choice.finish_reason:
                finish_reason = choice.finish_reason

        # Construir mensaje de respuesta
        response_message = {
            "role": "assistant",
            "content": text if text else None  # Incluir texto si existe, null si está vacío
        }

        # Agregar tool_calls si existen
        if tool_calls:
            # Convertir tool_calls a formato dict si es necesario
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            response_message["tool_calls"] = []
            for tc in tool_calls:
                if hasattr(tc, '__dict__'):
                    tc_dict = {
                        "id": getattr(tc, 'id', f"call_{request_id}"),
                        "type": getattr(tc, 'type', 'function'),
                        "function": {
                            "name": getattr(tc.function, 'name', '') if hasattr(tc, 'function') else '',
                            "arguments": getattr(tc.function, 'arguments', '{}') if hasattr(tc, 'function') else '{}'
                        }
                    }
                else:
                    tc_dict = tc
                response_message["tool_calls"].append(tc_dict)

            # Agregar campos adicionales requeridos por OpenAI
            response_message["refusal"] = None
            response_message["annotations"] = []

        # Construir respuesta en formato OpenAI
        openai_response = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": usage_dict.get("prompt_tokens", 0),
                "completion_tokens": usage_dict.get("completion_tokens", 0),
                "total_tokens": usage_dict.get("total_tokens", 0),
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "audio_tokens": 0
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            },
            "service_tier": "default",
            "system_fingerprint": None
        }

        logger.info(
            f"✅ [{request_id}] Respuesta lista | "
            f"Output tokens: {usage_dict.get('completion_tokens', 0)} | "
            f"Tool calls: {len(tool_calls) if tool_calls else 0} | "
            f"Content length: {len(text) if text else 0}"
        )

        # Retornar JSON formateado con indentación
        return Response(
            content=json.dumps(openai_response, indent=4, ensure_ascii=False) + "\n",
            media_type="application/json",
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    async def chat_completions_streaming(self, request_id: str, messages: list, kwargs: dict, model: str):
        """Maneja requests de chat completions con streaming."""
        async def event_generator():
            try:
                logger.info(f"🌊 [{request_id}] Iniciando streaming...")

                # Llamar a SAI streaming
                logger.info(f"🚀 [{request_id}] Llamando a sai_llm.astreaming()...")

                chunk_count = 0
                tool_calls_emitted = False
                first_content_chunk = True
                created_timestamp = int(time.time())

                async for chunk in sai_llm.astreaming(request_id, messages=messages, **kwargs):
                    chunk_count += 1

                    if chunk_count == 1:
                        logger.warning(f"⏱️ [{request_id}] PRIMER CHUNK recibido")

                    # Extraer tool_use del chunk
                    tool_use = chunk.get('tool_use') if isinstance(chunk, dict) else getattr(chunk, 'tool_use', None)

                    # Normalizar tool_use a lista (puede venir como dict o un solo objeto)
                    if tool_use and not isinstance(tool_use, list):
                        tool_use = [tool_use]

                    # Si hay tool_calls, emitirlos en formato streaming
                    if tool_use and not tool_calls_emitted:
                        # Chunk inicial con role
                        initial_chunk = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": model,
                            "service_tier": "default",
                            "system_fingerprint": None,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "tool_calls": []
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }

                        # Agregar tool_calls iniciales con metadata
                        for idx, tc in enumerate(tool_use):
                            initial_chunk["choices"][0]["delta"]["tool_calls"].append({
                                "index": idx,
                                "id": tc.get("id"),
                                "type": tc.get("type"),
                                "function": {
                                    "name": tc.get("function", {}).get("name"),
                                    "arguments": ""
                                }
                            })

                        yield f"data: {json.dumps(initial_chunk)}\n\n"

                        # Emitir argumentos en chunks
                        for idx, tc in enumerate(tool_use):
                            arguments = tc.get("function", {}).get("arguments", "{}")

                            # Asegurar que arguments sea string antes de hacer slicing
                            if not isinstance(arguments, str):
                                arguments = json.dumps(arguments, ensure_ascii=False)

                            # Dividir arguments en chunks pequeños
                            chunk_size = 20
                            for i in range(0, len(arguments), chunk_size):
                                arg_chunk = arguments[i:i + chunk_size]

                                arg_delta_chunk = {
                                    "id": f"chatcmpl-{request_id}",
                                    "object": "chat.completion.chunk",
                                    "created": created_timestamp,
                                    "model": model,
                                    "service_tier": "default",
                                    "system_fingerprint": None,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "index": idx,
                                                        "function": {
                                                            "arguments": arg_chunk
                                                        }
                                                    }
                                                ]
                                            },
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(arg_delta_chunk)}\n\n"

                        tool_calls_emitted = True

                    # Extraer texto del chunk (si no hay tool_calls)
                    chunk_text = chunk.get('text') if isinstance(chunk, dict) else getattr(chunk, 'text', None)

                    if chunk_text and not tool_use:
                        # Chunk inicial con role (solo si no se emitió con tool_calls)
                        if first_content_chunk:
                            initial_chunk = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": created_timestamp,
                                "model": model,
                                "service_tier": "default",
                                "system_fingerprint": None,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": "",
                                            "refusal": None
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(initial_chunk)}\n\n"
                            first_content_chunk = False

                        # Emitir el chunk de texto directamente (sin dividir por palabras)
                        # Los chunks ya vienen en tamaño ~50 caracteres desde sai_llm.astreaming()
                        content_chunk = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": model,
                            "service_tier": "default",
                            "system_fingerprint": None,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk_text},
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
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": model,
                            "service_tier": "default",
                            "system_fingerprint": None,
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


# Instancia global
chat_completions_handler = ChatCompletionsHandler()