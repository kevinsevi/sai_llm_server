# sai_converter.py

import time
import json
import uuid

from sai_handler import sai_llm, logger, VERBOSE_LOGGING
from starlette.responses import Response, StreamingResponse
from typing import AsyncIterator


# ---------------- Conversor de formatos OpenAI ----------------
class OpenAiSAIConverter:
    @staticmethod
    def _extract_text_recursive(content) -> str:
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

            # Prioridad 4: Si tiene "output" (formato Codex CLI)
            if "arguments" in content:
                return "EJECUTADO: " + str(content["arguments"])

            # Prioridad 4: Si tiene "output" (formato Codex CLI)
            if "output" in content:
                return "RESULTADO: " + str(content["output"])

            # Si no tiene ninguno de los campos esperados, retornar vacío
            return ""

        # Caso 4: Otro tipo (None, int, etc.)
        return ""

    @staticmethod
    def openai_to_litellm(openai_request: dict) -> tuple[list, dict]:
        messages = []

        # Extraer system prompt si existe
        system_prompt = openai_request.get("system", "")
        instructions = openai_request.get("instructions")
        system_text = None
        if system_prompt:
            system_text = OpenAiSAIConverter._extract_text_recursive(system_prompt)

        if instructions and system_text:
            system_text = system_text + "\n\n" + instructions
        elif instructions:
            system_text = instructions

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
            "max_tokens": openai_request.get("max_tokens"),
            "top_p": openai_request.get("top_p"),
            "top_k": openai_request.get("top_k"),
            "stop_sequences": openai_request.get("stop_sequences"),
        }

        # Agregar tools si existen y no es lista vacía
        tools = openai_request.get("tools")
        if tools is not None:
            # Omitir si es lista vacía
            if isinstance(tools, list) and len(tools) == 0:
                logger.info("[TOOLS] openai_to_litellm(): tools es lista vacía [] - omitiendo")
            else:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = openai_request.get("tool_choice", "auto")

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

    async def _completions_non_streaming(self, request_id: str, messages: list, kwargs: dict, model: str):
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

        # Retornar JSON formateado con indentación
        return Response(
            content=json.dumps(openai_response, indent=4, ensure_ascii=False) + "\n",
            media_type="application/json",
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    async def _completions_streaming(self, request_id: str, messages: list, kwargs: dict, model: str):
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

    async def _chat_completions_non_streaming(self, request_id: str, messages: list, kwargs: dict, model: str):
        logger.info(f"🚀 [{request_id}] Llamando a sai_llm.acompletion()...")
        litellm_response = await sai_llm.acompletion(messages=messages, **kwargs)

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

    async def _chat_completions_streaming(self, request_id: str, messages: list, kwargs: dict, model: str):
        async def event_generator():
            try:
                logger.info(f"🌊 [{request_id}] Iniciando streaming...")

                # Llamar a SAI streaming
                logger.info(f"🚀 [{request_id}] Llamando a sai_llm.astreaming()...")

                chunk_count = 0
                tool_calls_emitted = False
                first_content_chunk = True
                created_timestamp = int(time.time())

                async for chunk in sai_llm.astreaming(messages=messages, **kwargs):
                    chunk_count += 1

                    if chunk_count == 1:
                        logger.warning(f"⏱️ [{request_id}] PRIMER CHUNK recibido")

                    # Extraer tool_use del chunk
                    tool_use = chunk.get('tool_use') if isinstance(chunk, dict) else getattr(chunk, 'tool_use', None)

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

    async def _responses_non_streaming(self, request_id: str, output_item_id: str,
                                       messages: list, kwargs: dict, model: str):
        logger.info(f"🚀 [{request_id}] Llamando a sai_llm.acompletion()...")

        # Timestamp de inicio
        created_at = int(time.time())

        litellm_response = await sai_llm.acompletion(messages=messages, **kwargs)

        # Timestamp de finalización
        completed_at = int(time.time())

        # Extraer texto y tool_calls
        text = ""
        tool_calls = None

        if hasattr(litellm_response, 'choices') and litellm_response.choices:
            choice = litellm_response.choices[0]

            # Extraer tool_calls si existen
            if hasattr(choice, 'message'):
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    tool_calls = choice.message.tool_calls
                    logger.info(
                        f"🔧 [{request_id}] [RESPONSES] Tool calls detectados | "
                        f"Count: {len(tool_calls)}"
                    )

                # Extraer content (puede coexistir con tool_calls)
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
        finish_reason = "end_turn"
        if hasattr(litellm_response, 'choices') and litellm_response.choices:
            choice = litellm_response.choices[0]
            if hasattr(choice, 'finish_reason'):
                litellm_finish = choice.finish_reason
                finish_reason_map = {
                    "stop": "end_turn",
                    "tool_calls": "tool_calls",
                    "length": "max_tokens",
                    "error": "error"
                }
                finish_reason = finish_reason_map.get(litellm_finish, "end_turn")

        # Construir output según si hay tool_calls o texto
        output = []

        if tool_calls:
            # Formato para function_call (compatible con Responses API)
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

                output.append({
                    "type": "function_call",
                    "id": f"fc_{request_id}",
                    "call_id": tc_dict.get("id"),
                    "name": tc_dict.get("function", {}).get("name"),
                    "arguments": tc_dict.get("function", {}).get("arguments"),
                    "status": "completed"
                })

            logger.info(
                f"🔧 [{request_id}] [RESPONSES] Output con function_calls | "
                f"Count: {len(output)}"
            )
        else:
            # Formato para mensaje de texto normal
            output.append({
                "id": output_item_id,
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": text
                    }
                ],
                "role": "assistant"
            })

        response_id = f"msg_{request_id}"

        # Construir respuesta en formato OpenAI Responses API completo
        response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "background": False,
            "billing": {
                "payer": "developer"
            },
            "completed_at": completed_at,
            "error": None,
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": kwargs.get("max_tokens"),
            "max_tool_calls": None,
            "model": model,
            "output": output,
            "parallel_tool_calls": True,
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
            "previous_response_id": None,
            "prompt_cache_key": None,
            "prompt_cache_retention": None,
            "reasoning": {
                "effort": "none",
                "summary": None
            },
            "safety_identifier": None,
            "service_tier": "default",
            "store": True,
            "temperature": kwargs.get("temperature", 1.0),
            "text": {
                "format": {
                    "type": "text"
                },
                "verbosity": "medium"
            },
            "tool_choice": kwargs.get("tool_choice", "auto"),
            "tools": kwargs.get("tools", []),
            "top_logprobs": 0,
            "top_p": kwargs.get("top_p", 1.0),
            "truncation": "disabled",
            "usage": {
                "input_tokens": usage_dict.get("prompt_tokens", 0),
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens": usage_dict.get("completion_tokens", 0),
                "output_tokens_details": {
                    "reasoning_tokens": 0
                },
                "total_tokens": usage_dict.get("total_tokens", 0)
            },
            "user": None,
            "metadata": {}
        }

        logger.info(
            f"✅ [{request_id}] Respuesta lista | "
            f"Output tokens: {usage_dict.get('completion_tokens', 0)} | "
            f"Output type: {'function_call' if tool_calls else 'message'} | "
            f"Duration: {completed_at - created_at}s"
        )

        # Retornar JSON formateado con indentación
        return Response(
            content=json.dumps(response, indent=2, ensure_ascii=False) + "\n",
            media_type="application/json",
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    async def _responses_streaming(self, request_id: str, output_item_id: str,
                                   messages: list, kwargs: dict, model: str, has_reasoning=False):
        """
        Maneja requests a /v1/responses con streaming SSE.

        Compatible con OpenAI Responses API y Codex CLI.
        Emite eventos canónicos:
        - response.created
        - response.output_item.added
        - response.output_text.delta
        - response.output_item.done
        - response.done

        Args:
            request_id: ID único de la solicitud
            response_id: ID de la respuesta
            output_item_id: ID del item de salida
            messages: Lista de mensajes procesados
            kwargs: Parámetros adicionales para LiteLLM
            model: Nombre del modelo

        Returns:
            StreamingResponse: Stream de eventos SSE con formato Codex CLI
        """
        # Generar stream de eventos SSE
        async def event_generator() -> AsyncIterator[str]:
            try:
                # 🔥 EVENTO CANÓNICO: response.created
                # Timestamp de creación
                created_at = int(time.time())

                response_id = f"resp_{uuid.uuid4().hex[:50]}"

                created_event = {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_at,
                        "status": "in_progress",
                        "background": False,
                        "completed_at": None,
                        "error": None,
                        "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                        "incomplete_details": None,
                        "instructions": kwargs.get("instructions"),
                        "max_output_tokens": kwargs.get("max_tokens"),
                        "max_tool_calls": None,
                        "model": model,
                        "output": [],
                        "parallel_tool_calls": True,
                        "presence_penalty": kwargs.get("presence_penalty", 0.0),
                        "previous_response_id": None,
                        "prompt_cache_key": None,
                        "prompt_cache_retention": None,
                        "reasoning": {
                            "effort": "none",
                            "summary": None
                        },
                        "safety_identifier": None,
                        "service_tier": "auto",
                        "store": True,
                        "temperature": kwargs.get("temperature", 1.0),
                        "text": {
                            "format": {
                                "type": "text"
                            },
                            "verbosity": "medium"
                        },
                        "tool_choice": kwargs.get("tool_choice", "auto"),
                        "tools": kwargs.get("tools", []),
                        "top_logprobs": 0,
                        "top_p": kwargs.get("top_p", 1.0),
                        "truncation": "disabled",
                        "usage": None,
                        "user": None,
                        "metadata": {}
                    },
                    "sequence_number": 0
                }
                yield "event: response.created\n"
                yield f"data: {json.dumps(created_event)}\n\n"

                # 🔥 EVENTO CANÓNICO: response.in_progress
                in_progress_event = {
                    "type": "response.in_progress",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_at,
                        "status": "in_progress",
                        "background": False,
                        "completed_at": None,
                        "error": None,
                        "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                        "incomplete_details": None,
                        "instructions": kwargs.get("instructions"),
                        "max_output_tokens": kwargs.get("max_tokens"),
                        "max_tool_calls": None,
                        "model": model,
                        "output": [],
                        "parallel_tool_calls": True,
                        "presence_penalty": kwargs.get("presence_penalty", 0.0),
                        "previous_response_id": None,
                        "prompt_cache_key": None,
                        "prompt_cache_retention": None,
                        "reasoning": {
                            "effort": "none",
                            "summary": None
                        },
                        "safety_identifier": None,
                        "service_tier": "auto",
                        "store": True,
                        "temperature": kwargs.get("temperature", 1.0),
                        "text": {
                            "format": {
                                "type": "text"
                            },
                            "verbosity": "medium"
                        },
                        "tool_choice": kwargs.get("tool_choice", "auto"),
                        "tools": kwargs.get("tools", []),
                        "top_logprobs": 0,
                        "top_p": kwargs.get("top_p", 1.0),
                        "truncation": "disabled",
                        "usage": None,
                        "user": None,
                        "metadata": {}
                    },
                    "sequence_number": 1
                }
                yield "event: response.in_progress\n"
                yield f"data: {json.dumps(in_progress_event)}\n\n"

                logger.info(f"🌊 [{request_id}] Iniciando streaming SSE...")

                # Variables para tracking
                chunk_count = 0
                total_text = ""
                input_tokens = 0
                output_tokens = 0
                finish_reason = "end_turn"
                first_chunk_received = False

                # Variables para function_call streaming
                function_call_emitted = False
                function_name = None
                function_arguments_buffer = ""

                # Llamar a SAI streaming
                logger.info(f"🚀 [{request_id}] Llamando a sai_llm.astreaming()...")

                # 🔥 EVENTO CANÓNICO: response.output_item.added
                # Determinar el tipo de item según has_reasoning
                item_type = "reasoning" if has_reasoning else "message"

                rs_item_id = f"rs_{uuid.uuid4().hex[:50]}" if has_reasoning else f"msg_{uuid.uuid4().hex[:50]}"

                output_item_added = {
                    "type": "response.output_item.added",
                    "item": {
                        "id": rs_item_id,
                        "type": item_type,
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

                if has_reasoning:
                    output_item_done = {
                        "type": "response.output_item.done",
                        "item": {
                            "id": rs_item_id,
                            "type": item_type,
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": ""
                                }
                            ]
                        }
                    }
                    yield "event: response.output_item.done\n"
                    yield f"data: {json.dumps(output_item_done)}\n\n"

                fc_item_id = uuid.uuid4().hex[:50]
                call_item_id = uuid.uuid4().hex[:24]

                if has_reasoning:
                    output_item_added = {
                        "type": "response.output_item.added",
                        "item": {
                            "id": f"fc_{fc_item_id}",
                            "type": "function_call",
                            "status": "in_progress",
                            "arguments": "",
                            "call_id": f"call_{call_item_id}",
                            "name": "exec_command"
                        }
                    }
                    yield "event: response.output_item.added\n"
                    yield f"data: {json.dumps(output_item_added)}\n\n"
                else:
                    # 🔥 EVENTO CANÓNICO: response.content_part.added (después de output_item.added)
                    # Algunos clientes esperan este evento antes de comenzar a recibir deltas.
                    content_part_added = {
                        "type": "response.content_part.added",
                        "item_id": output_item_id,
                        "part": {
                            "type": "output_text",
                            "text": ""
                        },
                        "index": 0
                    }
                    yield "event: response.content_part.added\n"
                    yield f"data: {json.dumps(content_part_added)}\n\n"

                try:
                    async for chunk in sai_llm.astreaming(messages=messages, **kwargs):
                        chunk_count += 1

                        if chunk_count == 1:
                            logger.warning(f"⏱️ [{request_id}] PRIMER CHUNK recibido")

                        if not first_chunk_received:
                            logger.info(f"📦 [{request_id}] Primer chunk recibido de SAI")
                            first_chunk_received = True

                        # Extraer tool_use del chunk
                        tool_use = chunk.get('tool_use') if isinstance(chunk, dict) else getattr(chunk, 'tool_use', None)

                        # 🔧 NUEVO: Soporte para function_call streaming
                        if tool_use and not function_call_emitted:
                            # Detectar si es un solo tool_call (function_call legacy)
                            if len(tool_use) == 1:
                                tc = tool_use[0]
                                function_name = tc.get("function", {}).get("name")
                                function_arguments = tc.get("function", {}).get("arguments", "{}")

                                logger.info(
                                    f"🔧 [{request_id}] [FUNCTION_CALL] Detectado function_call único | "
                                    f"Name: {function_name} | "
                                    f"Arguments length: {len(function_arguments)}"
                                )

                                if not has_reasoning:
                                    # Emitir evento function_call.started
                                    function_call_started = {
                                        "type": "response.function_call.started",
                                        "item_id": output_item_id,
                                        "call_id": tc.get("id", f"call_{request_id}"),
                                        "name": function_name
                                    }
                                    yield "event: response.function_call.started\n"
                                    yield f"data: {json.dumps(function_call_started)}\n\n"

                                # Emitir argumentos en chunks pequeños
                                chunk_size = 20
                                for i in range(0, len(function_arguments), chunk_size):
                                    arg_chunk = function_arguments[i:i + chunk_size]
                                    function_arguments_buffer += arg_chunk

                                    function_call_delta = {
                                        "type": "response.function_call.arguments.delta",
                                        "item_id": output_item_id,
                                        "call_id": tc.get("id", f"call_{request_id}"),
                                        "delta": arg_chunk
                                    }
                                    yield "event: response.function_call.arguments.delta\n"
                                    yield f"data: {json.dumps(function_call_delta)}\n\n"

                                if has_reasoning:
                                    function_call_arguments_done = {
                                        "type": "response.function_call_arguments.done",
                                        "arguments": function_arguments,
                                        "item_id": f"fc_{fc_item_id}",
                                        "output_index": 1
                                    }
                                    yield "event: response.function_call_arguments.done\n"
                                    yield f"data: {json.dumps(function_call_arguments_done)}\n\n"
                                else:
                                    # Emitir evento function_call.completed
                                    function_call_completed = {
                                        "type": "response.function_call.completed",
                                        "item_id": output_item_id,
                                        "call_id": tc.get("id", f"call_{request_id}"),
                                        "name": function_name,
                                        "arguments": function_arguments
                                    }
                                    yield "event: response.function_call.completed\n"
                                    yield f"data: {json.dumps(function_call_completed)}\n\n"

                                function_call_emitted = True
                                finish_reason = "function_call"

                            else:
                                # Múltiples tool_calls - usar formato tool_calls estándar
                                logger.info(
                                    f"🔧 [{request_id}] [TOOL_CALLS] Detectados múltiples tool_calls | "
                                    f"Count: {len(tool_use)}"
                                )

                                # Chunk inicial con role
                                initial_chunk = {
                                    "id": f"chatcmpl-{request_id}",
                                    "object": "chat.completion.chunk",
                                    "created": created_at,
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

                                    # Dividir arguments en chunks pequeños
                                    chunk_size = 20
                                    for i in range(0, len(arguments), chunk_size):
                                        arg_chunk = arguments[i:i + chunk_size]

                                        arg_delta_chunk = {
                                            "id": f"chatcmpl-{request_id}",
                                            "object": "chat.completion.chunk",
                                            "created": created_at,
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

                                function_call_emitted = True
                                finish_reason = "tool_calls"

                        # Extraer texto del chunk (si no hay tool_calls)
                        chunk_text = chunk.get('text') if isinstance(chunk, dict) else getattr(chunk, 'text', None)

                        if VERBOSE_LOGGING:
                            logger.debug(
                                f"[{request_id}] Chunk #{chunk_count} | "
                                f"Type: {type(chunk).__name__} | "
                                f"Text length: {len(chunk_text) if chunk_text else 0}"
                            )

                        if chunk_text and not tool_use:
                            total_text += chunk_text

                            # 🔥 FORMATO CODEX CLI: OutputTextDelta
                            if has_reasoning:
                                function_call_arguments_delta_event = {
                                    "type": "response.function_call_arguments.delta",
                                    "delta": chunk_text,
                                    "item_id": f"fc_{fc_item_id}"
                                }
                                yield f"event: response.function_call_arguments.delta\n"
                                yield f"data: {json.dumps(function_call_arguments_delta_event)}\n\n"
                            else:
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
                        if chunk_finish_reason and not function_call_emitted:
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
                    f"Total chars: {len(total_text)} | "
                    f"Function call: {function_call_emitted}"
                )

                # 🔥 EVENTO CANÓNICO: response.output_text.done (solo si hay texto)
                if total_text:
                    if has_reasoning:
                        function_call_arguments_done = {
                            "type": "response.function_call_arguments.done",
                            "arguments": total_text,
                            "item_id": f"fc_{fc_item_id}",
                            "output_index": 1
                        }
                        yield "event: response.function_call_arguments.done\n"
                        yield f"data: {json.dumps(function_call_arguments_done)}\n\n"
                    else:
                        output_text_done = {
                            "type": "response.output_text.done",
                            "item_id": output_item_id,
                            "index": 0,
                            "text": total_text
                        }
                        yield "event: response.output_text.done\n"
                        yield f"data: {json.dumps(output_text_done)}\n\n"

                        # 🔥 EVENTO CANÓNICO: response.content_part.done
                        content_part_done = {
                            "type": "response.content_part.done",
                            "item_id": output_item_id,
                            "index": 0,
                            "part": {
                                "type": "output_text",
                                "text": total_text
                            }
                        }
                        yield "event: response.content_part.done\n"
                        yield f"data: {json.dumps(content_part_done)}\n\n"

                if has_reasoning:
                    # 🔥 EVENTO: output_item.done
                    output_item_done = {
                        "type": "response.output_item.done",
                        "item": {
                            "id": f"fc_{fc_item_id}",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": "{\"cmd\":\"ls\",\"yield_time_ms\":1000,\"max_output_tokens\":6000}",
                            "call_id": f"call_{call_item_id}",
                            "name": "exec_command"
                        },
                        "output_index": 0,
                        "sequence_number": 7
                    }

                    yield "event: response.output_item.done\n"
                    yield f"data: {json.dumps(output_item_done)}\n\n"

                    # 🔥 EVENTO CANÓNICO (compat): response.completed
                    # Algunos clientes esperan este evento antes del evento final response.done.
                    completed_at = int(time.time())

                    # Construir output según si hay tool_calls o texto
                    output = [{
                        "type": "function_call",
                        "id": f"fc_{fc_item_id}",
                        "call_id": f"call_{call_item_id}",
                        "name": function_name,
                        "arguments": function_arguments_buffer,
                        "status": "completed"
                    }]

                    # Formato para function_call

                    completed_event = {
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created_at": created_at,
                            "status": "completed",
                            "background": False,
                            "completed_at": completed_at,
                            "error": None,
                            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                            "incomplete_details": None,
                            "instructions": kwargs.get("instructions"),
                            "max_output_tokens": kwargs.get("max_tokens"),
                            "max_tool_calls": None,
                            "metadata": {},
                            "model": model,
                            "output": output,
                            "parallel_tool_calls": True,
                            "presence_penalty": kwargs.get("presence_penalty", 0.0),
                            "previous_response_id": None,
                            "prompt_cache_key": None,
                            "prompt_cache_retention": None,
                            "reasoning": {
                                "effort": "none",
                                "summary": None
                            },
                            "safety_identifier": None,
                            "service_tier": "default",
                            "store": True,
                            "temperature": kwargs.get("temperature", 1.0),
                            "text": {
                                "format": {
                                    "type": "text"
                                },
                                "verbosity": "medium"
                            },
                            "tool_choice": kwargs.get("tool_choice", "auto"),
                            "tools": kwargs.get("tools", []),
                            "top_logprobs": 0,
                            "top_p": kwargs.get("top_p", 0.98),
                            "truncation": "disabled",
                            "usage": {
                                "input_tokens": input_tokens,
                                "input_tokens_details": {
                                    "cached_tokens": 0
                                },
                                "output_tokens": output_tokens,
                                "output_tokens_details": {
                                    "reasoning_tokens": 0
                                },
                                "total_tokens": input_tokens + output_tokens
                            },
                            "user": None
                        },
                        "sequence_number": 8
                    }
                    yield "event: response.completed\n"
                    yield f"data: {json.dumps(completed_event)}\n\n"
                else:
                    # 🔥 EVENTO: output_item.done
                    output_item_done = {
                        "type": "response.output_item.done",
                        "item": {
                            "id": output_item_id,
                            "type": "message",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "annotations": [],
                                    "logprobs": [],
                                    "text": total_text
                                }
                            ],
                            "role": "assistant"
                        },
                        "output_index": 0,
                        "sequence_number": 7
                    }

                    # Si hubo function_call, agregar al output_item
                    if function_call_emitted and function_name:
                        output_item_done["item"]["function_call"] = {
                            "name": function_name,
                            "arguments": function_arguments_buffer
                        }

                    yield "event: response.output_item.done\n"
                    yield f"data: {json.dumps(output_item_done)}\n\n"

                    # 🔥 EVENTO CANÓNICO (compat): response.completed
                    # Algunos clientes esperan este evento antes del evento final response.done.
                    completed_at = int(time.time())

                    # Construir output según si hay tool_calls o texto
                    output = []

                    if function_call_emitted and function_name:
                        # Formato para function_call
                        output.append({
                            "type": "function_call",
                            "id": f"fc_{request_id}",
                            "call_id": f"call_{request_id}",
                            "name": function_name,
                            "arguments": function_arguments_buffer,
                            "status": "completed"
                        })
                    else:
                        # Formato para mensaje de texto normal
                        output.append({
                            "id": output_item_id,
                            "type": "message",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "annotations": [],
                                    "logprobs": [],
                                    "text": total_text
                                }
                            ],
                            "role": "assistant"
                        })

                    completed_event = {
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created_at": created_at,
                            "completed_at": completed_at,
                            "status": "completed",
                            "background": False,
                            "error": None,
                            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                            "incomplete_details": None,
                            "instructions": kwargs.get("instructions"),
                            "max_output_tokens": kwargs.get("max_tokens"),
                            "max_tool_calls": None,
                            "metadata": {},
                            "model": model,
                            "output": output,
                            "parallel_tool_calls": True,
                            "presence_penalty": kwargs.get("presence_penalty", 0.0),
                            "previous_response_id": None,
                            "prompt_cache_key": None,
                            "prompt_cache_retention": None,
                            "reasoning": {
                                "effort": "none",
                                "summary": None
                            },
                            "safety_identifier": None,
                            "service_tier": "default",
                            "store": True,
                            "temperature": kwargs.get("temperature", 1.0),
                            "text": {
                                "format": {
                                    "type": "text"
                                },
                                "verbosity": "medium"
                            },
                            "tool_choice": kwargs.get("tool_choice", "auto"),
                            "tools": kwargs.get("tools", []),
                            "top_logprobs": 0,
                            "top_p": kwargs.get("top_p", 0.98),
                            "truncation": "disabled",
                            "usage": {
                                "input_tokens": input_tokens,
                                "input_tokens_details": {
                                    "cached_tokens": 0
                                },
                                "output_tokens": output_tokens,
                                "output_tokens_details": {
                                    "reasoning_tokens": 0
                                },
                                "total_tokens": input_tokens + output_tokens
                            },
                            "user": None
                        },
                        "sequence_number": 8
                    }
                    yield "event: response.completed\n"
                    yield f"data: {json.dumps(completed_event)}\n\n"

                # 🔥 EVENTO FINAL OBLIGATORIO: response.done
                # done_event = {
                #     "type": "response.done",
                #     "response_id": response_id,
                #     "token_usage": {
                #         "input_tokens": input_tokens,
                #         "cached_input_tokens": 0,
                #         "output_tokens": output_tokens,
                #         "reasoning_output_tokens": 0,
                #         "total_tokens": input_tokens + output_tokens
                #     }
                # }
                # yield "event: response.done\n"
                # yield f"data: {json.dumps(done_event)}\n\n"

                logger.info(
                    f"✅ [{request_id}] Stream finalizando | "
                    f"Input tokens: {input_tokens} | "
                    f"Output tokens: {output_tokens} | "
                    f"Finish reason: {finish_reason}"
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

# ---------------- Instancia global ----------------
converter = OpenAiSAIConverter()
"""Instancia global del conversor de formatos OpenAI ↔ SAI."""