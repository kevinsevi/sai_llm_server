# sai_responses.py

import time
import json
import uuid

from sai_handler import sai_llm, logger, VERBOSE_LOGGING
from sai_builder import event_builder
from starlette.responses import Response, StreamingResponse
from typing import AsyncIterator

class ResponsesHandler:
    """Handler para el endpoint /v1/responses con soporte streaming y no streaming."""

    async def responses_non_streaming(self, request_id: str, output_item_id: str,
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

    async def responses_streaming(self, request_id: str, output_item_id: str,
                                  messages: list, kwargs: dict, model: str, has_reasoning=False):
        # Generar stream de eventos SSE
        async def event_generator() -> AsyncIterator[str]:
            try:
                # 🔢 Inicializar contador de secuencia
                sequence_number = 0

                # 🔥 EVENTO CANÓNICO: response.created
                # Timestamp de creación
                created_at = int(time.time())

                response_id = f"resp_{uuid.uuid4().hex[:50]}"

                response_created = event_builder.build_response_created_event(
                    response_id=response_id,
                    created_at=created_at,
                    kwargs=kwargs,
                    model=model,
                    sequence_number=sequence_number
                )
                sequence_number += 1
                yield "event: response.created\n"
                yield f"data: {json.dumps(response_created)}\n\n"

                # 🔥 EVENTO CANÓNICO: response.in_progress
                response_in_progress = event_builder.build_response_in_progress_event(
                    response_id=response_id,
                    created_at=created_at,
                    kwargs=kwargs,
                    model=model,
                    sequence_number=sequence_number
                )
                sequence_number += 1
                yield "event: response.in_progress\n"
                yield f"data: {json.dumps(response_in_progress)}\n\n"

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

                response_output_item_added = event_builder.build_response_output_item_added_event(
                    item_id=rs_item_id if has_reasoning else f"msg_{uuid.uuid4().hex[:50]}",
                    item_type="reasoning" if has_reasoning else "message",
                    sequence_number=sequence_number
                )
                sequence_number += 1
                yield "event: response.output_item.added\n"
                yield f"data: {json.dumps(response_output_item_added)}\n\n"

                if has_reasoning:
                    response_output_item_done = event_builder.build_response_output_item_done_event(
                        item_id=rs_item_id,
                        item_type=item_type,
                        text="",
                        sequence_number=sequence_number
                    )
                    sequence_number += 1
                    yield "event: response.output_item.done\n"
                    yield f"data: {json.dumps(response_output_item_done)}\n\n"

                fc_item_id = uuid.uuid4().hex[:50]
                call_item_id = uuid.uuid4().hex[:24]

                if has_reasoning:
                    response_output_item_added = event_builder.build_response_output_item_added_event(
                        item_id=f"fc_{fc_item_id}",
                        item_type="function_call",
                        call_id=f"call_{call_item_id}",
                        name="exec_command",
                        sequence_number=sequence_number
                    )
                    sequence_number += 1
                    yield "event: response.output_item.added\n"
                    yield f"data: {json.dumps(response_output_item_added)}\n\n"
                else:
                    # 🔥 EVENTO CANÓNICO: response.content_part.added (después de output_item.added)
                    # Algunos clientes esperan este evento antes de comenzar a recibir deltas.
                    response_content_part_added = event_builder.build_response_content_part_added_event(
                        item_id=output_item_id,
                        part_type="output_text",
                        text="",
                        index=0,
                        sequence_number=sequence_number
                    )
                    sequence_number += 1
                    yield "event: response.content_part.added\n"
                    yield f"data: {json.dumps(response_content_part_added)}\n\n"

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
                                    response_function_call_started = event_builder.build_response_function_call_started_event(
                                        item_id=output_item_id,
                                        call_id=tc.get("id", f"call_{request_id}"),
                                        name=function_name,
                                        sequence_number=sequence_number
                                    )
                                    sequence_number += 1
                                    yield "event: response.function_call.started\n"
                                    yield f"data: {json.dumps(response_function_call_started)}\n\n"

                                # Emitir argumentos en chunks pequeños
                                chunk_size = 20
                                for i in range(0, len(function_arguments), chunk_size):
                                    arg_chunk = function_arguments[i:i + chunk_size]
                                    function_arguments_buffer += arg_chunk

                                    response_function_call_arguments_delta = event_builder.build_response_function_call_arguments_delta_event(
                                        item_id=output_item_id,
                                        call_id=tc.get("id", f"call_{request_id}"),
                                        delta=arg_chunk,
                                        sequence_number=sequence_number
                                    )
                                    sequence_number += 1
                                    yield "event: response.function_call.arguments.delta\n"
                                    yield f"data: {json.dumps(response_function_call_arguments_delta)}\n\n"

                                if has_reasoning:
                                    response_function_call_arguments_done = event_builder.build_response_function_call_arguments_done_event(
                                        arguments=function_arguments,
                                        item_id=f"fc_{fc_item_id}",
                                        output_index=1,
                                        sequence_number=sequence_number
                                    )
                                    sequence_number += 1
                                    yield "event: response.function_call_arguments.done\n"
                                    yield f"data: {json.dumps(response_function_call_arguments_done)}\n\n"
                                else:
                                    # Emitir evento function_call.completed
                                    response_function_call_completed = event_builder.build_response_function_call_completed_event(
                                        item_id=output_item_id,
                                        call_id=tc.get("id", f"call_{request_id}"),
                                        name=function_name,
                                        arguments=function_arguments,
                                        sequence_number=sequence_number
                                    )
                                    sequence_number += 1
                                    yield "event: response.function_call.completed\n"
                                    yield f"data: {json.dumps(response_function_call_completed)}\n\n"

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
                                response_function_call_arguments_delta = event_builder.build_response_function_call_arguments_delta_event(
                                    item_id=f"fc_{fc_item_id}",
                                    delta=chunk_text,
                                    sequence_number=sequence_number
                                )
                                sequence_number += 1
                                yield f"event: response.function_call_arguments.delta\n"
                                yield f"data: {json.dumps(response_function_call_arguments_delta)}\n\n"
                            else:
                                response_output_text_delta = event_builder.build_response_output_text_delta_event(
                                    item_id=output_item_id,
                                    delta=chunk_text,
                                    sequence_number=sequence_number
                                )
                                sequence_number += 1
                                yield f"event: response.output_text.delta\n"
                                yield f"data: {json.dumps(response_output_text_delta)}\n\n"

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
                        response_function_call_arguments_done = event_builder.build_response_function_call_arguments_done_event(
                            arguments=total_text,
                            item_id=f"fc_{fc_item_id}",
                            output_index=1,
                            sequence_number=sequence_number
                        )
                        sequence_number += 1
                        yield "event: response.function_call_arguments.done\n"
                        yield f"data: {json.dumps(response_function_call_arguments_done)}\n\n"
                    else:
                        response_output_text_done = event_builder.build_response_output_text_done_event(
                            item_id=output_item_id,
                            text=total_text,
                            sequence_number=sequence_number
                        )
                        sequence_number += 1
                        yield "event: response.output_text.done\n"
                        yield f"data: {json.dumps(response_output_text_done)}\n\n"

                        # 🔥 EVENTO CANÓNICO: response.content_part.done
                        response_content_part_done = event_builder.build_response_content_part_done_event(
                            item_id=output_item_id,
                            text=total_text,
                            index=0,
                            sequence_number=sequence_number
                        )
                        sequence_number += 1
                        yield "event: response.content_part.done\n"
                        yield f"data: {json.dumps(response_content_part_done)}\n\n"

                if has_reasoning:
                    # 🔥 EVENTO: output_item.done
                    response_output_item_done = event_builder.build_response_output_item_done_event(
                        item_id=f"fc_{fc_item_id}",
                        item_type="function_call",
                        output_index=0,
                        sequence_number=sequence_number,
                        function_call_data={
                            "arguments": "{\"cmd\":\"ls\",\"yield_time_ms\":1000,\"max_output_tokens\":6000}",
                            "call_id": f"call_{call_item_id}",
                            "name": "exec_command",
                            "status": "completed"
                        }
                    )
                    sequence_number += 1
                    yield "event: response.output_item.done\n"
                    yield f"data: {json.dumps(response_output_item_done)}\n\n"

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

                    response_completed = event_builder.build_response_completed_event(
                        response_id=response_id,
                        created_at=created_at,
                        completed_at=completed_at,
                        kwargs=kwargs,
                        model=model,
                        output=output,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        sequence_number=sequence_number
                    )
                    sequence_number += 1
                    yield "event: response.completed\n"
                    yield f"data: {json.dumps(response_completed)}\n\n"
                else:
                    # 🔥 EVENTO: output_item.done
                    # Construir function_call_data si hubo function_call
                    function_call_data = None
                    if function_call_emitted and function_name:
                        function_call_data = {
                            "call_id": f"call_{request_id}",
                            "name": function_name,
                            "arguments": function_arguments_buffer,
                            "status": "completed"
                        }

                    response_output_item_done = event_builder.build_response_output_item_done_event(
                        item_id=output_item_id,
                        item_type="message",
                        text=total_text,
                        output_index=0,
                        sequence_number=sequence_number,
                        function_call_data=function_call_data
                    )
                    sequence_number += 1
                    yield "event: response.output_item.done\n"
                    yield f"data: {json.dumps(response_output_item_done)}\n\n"

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

                    response_completed = event_builder.build_response_completed_event(
                        response_id=response_id,
                        created_at=created_at,
                        completed_at=completed_at,
                        kwargs=kwargs,
                        model=model,
                        output=output,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        sequence_number=sequence_number
                    )
                    sequence_number += 1
                    yield "event: response.completed\n"
                    yield f"data: {json.dumps(response_completed)}\n\n"

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
responses_handler = ResponsesHandler()
"""Instancia global del handler de responses."""