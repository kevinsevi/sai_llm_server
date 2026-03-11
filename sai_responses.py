# sai_responses.py

import time
import json
import uuid

from sai_handler import sai_llm, logger, VERBOSE_LOGGING
from sai_builder import event_builder
from starlette.responses import Response, StreamingResponse
from typing import AsyncIterator
from sai_system_prompt import merge_system_prompts_for_responses
from sai_exceptions import (
    SAIAPIError,
    SAIAuthenticationError,
    SAIRateLimitError,
    SAIPromptTooLongError,
)

class ResponsesHandler:
    """Handler para el endpoint /v1/responses con soporte streaming y no streaming."""

    def _process_messages(self, messages: list) -> list:
        """
        Procesa mensajes y concatena system prompt adicional.
        
        Args:
            messages: Lista de mensajes originales.
        
        Returns:
            Lista de mensajes procesados con system prompt combinado.
        """
        processed = []
        
        for msg in messages:
            if msg.get("role") == "system":
                # Concatenar system prompt adicional
                original_content = msg.get("content", "")
                merged_content = merge_system_prompts_for_responses(original_content)
                processed.append({
                    "role": "system",
                    "content": merged_content
                })
            else:
                processed.append(msg)
        
        return processed

    async def responses_non_streaming(self, request_id: str, output_item_id: str,
                                      messages: list, kwargs: dict, model: str):
        # Procesar mensajes con system prompt adicional
        messages = self._process_messages(messages)
        
        logger.info(f"🚀 [{request_id}] Llamando a sai_llm.acompletion()...")

        # Timestamp de inicio
        created_at = int(time.time())

        litellm_response = await sai_llm.acompletion(request_id, messages=messages, **kwargs)

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
        # Procesar mensajes con system prompt adicional
        messages = self._process_messages(messages)
        
        # Generar stream de eventos SSE
        async def event_generator() -> AsyncIterator[str]:
            try:
                # 🔧 Enriquecer tools: agregar apply_patch dentro de kwargs["tools"]
                # (sin pisar los tools existentes y evitando duplicados)
                apply_patch_tool = {
                    "description": "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.",
                    "format": {
                        "definition": "start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF\nend_patch: \"*** End Patch\" LF?\n\nhunk: add_hunk | delete_hunk | update_hunk\nadd_hunk: \"*** Add File: \" filename LF add_line+\ndelete_hunk: \"*** Delete File: \" filename LF\nupdate_hunk: \"*** Update File: \" filename LF change_move? change?\n\nfilename: /(.+)/\nadd_line: \"+\" /(.*)/ LF -> line\n\nchange_move: \"*** Move to: \" filename LF\nchange: (change_context | change_line)+ eof_line?\nchange_context: (\"@@\" | \"@@ \" /(.+)/) LF\nchange_line: (\"+\" | \"-\" | \" \") /(.*)/ LF\neof_line: \"*** End of File\" LF\n\n%import common.LF\n",
                        "syntax": "lark",
                        "type": "grammar"
                    },
                    "name": "apply_patch",
                    "type": "custom"
                }

                tools = kwargs.get("tools")
                if tools is None:
                    tools = []
                    kwargs["tools"] = tools
                elif not isinstance(tools, list):
                    logger.warning(
                        f"⚠️ [{request_id}] kwargs['tools'] no es list (es {type(tools).__name__}). Normalizando a lista."
                    )
                    tools = [tools]
                    kwargs["tools"] = tools

                # Evitar duplicados: solo agregar si NO existe un tool con name="apply_patch" y type="custom"
                has_apply_patch_custom = False
                for t in tools:
                    if isinstance(t, dict) and t.get("name") == "apply_patch" and t.get("type") == "custom":
                        has_apply_patch_custom = True
                        break

                if not has_apply_patch_custom:
                    tools.append(apply_patch_tool)
                    logger.info(f"🧩 [{request_id}] Tool 'apply_patch' agregado a kwargs['tools'] (total={len(tools)})")

                # 🔢 Inicializar contador de secuencia
                sequence_number = 0

                try:
                    # Almacenar el generador en una variable antes de iterar
                    stream_generator = sai_llm.astreaming(request_id, messages=messages, **kwargs)

                    # Variables para tracking
                    chunk_count = 0
                    tool_use = None
                    # Timestamp de creación
                    created_at = int(time.time())
                    fc_item_id = uuid.uuid4().hex[:50]
                    call_item_id = request_id
                    response_id = f"resp_{uuid.uuid4().hex[:50]}"
                    # Variables para function_call streaming
                    function_call_emitted = False
                    function_name = None
                    total_text = ""
                    input_tokens = 0
                    output_tokens = 0
                    finish_reason = "end_turn"

                    function_arguments_buffer = ""
                    # 🆕 Variables para custom_tool_call
                    is_custom_tool = False
                    custom_tool_input_buffer = ""
                    item_type = None
                    # 🆕 Track de tool items emitidos (para response.completed)
                    emitted_tool_items = []

                    async for chunk in stream_generator:
                        chunk_count += 1

                        # Extraer tool_use del chunk
                        tool_use = chunk.get('tool_use') if isinstance(chunk, dict) else getattr(chunk, 'tool_use', None)

                        if chunk_count == 1:
                            logger.warning(f"⏱️ [{request_id}] PRIMER CHUNK recibido")

                            logger.info(f"🌊 [{request_id}] Iniciando streaming SSE...")

                            # 🔥 EVENTO CANÓNICO: response.created
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

                            logger.info(f"🚀 [{request_id}] Llamando a sai_llm.astreaming()...")

                            # 🔥 EVENTO CANÓNICO: response.output_item.added (reasoning/message)
                            item_type = "reasoning" if tool_use else "message"
                            rs_item_id = f"rs_{uuid.uuid4().hex[:50]}" if tool_use else f"msg_{uuid.uuid4().hex[:50]}"

                            response_output_item_added = event_builder.build_response_output_item_added_event(
                                item_id=rs_item_id if tool_use else f"msg_{uuid.uuid4().hex[:50]}",
                                item_type="reasoning" if tool_use else "message",
                                sequence_number=sequence_number
                            )
                            sequence_number += 1
                            yield "event: response.output_item.added\n"
                            yield f"data: {json.dumps(response_output_item_added)}\n\n"

                            if tool_use:
                                response_output_item_done = event_builder.build_response_output_item_done_event(
                                    item_id=rs_item_id,
                                    item_type=item_type,
                                    text="",
                                    sequence_number=sequence_number
                                )
                                sequence_number += 1
                                yield "event: response.output_item.done\n"
                                yield f"data: {json.dumps(response_output_item_done)}\n\n"
                            else:
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

                        # 🔧 Soporte para tool_calls (1 o múltiples) en formato Responses API
                        if tool_use and not function_call_emitted:
                            logger.info(
                                f"🔧 [{request_id}] [TOOL_CALLS] Detectados tool_calls | Count: {len(tool_use)}"
                            )

                            for idx, tc in enumerate(tool_use):
                                if not isinstance(tc, dict):
                                    logger.warning(
                                        f"⚠️ [{request_id}] tool_use[{idx}] no es dict, es {type(tc).__name__}. Saltando."
                                    )
                                    continue

                                tc_type = tc.get("type")
                                name = tc.get("function", {}).get("name")
                                call_id = tc.get("id", f"call_{request_id}_{idx}")

                                args_raw = tc.get("function", {}).get("arguments", "{}")
                                args = json.dumps(args_raw, ensure_ascii=False) if isinstance(args_raw, dict) else (args_raw or "")

                                is_custom = (name == "apply_patch")
                                fc_id = uuid.uuid4().hex[:50]
                                item_id = f"fc_{fc_id}"

                                # output_item.added (uno por tool)
                                response_output_item_added = event_builder.build_response_output_item_added_event(
                                    item_id=item_id,
                                    item_type=tc_type,
                                    call_id=call_id,
                                    name=name,
                                    sequence_number=sequence_number
                                )
                                sequence_number += 1
                                yield "event: response.output_item.added\n"
                                yield f"data: {json.dumps(response_output_item_added)}\n\n"

                                # deltas
                                buf = ""
                                chunk_size = 20
                                for i in range(0, len(args), chunk_size):
                                    part = args[i:i + chunk_size]
                                    buf += part

                                    if is_custom:
                                        response_delta = event_builder.build_response_custom_tool_call_input_delta_event(
                                            item_id=item_id,
                                            delta=part,
                                            output_index=0,
                                            sequence_number=sequence_number
                                        )
                                        sequence_number += 1
                                        yield "event: response.custom_tool_call_input.delta\n"
                                        yield f"data: {json.dumps(response_delta)}\n\n"
                                    else:
                                        response_delta = event_builder.build_response_function_call_arguments_delta_event(
                                            item_id=item_id,
                                            call_id=call_id,
                                            delta=part,
                                            sequence_number=sequence_number
                                        )
                                        sequence_number += 1
                                        yield "event: response.function_call_arguments.delta\n"
                                        yield f"data: {json.dumps(response_delta)}\n\n"

                                # done
                                if is_custom:
                                    response_done = event_builder.build_response_custom_tool_call_input_done_event(
                                        item_id=item_id,
                                        input_text=buf,
                                        output_index=0,
                                        sequence_number=sequence_number
                                    )
                                    sequence_number += 1
                                    yield "event: response.custom_tool_call_input.done\n"
                                    yield f"data: {json.dumps(response_done)}\n\n"
                                else:
                                    response_done = event_builder.build_response_function_call_arguments_done_event(
                                        arguments=buf,
                                        item_id=item_id,
                                        output_index=0,
                                        sequence_number=sequence_number
                                    )
                                    sequence_number += 1
                                    yield "event: response.function_call_arguments.done\n"
                                    yield f"data: {json.dumps(response_done)}\n\n"

                                # output_item.done
                                if is_custom:
                                    response_output_item_done = event_builder.build_response_output_item_done_event(
                                        item_id=item_id,
                                        item_type=tc_type,
                                        output_index=0,
                                        sequence_number=sequence_number,
                                        custom_tool_call_data={
                                            "input": buf,
                                            "call_id": call_id,
                                            "name": name,
                                            "status": "completed"
                                        }
                                    )
                                else:
                                    response_output_item_done = event_builder.build_response_output_item_done_event(
                                        item_id=item_id,
                                        item_type=tc_type,
                                        output_index=0,
                                        sequence_number=sequence_number,
                                        function_call_data={
                                            "arguments": buf,
                                            "call_id": call_id,
                                            "name": name,
                                            "status": "completed"
                                        }
                                    )
                                sequence_number += 1
                                yield "event: response.output_item.done\n"
                                yield f"data: {json.dumps(response_output_item_done)}\n\n"

                                emitted_tool_items.append({
                                    "type": tc_type,
                                    "id": item_id,
                                    "call_id": call_id,
                                    "name": name,
                                    "is_custom": is_custom,
                                    "arguments": None if is_custom else buf,
                                    "input": buf if is_custom else None,
                                })

                            function_call_emitted = True
                            finish_reason = "tool_calls"
                            continue

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

                            response_output_text_delta = event_builder.build_response_output_text_delta_event(
                                item_id=output_item_id,
                                delta=chunk_text,
                                sequence_number=sequence_number
                            )
                            sequence_number += 1
                            yield "event: response.output_text.delta\n"
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

                # 🛡️ FALLBACK: si el modelo mezcló texto + JSON de tool_calls,
                # extraer los tool_calls del texto acumulado y emitir eventos estructurados.
                if total_text and not tool_use and not function_call_emitted:
                    fallback_tool_calls, cleaned_total = sai_llm._extract_tool_calls_from_plain_text_end(
                        total_text, request_id
                    )
                    if fallback_tool_calls:
                        logger.info(
                            f"🛡️ [{request_id}] [FALLBACK] Tool calls detectados en total_text | "
                            f"Count: {len(fallback_tool_calls)} | "
                            f"Texto previo descartado: {len(total_text) - len(cleaned_total)} chars"
                        )
                        tool_use = fallback_tool_calls
                        total_text = cleaned_total
                        finish_reason = "tool_calls"
                        function_call_emitted = True

                        for idx, tc in enumerate(fallback_tool_calls):
                            if not isinstance(tc, dict):
                                continue

                            tc_type = tc.get("type", "function_call")
                            name = tc.get("function", {}).get("name") or tc.get("name", "")
                            call_id = tc.get("id", f"call_{request_id}_{idx}")
                            args_raw = tc.get("function", {}).get("arguments", "{}") if tc.get("function") else tc.get("arguments",
                                                                                                                   "{}")
                            args = json.dumps(args_raw, ensure_ascii=False) if isinstance(args_raw, dict) else (args_raw or "")

                            is_custom = (name == "apply_patch")
                            fc_id = uuid.uuid4().hex[:50]
                            item_id = f"fc_{fc_id}"

                            response_output_item_added = event_builder.build_response_output_item_added_event(
                                item_id=item_id,
                                item_type=tc_type,
                                call_id=call_id,
                                name=name,
                                sequence_number=sequence_number
                            )
                            sequence_number += 1
                            yield "event: response.output_item.added\n"
                            yield f"data: {json.dumps(response_output_item_added)}\n\n"

                            chunk_size = 20
                            buf = ""
                            for i in range(0, len(args), chunk_size):
                                part = args[i:i + chunk_size]
                                buf += part
                                if is_custom:
                                    delta_event = event_builder.build_response_custom_tool_call_input_delta_event(
                                        item_id=item_id,
                                        delta=part,
                                        output_index=0,
                                        sequence_number=sequence_number
                                    )
                                    sequence_number += 1
                                    yield "event: response.custom_tool_call_input.delta\n"
                                    yield f"data: {json.dumps(delta_event)}\n\n"
                                else:
                                    delta_event = event_builder.build_response_function_call_arguments_delta_event(
                                        item_id=item_id,
                                        call_id=call_id,
                                        delta=part,
                                        sequence_number=sequence_number
                                    )
                                    sequence_number += 1
                                    yield "event: response.function_call_arguments.delta\n"
                                    yield f"data: {json.dumps(delta_event)}\n\n"

                            if is_custom:
                                done_event = event_builder.build_response_custom_tool_call_input_done_event(
                                    item_id=item_id,
                                    input_text=buf,
                                    output_index=0,
                                    sequence_number=sequence_number
                                )
                                sequence_number += 1
                                yield "event: response.custom_tool_call_input.done\n"
                                yield f"data: {json.dumps(done_event)}\n\n"
                            else:
                                done_event = event_builder.build_response_function_call_arguments_done_event(
                                    arguments=buf,
                                    item_id=item_id,
                                    output_index=0,
                                    sequence_number=sequence_number
                                )
                                sequence_number += 1
                                yield "event: response.function_call_arguments.done\n"
                                yield f"data: {json.dumps(done_event)}\n\n"

                            if is_custom:
                                item_done_event = event_builder.build_response_output_item_done_event(
                                    item_id=item_id,
                                    item_type=tc_type,
                                    output_index=0,
                                    sequence_number=sequence_number,
                                    custom_tool_call_data={
                                        "input": buf,
                                        "call_id": call_id,
                                        "name": name,
                                        "status": "completed"
                                    }
                                )
                            else:
                                item_done_event = event_builder.build_response_output_item_done_event(
                                    item_id=item_id,
                                    item_type=tc_type,
                                    output_index=0,
                                    sequence_number=sequence_number,
                                    function_call_data={
                                        "arguments": buf,
                                        "call_id": call_id,
                                        "name": name,
                                        "status": "completed"
                                    }
                                )
                            sequence_number += 1
                            yield "event: response.output_item.done\n"
                            yield f"data: {json.dumps(item_done_event)}\n\n"

                            emitted_tool_items.append({
                                "type": tc_type,
                                "id": item_id,
                                "call_id": call_id,
                                "name": name,
                                "is_custom": is_custom,
                                "arguments": None if is_custom else buf,
                                "input": buf if is_custom else None,
                            })

                # 🔥 EVENTO CANÓNICO: response.output_text.done (solo si hay texto)
                if total_text:
                    if tool_use:
                        # 🆕 Usar builder apropiado según el tipo de tool
                        if is_custom_tool:
                            response_custom_input_done = event_builder.build_response_custom_tool_call_input_done_event(
                                item_id=f"fc_{fc_item_id}",
                                input_text=custom_tool_input_buffer,
                                output_index=1,
                                sequence_number=sequence_number
                            )
                            sequence_number += 1
                            yield "event: response.custom_tool_call_input.done\n"
                            yield f"data: {json.dumps(response_custom_input_done)}\n\n"
                        else:
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

                if tool_use:
                    # ✅ Los tool items ya fueron emitidos y cerrados arriba.
                    completed_at = int(time.time())

                    output = []
                    for it in emitted_tool_items:
                        if it["is_custom"]:
                            output.append({
                                "type": it["type"],
                                "id": it["id"],
                                "call_id": it["call_id"],
                                "name": it["name"],
                                "input": it["input"] or "",
                                "status": "completed"
                            })
                        else:
                            output.append({
                                "type": it["type"],
                                "id": it["id"],
                                "call_id": it["call_id"],
                                "name": it["name"],
                                "arguments": it["arguments"] or "",
                                "status": "completed"
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
                            "type": item_type,
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

            except SAIAuthenticationError as e:
                logger.error(f"🔐 [{request_id}] Autenticación fallida en streaming responses: {e}")
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": str(e)
                    }
                }
                yield f"event: error\n"
                yield f"data: {json.dumps(error_event)}\n\n"
            except SAIRateLimitError as e:
                logger.error(f"⚠️ [{request_id}] Rate limit en streaming responses: {e}")
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": str(e)
                    }
                }
                yield f"event: error\n"
                yield f"data: {json.dumps(error_event)}\n\n"
            except SAIPromptTooLongError as e:
                logger.error(f"⚠️ [{request_id}] Prompt demasiado largo en streaming responses: {e}")
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "prompt_too_long",
                        "message": str(e)
                    }
                }
                yield f"event: error\n"
                yield f"data: {json.dumps(error_event)}\n\n"
            except SAIAPIError as e:
                logger.error(f"❌ [{request_id}] Error SAI en streaming responses: {e}")
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "sai_error",
                        "message": str(e)
                    }
                }
                yield f"event: error\n"
                yield f"data: {json.dumps(error_event)}\n\n"
            except Exception as e:
                logger.error(f"❌ [{request_id}] Error en streaming: {type(e).__name__}: {str(e)}")
                # Enviar evento de error genérico
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