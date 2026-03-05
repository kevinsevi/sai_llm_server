# sai_completions.py

import time
import json
from starlette.responses import Response, StreamingResponse
from sai_handler import sai_llm, logger
from sai_exceptions import (
    SAIAPIError,
    SAIAuthenticationError,
    SAIRateLimitError,
    SAIPromptTooLongError,
)

class CompletionsHandler:
    """Handler para endpoints de Completions (legacy OpenAI format)"""

    async def completions_non_streaming(self, request_id: str, messages: list, kwargs: dict, model: str):
        """Maneja requests de completions sin streaming"""
        logger.info(f"🚀 [{request_id}] Llamando a sai_llm.acompletion()...")
        litellm_response = await sai_llm.acompletion(request_id, messages=messages, **kwargs)

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

    async def completions_streaming(self, request_id: str, messages: list, kwargs: dict, model: str):
        """Maneja requests de completions con streaming"""
        async def event_generator():
            try:
                logger.info(f"🌊 [{request_id}] Iniciando streaming...")

                # Llamar a SAI streaming (reutiliza sai_handler.py)
                logger.info(f"🚀 [{request_id}] Llamando a sai_llm.astreaming()...")

                chunk_count = 0

                async for chunk in sai_llm.astreaming(request_id, messages=messages, **kwargs):
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

            except SAIAuthenticationError as e:
                logger.error(f"🔐 [{request_id}] Autenticación fallida en streaming completions: {e}")
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "authentication_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            except SAIRateLimitError as e:
                logger.error(f"⚠️ [{request_id}] Rate limit en streaming completions: {e}")
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "rate_limit_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            except SAIPromptTooLongError as e:
                logger.error(f"⚠️ [{request_id}] Prompt demasiado largo en streaming completions: {e}")
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "prompt_too_long"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            except SAIAPIError as e:
                logger.error(f"❌ [{request_id}] Error SAI en streaming completions: {e}")
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "sai_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
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
completions_handler = CompletionsHandler()
"""Instancia global del handler de Completions."""