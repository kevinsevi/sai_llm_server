"""
sai_handler.py - Handler principal para integración con SAI (Semantic AI)

Este módulo proporciona una implementación de CustomLLM de LiteLLM que actúa como
puente entre clientes OpenAI-compatible y la API de SAI. Incluye:

- Autenticación flexible (API Key o Cookie)
- Conversión de formatos OpenAI ↔ SAI
- Soporte para streaming y non-streaming
- Manejo robusto de errores y reintentos
- Logging detallado con modo verbose opcional
- Compatibilidad con múltiples clientes (GitKraken, Codex CLI, etc.)

Variables de entorno requeridas:
    SAI_TEMPLATE_ID: ID del template de SAI a utilizar
    SAI_URL: URL base del servidor SAI
    SAI_KEY o SAI_COOKIE: Credenciales de autenticación (al menos una)

Variables de entorno opcionales:
    REQUEST_TIMEOUT: Timeout en segundos (default: 600)
    MAX_RETRIES: Número máximo de reintentos (default: 3)
    VERBOSE_LOGGING: Activar logs detallados (default: false)
"""

# sai_handler.py
import json
import asyncio
import requests
import os
import time
import uuid
import logging
from typing import AsyncIterator, Optional
from dotenv import load_dotenv
from litellm import CustomLLM, ModelResponse
from litellm.types.utils import GenericStreamingChunk
from logging.handlers import RotatingFileHandler
from fastapi.responses import StreamingResponse, JSONResponse, Response

from sai_models import is_valid_model  # Importar validación de modelos

# Cargar variables de entorno
load_dotenv()

# ---------------- Logging ----------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("sai_handler")

file_handler = RotatingFileHandler(
    filename=os.path.join(log_dir, "sai_handler.log"),
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

SAI_TEMPLATE_ID = os.getenv("SAI_TEMPLATE_ID")
SAI_URL = os.getenv("SAI_URL")

def get_secret(env_var):
    value = os.getenv(env_var)
    if value and value.startswith('/run/secrets/'):
        with open(value, 'r') as f:
            return f.read().strip()
    return value

SAI_COOKIE = get_secret('SAI_COOKIE')
SAI_KEY = get_secret("SAI_KEY")

# Validar variables de entorno críticas
if not SAI_TEMPLATE_ID:
    error_msg = "SAI_TEMPLATE_ID no está configurado en las variables de entorno"
    logger.critical(f"❌ INICIALIZACIÓN FALLIDA: {error_msg}")
    raise ValueError(error_msg)
if not SAI_KEY and not SAI_COOKIE:
    error_msg = "Debe configurar al menos SAI_KEY o SAI_COOKIE en las variables de entorno"
    logger.critical(f"❌ INICIALIZACIÓN FALLIDA: {error_msg}")
    raise ValueError(error_msg)

CHUNK_SIZE = 50  # caracteres por chunk
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))  # segundos
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"

# Cambia según la variable de entorno VERBOSE_LOGGING
if VERBOSE_LOGGING:
    logger.setLevel(logging.DEBUG)
    logger.info("🔍 VERBOSE_LOGGING activado - Se mostrarán logs detallados de DEBUG")
else:
    logger.setLevel(logging.INFO)
    logger.info("📊 Logging en modo INFO - Use VERBOSE_LOGGING=true para logs detallados")

# Log de configuración inicial
logger.info(
    f"⚙️ Configuración cargada | "
    f"Template: {SAI_TEMPLATE_ID} | "
    f"URL: {SAI_URL} | "
    f"Timeout: {REQUEST_TIMEOUT}s | "
    f"Max Retries: {MAX_RETRIES} | "
    f"Chunk Size: {CHUNK_SIZE} chars | "
    f"Auth disponible: API_KEY={'✓' if SAI_KEY else '✗'}, COOKIE={'✓' if SAI_COOKIE else '✗'}"
)

# Configurar sesión HTTP reutilizable con pool optimizado
http_session = requests.Session()
http_session.timeout = REQUEST_TIMEOUT
adapter = requests.adapters.HTTPAdapter(
    max_retries=MAX_RETRIES,
    pool_connections=10,  # Mantener más conexiones en pool
    pool_maxsize=20       # Tamaño máximo del pool
)
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)


# ---------------- Excepciones personalizadas ----------------
class SAIAPIError(Exception):
    """
    Error base para todas las excepciones relacionadas con la API de SAI.
    
    Utilizada como clase padre para errores específicos de SAI.
    """
    pass


class SAIRateLimitError(SAIAPIError):
    """
    Error cuando se excede el límite de uso del template o API.
    
    Se lanza cuando SAI retorna HTTP 429 indicando que se ha alcanzado
    el límite de requests permitidos.
    """
    pass


class SAIAuthenticationError(SAIAPIError):
    """
    Error de autenticación con la API de SAI.
    
    Se lanza cuando las credenciales (API Key o Cookie) son inválidas
    o han expirado (HTTP 401).
    """
    pass


# ---------------- Conversor de formatos OpenAI ----------------
class OpenAiSAIConverter:
    """
    Conversor bidireccional entre formatos de API OpenAI y SAI.
    
    Proporciona métodos para:
    - Convertir requests de OpenAI Messages API a formato LiteLLM
    - Convertir responses de LiteLLM a formato OpenAI
    - Manejar múltiples formatos de API (Completions, Chat Completions, Responses)
    - Procesar estructuras anidadas complejas (ej: Codex CLI)
    
    Soporta tres tipos de endpoints:
    1. /v1/completions - Completions API (legacy)
    2. /v1/chat/completions - Chat Completions API
    3. /v1/responses - Responses API (formato Codex CLI)
    """

    @staticmethod
    def _extract_text_recursive(content) -> str:
        """
        Extrae texto de estructuras anidadas recursivamente.

        Maneja múltiples formatos de contenido:
        - Strings simples: "hola"
        - Content blocks: [{"type": "text", "text": "hola"}]
        - Estructuras anidadas: [{"type": "message", "content": [...]}]
        - Formato Codex CLI: {"input_text": "..."}

        Args:
            content: Puede ser str, list, dict o cualquier combinación anidada

        Returns:
            str: Texto extraído y concatenado. Retorna "" si no hay texto válido.
            
        Examples:
            >>> _extract_text_recursive("hola")
            "hola"
            >>> _extract_text_recursive([{"type": "text", "text": "hola"}])
            "hola"
            >>> _extract_text_recursive({"input_text": "test"})
            "test"
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

        Procesa:
        - System prompt (si existe)
        - Lista de mensajes con roles y contenido
        - Parámetros de generación (temperature, max_tokens, etc.)
        - Tools/funciones disponibles
        - Metadata adicional

        Args:
            openai_request: Request en formato OpenAI con estructura:
                {
                    "system": str (opcional),
                    "messages": [{"role": str, "content": str/list/dict}],
                    "model": str,
                    "temperature": float (opcional),
                    "max_tokens": int (opcional),
                    "tools": list (opcional),
                    ...
                }

        Returns:
            tuple: (messages, kwargs_for_litellm)
                - messages: Lista de mensajes procesados
                - kwargs: Diccionario con parámetros para LiteLLM
                
        Example:
            >>> messages, kwargs = openai_to_litellm({
            ...     "messages": [{"role": "user", "content": "Hello"}],
            ...     "model": "claude-sonnet-4-5-20250929"
            ... })
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
        """
        Convierte respuesta de LiteLLM a formato OpenAI Messages API.

        Extrae:
        - Texto de la respuesta
        - Información de uso (tokens)
        - Razón de finalización (finish_reason)

        Args:
            litellm_response: Respuesta de sai_llm.acompletion()
            model: Nombre del modelo utilizado
            request_id: ID único de la solicitud para tracking

        Returns:
            dict: Respuesta en formato OpenAI Messages API:
                {
                    "id": "msg_...",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "..."}],
                    "model": str,
                    "stop_reason": str,
                    "usage": {"input_tokens": int, "output_tokens": int}
                }
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

    async def _completions_non_streaming(self, request_id: str, messages: list, kwargs: dict, model: str):
        """
        Maneja requests a /v1/completions sin streaming.
        
        Formato de respuesta compatible con OpenAI Completions API (legacy).

        Args:
            request_id: ID único de la solicitud
            messages: Lista de mensajes procesados
            kwargs: Parámetros adicionales para LiteLLM
            model: Nombre del modelo

        Returns:
            JSONResponse: Respuesta en formato OpenAI Completions
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
        """
        Maneja requests a /v1/completions con streaming SSE.
        
        Emite eventos Server-Sent Events con chunks de texto.

        Args:
            request_id: ID único de la solicitud
            messages: Lista de mensajes procesados
            kwargs: Parámetros adicionales para LiteLLM
            model: Nombre del modelo

        Returns:
            StreamingResponse: Stream de eventos SSE con formato:
                data: {"id": "cmpl-...", "choices": [{"text": "...", ...}]}
                data: [DONE]
        """
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
        """
        Maneja requests a /v1/chat/completions sin streaming.

        Formato de respuesta compatible con OpenAI Chat Completions API.

        Args:
            request_id: ID único de la solicitud
            messages: Lista de mensajes procesados
            kwargs: Parámetros adicionales para LiteLLM
            model: Nombre del modelo

        Returns:
            JSONResponse: Respuesta en formato OpenAI Chat Completions
        """
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
        """
        Maneja requests a /v1/chat/completions con streaming SSE.
        
        Emite eventos Server-Sent Events con deltas de contenido.

        Args:
            request_id: ID único de la solicitud
            messages: Lista de mensajes procesados
            kwargs: Parámetros adicionales para LiteLLM
            model: Nombre del modelo

        Returns:
            StreamingResponse: Stream de eventos SSE con formato:
                data: {"id": "chatcmpl-...", "choices": [{"delta": {"content": "..."}, ...}]}
                data: [DONE]
        """
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

    async def _responses_non_streaming(self, request_id: str, response_id: str, output_item_id: str,
                                       messages: list, kwargs: dict, model: str):
        """
        Maneja requests a /v1/responses sin streaming.

        Compatible con OpenAI Responses API y Codex CLI.
        """
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
                "effort": None,
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

    async def _responses_streaming(self, request_id: str, response_id: str, output_item_id: str,
                                   messages: list, kwargs: dict, model: str):
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
                            "effort": None,
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
                            "effort": None,
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
                completed_event = {
                    "type": "response.completed",
                    "response_id": response_id
                }
                yield "event: response.completed\n"
                yield f"data: {json.dumps(completed_event)}\n\n"

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

class SAILLM(CustomLLM):
    """
    Implementación de CustomLLM para integración con SAI.
    
    Proporciona métodos síncronos y asíncronos para:
    - Completions (completion, acompletion)
    - Streaming (astreaming)
    
    Características:
    - Autenticación flexible (API Key o Cookie)
    - Extracción de credenciales desde múltiples fuentes
    - Detección de user-agent para compatibilidad con clientes específicos
    - Procesamiento de mensajes envueltos por plugins de IDE
    - Manejo robusto de errores con reintentos automáticos
    - Logging detallado con modo verbose
    
    Attributes:
        Hereda de CustomLLM de LiteLLM
    """

    def __init__(self):
        """Inicializa la instancia de SAILLM."""
        super().__init__()

    def _extract_from_litellm_params(self, kwargs: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Extrae la API key desde litellm_params['metadata']['user_api_key'].

        Returns:
            tuple[Optional[str], Optional[str]]: (api_key, source)
        """
        litellm_params = kwargs.get('litellm_params', {})
        if not isinstance(litellm_params, dict):
            return None, None

        metadata = litellm_params.get('metadata', {})
        if not isinstance(metadata, dict):
            return None, None

        user_api_key = metadata.get('user_api_key', '')
        if user_api_key:
            return user_api_key, "litellm_params.metadata"

        return None, None

    def _extract_from_headers(self, kwargs: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Extrae la API key desde headers['user_api_key'].

        Returns:
            tuple[Optional[str], Optional[str]]: (api_key, source)
        """
        headers = kwargs.get('headers', {})
        if not isinstance(headers, dict):
            return None, None

        user_api_key = headers.get('user_api_key', '')
        if user_api_key:
            return user_api_key, "headers"

        return None, None

    def _is_valid_api_key(self, api_key: str) -> bool:
        """
        Valida si una API key es válida (no vacía y no es 'raspberry').

        Args:
            api_key: La API key a validar

        Returns:
            bool: True si es válida, False en caso contrario
        """
        if not api_key:
            return False

        trimmed = str(api_key).strip()
        if not trimmed or trimmed.lower() == "raspberry":
            return False

        return True

    def _extract_user_api_key(self, kwargs: dict, request_id: str) -> Optional[str]:
        """
        Extrae y valida la API key del usuario desde kwargs.

        Args:
            kwargs: Diccionario de argumentos que puede contener litellm_params o headers
            request_id: ID de la solicitud para logging

        Returns:
            La API key del usuario si es válida, None en caso contrario
        """
        try:
            # Intentar extraer desde litellm_params (prioridad 1)
            user_api_key, source = self._extract_from_litellm_params(kwargs)

            # Fallback: intentar extraer desde headers (prioridad 2)
            if not user_api_key:
                user_api_key, source = self._extract_from_headers(kwargs)

            # Si no se encontró en ninguna ubicación
            if not user_api_key:
                if VERBOSE_LOGGING:
                    logger.debug(
                        f"[{request_id}] [AUTH] user_api_key NO encontrada | "
                        f"Ubicaciones verificadas: litellm_params.metadata, headers | "
                        f"Resultado: Se usará credencial por defecto del sistema"
                    )
                return None

            # Validar la API key
            if not self._is_valid_api_key(user_api_key):
                if VERBOSE_LOGGING:
                    trimmed = str(user_api_key).strip()
                    reason = "valor vacío" if not trimmed else "valor 'raspberry' (placeholder)"
                    logger.debug(
                        f"[{request_id}] [AUTH] user_api_key RECHAZADA | "
                        f"Fuente: {source} | "
                        f"Razón: {reason} | "
                        f"Resultado: Se usará credencial por defecto del sistema"
                    )
                return None

            # API key válida encontrada
            user_api_key_trimmed = str(user_api_key).strip()
            logger.info(
                f"🔑 [{request_id}] [AUTH] user_api_key ACEPTADA | "
                f"Fuente: {source} | "
                f"Longitud: {len(user_api_key_trimmed)} caracteres | "
                f"Acción: Se usará en lugar de SAI_KEY del sistema"
            )
            return user_api_key_trimmed

        except Exception as e:
            logger.warning(
                f"⚠️ [{request_id}] [AUTH] Excepción al extraer user_api_key | "
                f"Error: {type(e).__name__}: {str(e)} | "
                f"Fallback: Se usará SAI_KEY del sistema"
            )
            return None

    def _extract_user_agent(self, kwargs: dict, request_id: str) -> Optional[str]:
        """
        Extrae el user-agent desde metadata.headers.user-agent.

        Args:
            kwargs: Diccionario de argumentos que puede contener litellm_params
            request_id: ID de la solicitud para logging

        Returns:
            El user-agent si existe, None en caso contrario
        """
        try:
            litellm_params = kwargs.get('litellm_params', {})
            if not isinstance(litellm_params, dict):
                return None

            metadata = litellm_params.get('metadata', {})
            if not isinstance(metadata, dict):
                return None

            headers = metadata.get('headers', {})
            if not isinstance(headers, dict):
                return None

            user_agent = headers.get('user-agent', '')
            if user_agent:
                logger.info(
                    f"🌐 [{request_id}] [USER-AGENT] Detectado | "
                    f"Valor: {user_agent}"
                )
                return user_agent

            return None

        except Exception as e:
            logger.warning(
                f"⚠️ [{request_id}] [USER-AGENT] Excepción al extraer user-agent | "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            return None

    def _extract_plugin_wrapped_message(self, content: str) -> tuple[bool, str]:
        """
        Detecta si el mensaje fue envuelto por el plugin del IDE y extrae el mensaje original.

        Returns:
            tuple[bool, str]: (es_mensaje_plugin, mensaje_original_o_contenido)
        """
        if not isinstance(content, str):
            return False, content

        # Detectar el patrón del plugin
        plugin_prefix = "Determine if the following context is required to solve the task in the user's input in the chat session: \""
        plugin_suffix_start = "\"\nContext:"

        if content.startswith(plugin_prefix) and plugin_suffix_start in content:
            # Extraer el mensaje original entre las comillas
            start_idx = len(plugin_prefix)
            end_idx = content.find(plugin_suffix_start, start_idx)

            if end_idx > start_idx:
                original_message = content[start_idx:end_idx]
                logger.info(
                    f"🔍 [PLUGIN] Mensaje envuelto por IDE detectado | "
                    f"Longitud original: {len(content)} chars | "
                    f"Longitud extraída: {len(original_message)} chars | "
                    f"Preview: {original_message[:80]}{'...' if len(original_message) > 80 else ''}"
                )
                return True, original_message

        return False, content

    def _process_plugin_messages(self, messages: list, request_id: str) -> tuple[bool, int]:
        """
        Procesa mensajes envueltos por el plugin del IDE.

        Returns:
            tuple[bool, int]: (plugin_detected, plugin_count)
        """
        plugin_detected = False
        plugin_count = 0

        for idx, msg in enumerate(messages):
            if isinstance(msg, dict) and "content" in msg:
                is_plugin_msg, original_content = self._extract_plugin_wrapped_message(msg["content"])
                if is_plugin_msg:
                    msg["content"] = original_content
                    plugin_detected = True
                    plugin_count += 1
                    logger.info(
                        f"🔧 [PLUGIN] [{request_id}] Mensaje #{idx} procesado | "
                        f"Tipo: {msg.get('role', 'unknown')} | "
                        f"Contenido extraído: {len(original_content)} chars | "
                        f"Preview: {original_content[:60]}{'...' if len(original_content) > 60 else ''}"
                    )

        return plugin_detected, plugin_count

    def _log_message_statistics(self, messages: list, request_id: str, plugin_detected: bool, plugin_count: int):
        """
        Calcula y registra estadísticas de los mensajes recibidos.
        """
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        roles_count = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            roles_count[role] = roles_count.get(role, 0) + 1

        logger.info(
            f"🔌 [CLIENT → SERVER] [{request_id}] Mensajes recibidos | "
            f"Total: {len(messages)} mensajes | "
            f"Distribución: {', '.join(f'{k}={v}' for k, v in roles_count.items())} | "
            f"Tamaño: {total_chars} caracteres | "
            f"Plugin: {'Sí (' + str(plugin_count) + ' procesados)' if plugin_detected else 'No'}"
        )

        if VERBOSE_LOGGING:
            logger.debug(f"[{request_id}] [VERBOSE] Estructura completa de messages:")
            for idx, msg in enumerate(messages):
                content_preview = str(msg.get("content", ""))[:100]
                logger.debug(
                    f"  [{idx}] role={msg.get('role')} | "
                    f"content_length={len(str(msg.get('content', '')))} | "
                    f"preview={content_preview!r}{'...' if len(str(msg.get('content', ''))) > 100 else ''}"
                )

        return total_chars

    def _validate_message_structure(self, messages: list):
        """
        Valida que cada mensaje tenga la estructura correcta.
        """
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Cada mensaje debe ser un diccionario")
            if "role" not in msg or "content" not in msg:
                raise ValueError("Cada mensaje debe tener 'role' y 'content'")

    def _convert_to_sai_format(self, messages: list) -> list:
        """
        Convierte mensajes al formato esperado por SAI.
        """
        return [{
            "content": msg.get("content", ""),
            "role": msg.get("role"),
            "id": int(time.time() * 1000) + idx
        } for idx, msg in enumerate(messages)]

    def _check_context_size(self, total_chars: int, request_id: str):
        """
        Valida el tamaño del contexto y registra advertencias si es necesario.
        """
        estimated_tokens = total_chars // 4
        max_context_tokens = 128000

        if estimated_tokens > max_context_tokens:
            logger.warning(
                f"⚠️ [SERVER] [{request_id}] Contexto potencialmente demasiado grande | "
                f"Tokens estimados: {estimated_tokens} | "
                f"Máximo recomendado: {max_context_tokens} | "
                f"El cliente debería reducir el historial"
            )

    def _prepare_messages(self, messages, request_id: str):
        if not messages or not isinstance(messages, list):
            raise ValueError("messages debe ser una lista no vacía")

        # Procesar mensajes envueltos por el plugin del IDE
        plugin_detected, plugin_count = self._process_plugin_messages(messages, request_id)

        # Calcular estadísticas y registrar logs
        total_chars = self._log_message_statistics(messages, request_id, plugin_detected, plugin_count)

        # Validar estructura de mensajes
        self._validate_message_structure(messages)

        # Extraer system prompt si existe (soportar tanto "system" como "developer")
        system_prompt = ""
        processed_messages = messages

        if messages and messages[0].get("role") in ("system", "developer"):
            system_prompt = messages[0].get("content", "")
            processed_messages = messages[1:]
            logger.info(
                f"📋 [{request_id}] System prompt detectado | "
                f"Rol: {messages[0].get('role')} | "
                f"Longitud: {len(system_prompt)} chars"
            )

        # Extraer tool prompt (último mensaje con role="tool")
        tool_prompt = ""
        tool_call_id = None
        last_tool_idx = -1
        
        # Buscar el último mensaje tool
        for idx in range(len(processed_messages) - 1, -1, -1):
            msg = processed_messages[idx]
            if msg.get("role") == "tool":
                tool_prompt = msg.get("content", "")
                tool_call_id = msg.get("tool_call_id", "unknown")
                last_tool_idx = idx
                logger.info(
                    f"🔧 [{request_id}] Tool prompt detectado | "
                    f"Tool call ID: {tool_call_id} | "
                    f"Índice: {idx} | "
                    f"Longitud: {len(tool_prompt)} chars"
                )
                break

        # Extraer user prompt (último mensaje que NO sea tool)
        user_prompt = ""
        last_user_idx = -1
        
        # Buscar el último mensaje de usuario (ignorando mensajes tool)
        for idx in range(len(processed_messages) - 1, -1, -1):
            msg = processed_messages[idx]
            if msg.get("role") != "tool":
                user_prompt = msg.get("content", "")
                last_user_idx = idx
                logger.info(
                    f"📝 [{request_id}] User prompt detectado | "
                    f"Índice: {idx} | "
                    f"Longitud: {len(user_prompt)} chars"
                )
                break
        
        # Determinar qué mensajes van al historial
        # Excluir tanto el último user como el último tool
        indices_to_exclude = set()
        if last_user_idx >= 0:
            indices_to_exclude.add(last_user_idx)
        if last_tool_idx >= 0:
            indices_to_exclude.add(last_tool_idx)
        
        chat_messages_raw = [
            msg for idx, msg in enumerate(processed_messages)
            if idx not in indices_to_exclude
        ]

        # Convertir mensajes al formato esperado por SAI
        chat_messages = []
        for idx, msg in enumerate(chat_messages_raw):
            role = msg.get("role")
            content = msg.get("content", "")
            
            # Los mensajes tool que NO son el último se agregan al historial como user
            if role == "tool":
                tc_id = msg.get("tool_call_id", "unknown")
                tool_content = f"[Tool Response - ID: {tc_id}]\n{content}"
                
                chat_messages.append({
                    "content": tool_content,
                    "role": "user",
                    "id": int(time.time() * 1000) + idx
                })
                
                logger.info(
                    f"🔧 [{request_id}] Mensaje tool histórico procesado | "
                    f"Tool call ID: {tc_id} | "
                    f"Content length: {len(content)} chars"
                )
            else:
                # Mensaje normal (assistant, user, etc.)
                chat_messages.append({
                    "content": content,
                    "role": role,
                    "id": int(time.time() * 1000) + idx
                })

        # Validar tamaño del contexto
        self._check_context_size(total_chars, request_id)

        return system_prompt, user_prompt, tool_prompt, tool_call_id, chat_messages

    def _extract_tool_calls_from_plain_text_end(
            self,
            response_text: Optional[str],
            request_id: str
    ) -> tuple[Optional[list], str]:
        """
        Extrae un JSON al FINAL de la respuesta plana con {"tool_calls":[...]}.
        Evita el bug de rfind("{") cuando hay "{...}" dentro de strings (p.ej. arguments).
        Devuelve (tool_calls_normalizadas, texto_sin_json).
        """
        if response_text is None:
            logger.info(f"🧪 [{request_id}] [TOOLS] extractor: response_text=None")
            return None, ""

        text = str(response_text)

        # Logs "antes"
        tail_preview = text[-500:] if len(text) > 500 else text
        logger.info(f"🧪 [{request_id}] [TOOLS] BEFORE extracted | len={len(text)} | tail_preview={tail_preview!r}")

        trimmed = text.rstrip()
        if not trimmed:
            logger.info(f"🧪 [{request_id}] [TOOLS] extractor: trimmed vacío")
            return None, ""

        if not trimmed.endswith("}"):
            logger.info(f"[{request_id}] [TOOLS] extractor: no termina en '}}' -> no hay JSON final")
            return None, text

        decoder = json.JSONDecoder()

        def _normalize_tool_calls(tool_calls) -> Optional[list]:
            # ... existing code ...
            if not tool_calls:
                return None
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            normalized = []
            for i, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue

                # A) Estilo OpenAI-ish
                if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    arguments = fn.get("arguments", "{}")
                # B) Estilo compacto
                else:
                    name = tc.get("name")
                    arguments = tc.get("arguments", "{}")

                if not name:
                    continue

                normalized.append({
                    "id": tc.get("id", f"call_{request_id}_{i}"),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
                    }
                })
            return normalized or None

        def _try_parse_from(start_idx: int) -> tuple[Optional[list], Optional[int], Optional[int], Optional[str]]:
            # ... existing code ...
            candidate = trimmed[start_idx:]
            try:
                obj, end = decoder.raw_decode(candidate)
            except Exception:
                return None, None, None, None

            # Debe consumir todo el final salvo whitespace
            if candidate[end:].strip():
                return None, None, None, None

            if not (isinstance(obj, dict) and "tool_calls" in obj):
                return None, None, None, None

            tool_calls = _normalize_tool_calls(obj.get("tool_calls"))
            if not tool_calls:
                return None, None, None, None

            json_raw = candidate[:end]
            return tool_calls, start_idx, start_idx + end, json_raw

        # 1) Camino rápido: buscar el marcador {"tool_calls" desde el final
        marker_idx = trimmed.rfind('{"tool_calls"')
        if marker_idx != -1:
            tool_calls, js, je, json_raw = _try_parse_from(marker_idx)
            if tool_calls:
                cleaned = trimmed[:js].rstrip()

                # Logs "después" (AFTER)
                after_tail = cleaned[-500:] if len(cleaned) > 500 else cleaned
                logger.info(
                    f"🧪 [{request_id}] [TOOLS] AFTER extracted | "
                    f"cleaned_len={len(cleaned)} | "
                    f"tool_calls_count={len(tool_calls)} | "
                    f"json_removed_len={len(json_raw)} | "
                    f"cleaned_tail_preview={after_tail!r}"
                )

                print(f"[{request_id}] [TOOLS] FOUND via marker at {marker_idx} | tool_calls_count={len(tool_calls)} | json_len={len(json_raw)}")
                print(f"[{request_id}] [TOOLS] AFTER cleaned_len={len(cleaned)} cleaned_tail_preview={after_tail!r}")
                return tool_calls, cleaned

        # 2) Fallback: escanear todos los '{' hacia atrás (evita caer en '{' dentro de strings)
        brace_positions = [i for i, ch in enumerate(trimmed) if ch == "{"]

        logger.info(f"🧪 [{request_id}] [TOOLS] fallback scan: brace_positions={len(brace_positions)}")

        for start_idx in reversed(brace_positions):
            tool_calls, js, je, json_raw = _try_parse_from(start_idx)
            if tool_calls:
                cleaned = trimmed[:js].rstrip()

                # Logs "después" (AFTER)
                after_tail = cleaned[-500:] if len(cleaned) > 500 else cleaned
                logger.info(
                    f"🧪 [{request_id}] [TOOLS] AFTER extracted | "
                    f"cleaned_len={len(cleaned)} | "
                    f"tool_calls_count={len(tool_calls)} | "
                    f"json_removed_len={len(json_raw)} | "
                    f"cleaned_tail_preview={after_tail!r}"
                )

                print(f"[{request_id}] [TOOLS] FOUND via scan at {start_idx} | tool_calls_count={len(tool_calls)} | json_len={len(json_raw)}")
                print(f"[{request_id}] [TOOLS] AFTER cleaned_len={len(cleaned)} cleaned_tail_preview={after_tail!r}")
                return tool_calls, cleaned

        # No se encontró JSON tool_calls al final
        logger.info(f"🧪 [{request_id}] [TOOLS] extractor: NO tool_calls JSON encontrado al final")
        return None, text

    # ---------------- Síncrono ----------------
    def completion(self, messages=None, **kwargs) -> ModelResponse:
        request_id = str(uuid.uuid4())[:8]

        if VERBOSE_LOGGING:
            logger.debug(f"⚙️ [{request_id}] kwargs recibidos en completion: {kwargs}")

        if not messages:
            raise ValueError("Se requiere al menos un mensaje")

        user_api_key = self._extract_user_api_key(kwargs, request_id)
        user_agent = self._extract_user_agent(kwargs, request_id)

        # Tools (si vienen desde gateway/converter)
        tools = kwargs.get("tools")
        # Omitir si es lista vacía
        if tools is not None and isinstance(tools, list) and len(tools) == 0:
            tools = None
            logger.info(f"🔧 [{request_id}] [TOOLS] tools es lista vacía [] - omitiendo para evitar falsos positivos")
        
        # NUEVA LÓGICA: Detectar si hay mensajes con role="tool"
        has_tool_messages = any(msg.get("role") == "tool" for msg in messages if isinstance(msg, dict))
        if has_tool_messages and tools is not None:
            logger.info(
                f"🔧 [{request_id}] [TOOLS] Mensajes con role='tool' detectados | "
                f"Acción: Eliminando tools de la entrada (se espera respuesta, no tool_calls)"
            )
            tools = None
        
        has_tools = bool(tools)  # Guardar si hay tools en la entrada
        if has_tools:
            print(f"[{request_id}] [TOOLS] completion(): tools recibidas count={len(tools)}")

        system, user_prompt, tool_prompt, tool_call_id, chat_messages = self._prepare_messages(messages, request_id)

        response_text, finish_reason, usage_data = self._call_sai(
            system,
            user_prompt,
            tool_prompt,
            tool_call_id,
            chat_messages,
            request_id,
            user_api_key=user_api_key,
            model=kwargs.get('model'),
            tools=tools
        )

        # SOLO extraer tool_calls si se enviaron tools en la entrada
        tool_calls = None
        cleaned_text = response_text
        
        if has_tools:
            tool_calls, cleaned_text = self._extract_tool_calls_from_plain_text_end(response_text, request_id)
        else:
            logger.info(f"🧪 [{request_id}] [TOOLS] completion(): NO tools en entrada -> extractor OMITIDO")

        if tool_calls:
            response = ModelResponse(
                usage={
                    "prompt_tokens": usage_data["prompt_tokens"],
                    "completion_tokens": usage_data["completion_tokens"],
                    "total_tokens": usage_data["total_tokens"]
                }
            )
            # CORRECCIÓN: content debe ser None (null en JSON), no string vacío
            response.choices[0].message.content = cleaned_text
            response.choices[0].message.tool_calls = tool_calls
            response.choices[0].finish_reason = "tool_calls"
            response.model = usage_data["model"]
            return response

        # BUGFIX: Siempre asignar a message.content para compatibilidad con OpenAI Chat Completions
        # Solo usar .text para casos legacy específicos
        response = ModelResponse(
            usage={
                "prompt_tokens": usage_data["prompt_tokens"],
                "completion_tokens": usage_data["completion_tokens"],
                "total_tokens": usage_data["total_tokens"]
            }
        )
        response.choices[0].message.content = cleaned_text
        
        # Para compatibilidad con clientes legacy que esperan .text
        if not (user_agent and ('GitKraken' in user_agent or 'Go-http-client' in user_agent)):
            response.text = cleaned_text
            logger.info(
                f"✅ [{request_id}] [RESPONSE] Asignado a message.content Y text | "
                f"Longitud: {len(cleaned_text)} chars"
            )
        else:
            logger.info(
                f"✅ [{request_id}] [RESPONSE] Asignado solo a message.content | "
                f"User-Agent: {user_agent} | "
                f"Longitud: {len(cleaned_text)} chars"
            )

        response.choices[0].finish_reason = finish_reason
        response.model = usage_data["model"]

        return response

    # ---------------- Asíncrono ----------------
    async def acompletion(self, messages=None, **kwargs) -> ModelResponse:
        request_id = kwargs.pop('_request_id', None) or str(uuid.uuid4())[:8]

        if VERBOSE_LOGGING:
            logger.debug(f"⚙️ [{request_id}] kwargs recibidos en acompletion: {kwargs}")

        if not messages:
            raise ValueError("Se requiere al menos un mensaje")

        user_api_key = self._extract_user_api_key(kwargs, request_id)
        user_agent = self._extract_user_agent(kwargs, request_id)
        model = kwargs.get('model')

        tools = kwargs.get("tools")
        # Omitir si es lista vacía
        if tools is not None and isinstance(tools, list) and len(tools) == 0:
            tools = None
            logger.info(f"🔧 [{request_id}] [TOOLS] tools es lista vacía [] - omitiendo para evitar falsos positivos")
        
        # NUEVA LÓGICA: Detectar si hay mensajes con role="tool"
        has_tool_messages = any(msg.get("role") == "tool" for msg in messages if isinstance(msg, dict))
        if has_tool_messages and tools is not None:
            logger.info(
                f"🔧 [{request_id}] [TOOLS] Mensajes con role='tool' detectados | "
                f"Acción: Eliminando tools de la entrada (se espera respuesta, no tool_calls)"
            )
            tools = None
        
        has_tools = bool(tools)  # Guardar si hay tools en la entrada
        logger.info(f"🧪 [{request_id}] [TOOLS] acompletion(): tools_present={has_tools} tools_type={type(tools).__name__ if tools else 'None'}")

        system, user_prompt, tool_prompt, tool_call_id, chat_messages = self._prepare_messages(messages, request_id)

        loop = asyncio.get_running_loop()

        from functools import partial
        call_sai_with_params = partial(
            self._call_sai, 
            user_api_key=user_api_key, 
            model=model, 
            tools=tools
        )

        response_text, finish_reason, usage_data = await loop.run_in_executor(
            None,
            call_sai_with_params,
            system,
            user_prompt,
            tool_prompt,
            tool_call_id,
            chat_messages,
            request_id
        )

        # Log explícito de llegada de respuesta
        resp_tail = str(response_text)[-400:] if response_text else ""
        logger.info(
            f"🧪 [{request_id}] [TOOLS] acompletion(): response_text_len={len(str(response_text)) if response_text is not None else 0} "
            f"resp_tail={resp_tail!r}"
        )

        # SOLO extraer tool_calls si se enviaron tools en la entrada
        tool_calls = None
        cleaned_text = response_text
        
        if has_tools:
            tool_calls, cleaned_text = self._extract_tool_calls_from_plain_text_end(response_text, request_id)
        else:
            logger.info(f"🧪 [{request_id}] [TOOLS] acompletion(): NO tools en entrada -> extractor OMITIDO")

        logger.info(f"🧪 [{request_id}] [TOOLS] acompletion(): extracted_tool_calls={bool(tool_calls)} cleaned_len={len(cleaned_text)}")

        if tool_calls:
            response = ModelResponse(
                usage={
                    "prompt_tokens": usage_data["prompt_tokens"],
                    "completion_tokens": usage_data["completion_tokens"],
                    "total_tokens": usage_data["total_tokens"]
                }
            )
            response.choices[0].message.content = cleaned_text  # ← Cambio aquí
            response.choices[0].message.tool_calls = tool_calls
            response.choices[0].finish_reason = "tool_calls"
            response.model = usage_data["model"]

            return response

        # BUGFIX: Siempre asignar a message.content para compatibilidad con OpenAI Chat Completions
        # Solo usar .text para casos legacy específicos
        response = ModelResponse(
            usage={
                "prompt_tokens": usage_data["prompt_tokens"],
                "completion_tokens": usage_data["completion_tokens"],
                "total_tokens": usage_data["total_tokens"]
            }
        )
        response.choices[0].message.content = cleaned_text
        
        # Para compatibilidad con clientes legacy que esperan .text
        if not (user_agent and ('GitKraken' in user_agent or 'Go-http-client' in user_agent)):
            response.text = cleaned_text
            logger.info(
                f"✅ [{request_id}] [RESPONSE] Asignado a message.content Y text | "
                f"Longitud: {len(cleaned_text)} chars"
            )
        else:
            logger.info(
                f"✅ [{request_id}] [RESPONSE] Asignado solo a message.content | "
                f"User-Agent: {user_agent} | "
                f"Longitud: {len(cleaned_text)} chars"
            )

        response.choices[0].finish_reason = finish_reason
        response.model = usage_data["model"]
        return response

    # ---------------- Streaming ----------------
    async def astreaming(self, messages=None, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        # Generar request_id siempre (independiente de VERBOSE_LOGGING)
        request_id = str(uuid.uuid4())[:8]

        # Log detallado de kwargs solo si VERBOSE_LOGGING está activado
        if VERBOSE_LOGGING:
            logger.debug(f"⚙️ [{request_id}] kwargs recibidos en astreaming: {kwargs}")

        # Pasar el request_id y todos los kwargs a acompletion
        kwargs['_request_id'] = request_id

        response = await self.acompletion(messages, **kwargs)

        # Extraer texto y tool_calls de la respuesta
        text = None
        tool_calls = None
    
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
        
            if hasattr(choice, 'message'):
                # Extraer tool_calls si existen
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    tool_calls = choice.message.tool_calls
                    logger.info(
                        f"🔧 [{request_id}] [STREAMING] Response contiene tool_calls | "
                        f"Count: {len(tool_calls)}"
                    )
                
                # Extraer content (puede coexistir con tool_calls)
                if hasattr(choice.message, 'content') and choice.message.content:
                    text = choice.message.content
                    logger.info(
                        f"choice.message.content: {bool(text)}"
                    )
        elif hasattr(response, 'text') and response.text:
            text = response.text
            logger.info(
                f"response.text: {bool(text)}"
            )

        # Validar que haya al menos texto o tool_calls
        if not text and not tool_calls:
            logger.error(
                f"❌ [{request_id}] [STREAMING] No se pudo extraer ni texto ni tool_calls de la respuesta | "
                f"response.choices existe: {hasattr(response, 'choices')} | "
                f"response.text existe: {hasattr(response, 'text')} | "
                f"Tipo de response: {type(response).__name__}"
            )
            return

        usage_dict = response.usage.__dict__ if not isinstance(response.usage, dict) else response.usage
        finish_reason = response.choices[0].finish_reason

        # Si hay tool_calls, emitir un chunk especial con tool_use
        if tool_calls:
            logger.info(
                f"🔧 [{request_id}] [STREAMING] Emitiendo chunk con tool_calls | "
                f"Count: {len(tool_calls)} | "
                f"Tiene texto adicional: {bool(text)}"
            )
            
            # Convertir tool_calls a formato serializable
            tool_calls_list = []
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
                tool_calls_list.append(tc_dict)
            
            # Emitir chunk con tool_calls (sin texto)
            yield GenericStreamingChunk(
                text="",
                index=0,
                is_finished=not text,  # Solo es final si no hay texto adicional
                finish_reason=finish_reason if not text else None,
                tool_use=tool_calls_list,
                usage=usage_dict
            )

        # Si hay texto, emitirlo en chunks (puede ser adicional a tool_calls)
        if text:
            logger.info(
                f"📝 [{request_id}] [STREAMING] Emitiendo texto en chunks | "
                f"Longitud: {len(text)} chars | "
                f"Chunk size: {CHUNK_SIZE} | "
                f"Tiene tool_calls previos: {bool(tool_calls)}"
            )
            
            start_index = 1 if tool_calls else 0  # Ajustar índice si ya se emitió chunk de tool_calls
            
            for idx, start in enumerate(range(0, len(text), CHUNK_SIZE), start=start_index):
                chunk_text = text[start:start + CHUNK_SIZE]
                await asyncio.sleep(0.001)
                is_final = start + CHUNK_SIZE >= len(text)
                
                yield GenericStreamingChunk(
                    text=chunk_text,
                    index=idx,
                    is_finished=is_final,
                    finish_reason=finish_reason if is_final else None,
                    tool_use=None,
                    usage=usage_dict if is_final else None
                )

    # ---------------- Métodos auxiliares para reducir complejidad ----------------
    def _determine_auth_method(self, user_api_key: Optional[str], request_id: str) -> tuple[Optional[str], Optional[str], str]:
        """
        Determina el método de autenticación a usar.

        Returns:
            tuple: (custom_cookie, api_key_to_use, auth_type)
        """
        custom_cookie = None
        api_key_to_use = None

        if user_api_key and "Cookies" in user_api_key:
            custom_cookie = user_api_key
            logger.info(
                f"🍪 [{request_id}] [AUTH] user_api_key contiene 'Cookies' | "
                f"Longitud: {len(custom_cookie)} caracteres | "
                f"Acción: Se usará como Cookie personalizada en lugar de API Key"
            )
            auth_type = "Cookie personalizada"
        else:
            api_key_to_use = user_api_key if user_api_key else SAI_KEY
            auth_type = "API Key personalizada" if user_api_key else "API Key por defecto"

        return custom_cookie, api_key_to_use, auth_type

    def _execute_request_with_retry(self, url: str, data: dict, custom_cookie: Optional[str], 
                                   api_key_to_use: Optional[str], user_api_key: Optional[str], 
                                   request_id: str) -> tuple[Optional[str], Optional[dict], str]:
        """
        Ejecuta el request con lógica de reintento.

        Returns:
            tuple: (response, response_headers, auth_method_used)
        """
        response = None
        response_headers = None
        auth_method_used = None

        if custom_cookie:
            logger.info(
                f"🍪 [{request_id}] [AUTH] Usando Cookie personalizada del usuario | "
                f"Longitud: {len(custom_cookie)} caracteres"
            )
            response, response_headers = self._make_request(
                url, data, use_api_key=False, request_id=request_id, custom_cookie=custom_cookie
            )
            auth_method_used = "Cookie personalizada del usuario"
        elif api_key_to_use:
            api_key_type = "personalizada del usuario" if user_api_key else "del sistema (SAI_KEY)"
            logger.info(
                f"🔑 [{request_id}] [AUTH] Intento #1 con API Key {api_key_type} | "
                f"Longitud: {len(api_key_to_use)} caracteres"
            )
            response, response_headers = self._make_request(
                url, data, use_api_key=True, request_id=request_id, custom_api_key=api_key_to_use
            )
            auth_method_used = f"API Key ({api_key_type})"

            if response == "UNAUTHORIZED_ERROR":
                logger.error(
                    f"❌ [{request_id}] [AUTH] Intento #1 FALLIDO: HTTP 401 Unauthorized | "
                    f"API Key {api_key_type} rechazada por el servidor | "
                    f"Decisión: NO se reintentará con Cookie (error de credenciales)"
                )
            elif response is None and SAI_COOKIE:
                logger.info(
                    f"🔄 [{request_id}] [AUTH] Intento #1 FALLIDO: Rate limit (429) con API Key | "
                    f"Razón probable: 'Test template usage limit exceeded' | "
                    f"Decisión: Reintentando con Cookie (Intento #2)"
                )
                response, response_headers = self._make_request(url, data, use_api_key=False, request_id=request_id)
                auth_method_used = "Cookie (fallback desde API Key)"
                if response:
                    logger.info(f"✅ [{request_id}] [AUTH] Intento #2 EXITOSO con Cookie")
            elif response is None and not SAI_COOKIE:
                logger.error(
                    f"❌ [{request_id}] [AUTH] Intento #1 FALLIDO: Rate limit (429) con API Key | "
                    f"Problema: No hay SAI_COOKIE configurada para reintentar | "
                    f"Solución: Configure SAI_COOKIE como método de autenticación alternativo"
                )
        else:
            logger.info(
                f"🍪 [{request_id}] [AUTH] Usando Cookie del sistema | "
                f"Razón: No hay API Key configurada (ni personalizada ni SAI_KEY)"
            )
            response, response_headers = self._make_request(url, data, use_api_key=False, request_id=request_id)
            auth_method_used = "Cookie (única opción disponible)"

        return response, response_headers, auth_method_used

    def _build_auth_error_message(self, auth_method_used: str) -> str:
        """Construye el mensaje de error de autenticación según el método usado."""
        if "API Key" in auth_method_used:
            return (
                "🔐 **Error de Autenticación (HTTP 401)**\n\n"
                "La **API Key** proporcionada no es válida o ha expirado.\n\n"
                "**Acciones sugeridas:**\n"
                "1. Verifique que la API Key esté correctamente configurada en SAI_KEY\n"
                "2. Genere una nueva API Key desde el panel de administración de SAI\n"
                "3. Actualice la variable de entorno SAI_KEY con la nueva clave\n"
                "4. Reinicie el servicio después de actualizar las credenciales\n\n"
                "Si el problema persiste, contacte al administrador del sistema."
            )
        elif "Cookie" in auth_method_used:
            return (
                "🔐 **Error de Autenticación (HTTP 401)**\n\n"
                "La **Cookie de sesión** proporcionada no es válida o ha expirado.\n\n"
                "**Acciones sugeridas:**\n"
                "1. Verifique que la Cookie esté correctamente configurada en SAI_COOKIE\n"
                "2. Inicie sesión nuevamente en SAI y obtenga una nueva cookie de sesión\n"
                "3. Actualice la variable de entorno SAI_COOKIE con la nueva cookie\n"
                "4. Reinicie el servicio después de actualizar las credenciales\n\n"
                "Si el problema persiste, contacte al administrador del sistema."
            )
        else:
            return (
                "🔐 **Error de Autenticación (HTTP 401)**\n\n"
                "Las credenciales de autenticación proporcionadas no son válidas o han expirado.\n\n"
                "**Acciones sugeridas:**\n"
                "1. Verifique que SAI_KEY o SAI_COOKIE estén correctamente configurados\n"
                "2. Genere nuevas credenciales desde el panel de SAI\n"
                "3. Actualice las variables de entorno correspondientes\n"
                "4. Reinicie el servicio después de actualizar las credenciales\n\n"
                "Si el problema persiste, contacte al administrador del sistema."
            )

    def _handle_error_response(self, response: Optional[str], auth_method_used: str,
                               request_id: str, chat_messages: list, url: str) -> Optional[tuple[str, str, dict]]:
        """
        Maneja respuestas de error y retorna el mensaje apropiado.

        Returns:
            tuple o None: (error_message, finish_reason, usage_data) si hay error, None si no hay error
        """
        usage_data = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": "unknown",
            "response_time": 0.0
        }

        if response == "UNAUTHORIZED_ERROR":
            auth_info = auth_method_used if auth_method_used else "desconocido"
            logger.error(
                f"❌ [SERVER → CLIENT] [{request_id}] Error de autenticación (HTTP 401) | "
                f"Método usado: {auth_info} | "
                f"Las credenciales proporcionadas no son válidas"
            )
            error_message = self._build_auth_error_message(auth_method_used)
            return error_message, "error", usage_data

        if response == "PROMPT_TOO_LONG":
            logger.error(
                f"❌ [SERVER → CLIENT] [{request_id}] Error: Contexto demasiado largo | "
                f"Mensajes en historial: {len(chat_messages)} | "
                f"Acción requerida: El cliente debe reducir el historial"
            )
            return (
                "⚠️ **Contexto demasiado largo**\n\n"
                f"El historial de conversación excede el límite del modelo ({len(chat_messages)} mensajes).\n"
                "**Acciones sugeridas:**\n"
                "1. Reduzca el número de mensajes en el historial\n"
                "2. Inicie una nueva conversación\n"
                "3. Resuma el contexto anterior en un mensaje más corto"
            ), "length", usage_data

        if response == "HTTP_500_ERROR":
            logger.error(
                f"❌ [SERVER → CLIENT] [{request_id}] Error HTTP 500 no controlado | "
                f"Template: {SAI_TEMPLATE_ID}"
            )
            return (
                "❌ **Error interno del servidor SAI (HTTP 500)**\n\n"
                "El servidor SAI encontró un error inesperado al procesar la solicitud.\n"
                "**Posibles causas:**\n"
                "1. Error interno del modelo o servicio\n"
                "2. Configuración incorrecta del template\n"
                "3. Problema temporal del servidor\n\n"
                "Por favor, intente nuevamente. Si el problema persiste, contacte al administrador."
            ), "error", usage_data

        if response is None:
            logger.error(
                f"❌ [SERVER → CLIENT] [{request_id}] Error: Sin respuesta de SAI | "
                f"Template: {SAI_TEMPLATE_ID} | "
                f"URL: {url} | "
                f"Auth disponible: API Key={bool(SAI_KEY)}, Cookie={bool(SAI_COOKIE)}"
            )
            return (
                "❌ **Error de conexión con SAI**\n\n"
                "No se pudo obtener respuesta del servidor SAI.\n"
                "**Posibles causas:**\n"
                "1. Problemas de red o conectividad\n"
                "2. Credenciales de autenticación inválidas\n"
                "3. Servicio SAI temporalmente no disponible\n\n"
                "Por favor, intente nuevamente en unos momentos."
            ), "error", usage_data

        return None

    def _update_usage_data(self, response_headers: Optional[dict]) -> dict:
        """Actualiza y retorna los datos de uso desde los headers de respuesta."""
        usage_data = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": "unknown",
            "response_time": 0.0
        }

        if response_headers:
            usage_data["prompt_tokens"] = response_headers.get("prompt_tokens", 0)
            usage_data["completion_tokens"] = response_headers.get("completion_tokens", 0)
            usage_data["total_tokens"] = usage_data["prompt_tokens"] + usage_data["completion_tokens"]
            usage_data["model"] = response_headers.get("model", "unknown")
            usage_data["response_time"] = response_headers.get("response_time", 0.0)

        return usage_data

    def _log_successful_response(self, request_id: str, response: str, response_headers: Optional[dict], usage_data: dict):
        """Registra información de una respuesta exitosa."""
        status_code = response_headers.get("status_code", "N/A") if response_headers else "N/A"
        response_time = usage_data['response_time']
        tokens_per_second = response_headers.get("tokens_per_second", 0.0) if response_headers else 0.0

        logger.info(
            f"✅ [SERVER → CLIENT] [{request_id}] Respuesta lista para enviar | "
            f"Status: {status_code} | "
            f"⏱️ Latencia: {response_time:.2f}s | "
            f"Longitud: {len(response)} chars | "
            f"Tokens: {usage_data['prompt_tokens']} → {usage_data['completion_tokens']} (total: {usage_data['total_tokens']}) | "
            f"Velocidad: {tokens_per_second:.1f} tok/s | "
            f"Modelo: {usage_data['model']} | "
            f"Preview: {response[:120]!r}{'...' if len(response) > 120 else ''}"
        )

    # ---------------- Métodos auxiliares para _make_request ----------------
    def _setup_request_headers(self, use_api_key: bool, custom_api_key: Optional[str],
                               custom_cookie: Optional[str], request_id: str) -> tuple[dict, str]:
        """
        Configura los headers de autenticación para la petición.

        Returns:
            tuple: (headers, auth_method)
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate"
        }

        auth_method = "API Key" if use_api_key else "Cookie"

        if use_api_key and (custom_api_key or SAI_KEY):
            headers["X-Api-Key"] = custom_api_key if custom_api_key else SAI_KEY
        elif custom_cookie:
            headers["Cookie"] = custom_cookie
            auth_method = "Cookie personalizada"
        elif SAI_COOKIE:
            headers["Cookie"] = SAI_COOKIE
        else:
            logger.error(f"❌ [{request_id}] No hay método de autenticación disponible (ni API Key ni Cookie)")
            return None, None

        return headers, auth_method

    def _log_request_payload(self, data: dict, auth_method: str, request_timeout: int, request_id: str):
        """Registra información del payload de la petición."""
        chat_msg_count = len(data.get("chatMessages", []))
        system_length = len(data.get("inputs", {}).get("system", ""))
        user_length = len(data.get("inputs", {}).get("user", ""))

        logger.info(
            f"🌐 [SERVER → SAI] [{request_id}] Enviando HTTP POST | "
            f"Auth: {auth_method} | "
            f"Timeout: {request_timeout}s | "
            f"Payload: system={system_length} chars, user={user_length} chars, historial={chat_msg_count} msgs"
        )

        if VERBOSE_LOGGING:
            system_preview = data.get("inputs", {}).get("system", "")[:80]
            user_preview = data.get("inputs", {}).get("user", "")[:80]
            logger.debug(
                f"[{request_id}] [VERBOSE] Payload details | "
                f"System preview: {system_preview!r}{'...' if len(system_preview) >= 80 else ''} | "
                f"User preview: {user_preview!r}{'...' if len(user_preview) >= 80 else ''}"
            )
            logger.debug(
                f"[{request_id}] [VERBOSE] Chat messages ({chat_msg_count} total): "
                f"{data.get('chatMessages', [])}"
            )

    def _execute_http_request(self, url: str, data: dict, headers: dict,
                             request_timeout: int, request_id: str):
        """
        Ejecuta la petición HTTP POST.

        Returns:
            requests.Response object
        """
        start_time = time.time()
        logger.debug(f"[{request_id}] [HTTP] Iniciando petición POST a SAI...")

        resp = http_session.post(url, json=data, headers=headers, timeout=request_timeout, verify=False)
        resp.raise_for_status()

        response_time = time.time() - start_time
        logger.debug(f"[{request_id}] [HTTP] Respuesta recibida en {response_time:.2f}s | Status: {resp.status_code}")

        return resp

    def _extract_response_headers(self, resp, response_time: float) -> dict:
        """Extrae y procesa los headers de respuesta."""
        try:
            prompt_tokens = int(resp.headers.get("prompttokens", 0))
        except (ValueError, TypeError):
            prompt_tokens = 0

        try:
            completion_tokens = int(resp.headers.get("completiontokens", 0))
        except (ValueError, TypeError):
            completion_tokens = 0

        response_headers = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": resp.headers.get("model", "unknown"),
            "response_time": response_time
        }

        tokens_per_second = response_headers['completion_tokens'] / response_time if response_time > 0 else 0
        response_headers['status_code'] = resp.status_code
        response_headers['tokens_per_second'] = tokens_per_second
        response_headers['response_length'] = len(resp.text)

        return response_headers

    def _handle_http_401_error(self, resp, auth_method: str, url: str, request_id: str) -> tuple[str, None]:
        """Maneja errores HTTP 401 Unauthorized."""
        logger.error(
            f"🔐 [{request_id}] [HTTP 401] Unauthorized | "
            f"Auth usado: {auth_method} | "
            f"Diagnóstico: Credencial rechazada por el servidor SAI | "
            f"URL: {url} | "
            f"Acción: Retornando UNAUTHORIZED_ERROR (no se reintentará)"
        )
        return "UNAUTHORIZED_ERROR", None

    def _handle_http_429_error(self, resp, auth_method: str, request_id: str) -> tuple[Optional[str], None]:
        """Maneja errores HTTP 429 Rate Limit."""
        response_text = resp.text if resp else ""

        if "Test template usage limit exceeded" in response_text:
            logger.warning(
                f"⚠️ [{request_id}] [HTTP 429] Rate Limit - Test Template | "
                f"Auth usado: {auth_method} | "
                f"Diagnóstico: Límite de uso de template de prueba excedido | "
                f"Acción: Retornando None para reintentar con Cookie si está disponible"
            )
            return None, None
        else:
            logger.error(
                f"❌ [{request_id}] [HTTP 429] Rate Limit - Otro tipo | "
                f"Auth usado: {auth_method} | "
                f"Respuesta del servidor: {response_text[:200]} | "
                f"Acción: Retornando None (sin reintento)"
            )
            return None, None

    def _handle_http_500_error(self, resp, auth_method: str, request_id: str) -> tuple[str, None]:
        """Maneja errores HTTP 500 Internal Server Error."""
        response_text = resp.text if resp else ""

        if "prompt is too long" in response_text.lower() or "openaicompatible" in response_text.lower():
            logger.warning(
                f"⚠️ [{request_id}] [HTTP 500] Prompt Too Long | "
                f"Auth usado: {auth_method} | "
                f"Diagnóstico: El contexto excede el límite del modelo | "
                f"Respuesta SAI (preview): {response_text[:200]} | "
                f"Acción: Retornando PROMPT_TOO_LONG (finish_reason=length)"
            )
            return "PROMPT_TOO_LONG", None
        else:
            logger.error(
                f"❌ [{request_id}] [HTTP 500] Internal Server Error | "
                f"Auth usado: {auth_method} | "
                f"Diagnóstico: Error interno del servidor SAI (no relacionado con tamaño de prompt) | "
                f"Respuesta SAI (preview): {response_text[:200]} | "
                f"Acción: Retornando HTTP_500_ERROR (finish_reason=error)"
            )
            return "HTTP_500_ERROR", None

    def _handle_other_http_errors(self, resp, auth_method: str, url: str, e: Exception, request_id: str) -> tuple[None, None]:
        """Maneja otros errores HTTP no específicos."""
        status_code = resp.status_code if resp else "N/A"
        response_text = resp.text[:200] if resp else ""

        logger.error(
            f"❌ [{request_id}] [HTTP {status_code}] Error no manejado específicamente | "
            f"Auth usado: {auth_method} | "
            f"URL: {url} | "
            f"Exception: {type(e).__name__}: {str(e)} | "
            f"Respuesta del servidor: {response_text} | "
            f"Acción: Retornando None"
        )
        return None, None

    def _handle_request_exceptions(self, e: Exception, resp, auth_method: str, url: str,
                                   request_timeout: int, request_id: str) -> tuple[Optional[str], Optional[dict]]:
        """
        Maneja todas las excepciones que pueden ocurrir durante una petición HTTP.

        Returns:
            tuple: (response_text, response_headers) o (None, None) en caso de error
        """
        if isinstance(e, requests.HTTPError):
            if resp is not None and resp.status_code == 401:
                return self._handle_http_401_error(resp, auth_method, url, request_id)

            if resp is not None and resp.status_code == 429:
                return self._handle_http_429_error(resp, auth_method, request_id)

            if resp is not None and resp.status_code == 500:
                return self._handle_http_500_error(resp, auth_method, request_id)

            return self._handle_other_http_errors(resp, auth_method, url, e, request_id)

        elif isinstance(e, requests.Timeout):
            logger.error(
                f"⏱️ [{request_id}] [TIMEOUT] Tiempo de espera agotado | "
                f"Timeout configurado: {request_timeout}s | "
                f"Auth usado: {auth_method} | "
                f"URL: {url} | "
                f"Diagnóstico: El servidor SAI no respondió en el tiempo esperado | "
                f"Acción: Retornando None"
            )
            return None, None

        elif isinstance(e, requests.RequestException):
            logger.error(
                f"❌ [{request_id}] [NETWORK ERROR] Error de conexión | "
                f"Auth usado: {auth_method} | "
                f"URL: {url} | "
                f"Exception: {type(e).__name__}: {str(e)} | "
                f"Diagnóstico: Problema de red o conectividad con SAI | "
                f"Acción: Retornando None"
            )
            return None, None

        return None, None

    # ---------------- Llamada privada a SAI (refactorizada) ----------------
    def _call_sai(self, system: str, user: str, tool: str, tool_call_id: Optional[str], 
                  chat_messages: list, request_id: str,
                  user_api_key: Optional[str] = None, model: Optional[str] = None, 
                  tools: Optional = None) -> tuple[str, str, dict]:
        # Construir URL base
        url = f"{SAI_URL}/api/templates/{SAI_TEMPLATE_ID}/execute"

        # Agregar modelOverride si se proporciona un modelo
        if model:
            url = f"{url}?modelOverride={model}"
            logger.info(
                f"🎯 [{request_id}] [MODEL] Model override aplicado | "
                f"Modelo solicitado: {model} | "
                f"URL: {url}"
            )

        # Construir inputs con system, user, tool y tools
        data = {
            "inputs": {
                "system": system,
                "user": user,
                "tool": tool if tool else None,  # Agregar tool message
                "tools": None  # tools definitions (se llenará después)
            }
        }
        
        if chat_messages:
            data["chatMessages"] = chat_messages

        # Serializar tools definitions como JSON string
        tools_json_str = None
        if tools is None:
            tools_json_str = None
            logger.debug(f"[{request_id}] [TOOLS] _call_sai(): tools=None -> inputs.tools=None")
        elif isinstance(tools, list) and len(tools) == 0:
            tools_json_str = None
            logger.info(f"🔧 [{request_id}] [TOOLS] _call_sai(): tools=[] (lista vacía) -> inputs.tools=None (omitido)")
        elif isinstance(tools, str):
            tools_json_str = tools
            try:
                json.loads(tools_json_str)
                logger.debug(f"[{request_id}] [TOOLS] _call_sai(): tools ya es JSON string válido (len={len(tools_json_str)})")
            except Exception as e:
                logger.warning(f"⚠️ [{request_id}] [TOOLS] _call_sai(): WARNING tools es string pero NO es JSON válido: {type(e).__name__}: {str(e)}")
        else:
            tools_json_str = json.dumps(tools, ensure_ascii=False)
            logger.debug(f"[{request_id}] [TOOLS] _call_sai(): tools serializadas a JSON string (len={len(tools_json_str)})")

        data["inputs"]["tools"] = tools_json_str

        # Logging de tool message si existe
        if tool:
            logger.info(
                f"🔧 [{request_id}] [TOOL] Tool message incluido en inputs.tool | "
                f"Tool call ID: {tool_call_id} | "
                f"Longitud: {len(tool)} chars"
            )
            if VERBOSE_LOGGING:
                logger.debug(f"[{request_id}] [TOOL] inputs.tool preview={tool[:180]!r}")

        if tools_json_str and VERBOSE_LOGGING:
            logger.debug(f"[{request_id}] [TOOLS] inputs.tools preview={tools_json_str[:180]!r}")

        # Determinar método de autenticación
        custom_cookie, api_key_to_use, auth_type = self._determine_auth_method(user_api_key, request_id)

        # Logging del request
        if not custom_cookie:
            logger.info(
                f"🐍 [SERVER → SAI] [{request_id}] Preparando request | "
                f"System: {len(system)} chars | "
                f"User: {len(user)} chars | "
                f"Tool: {len(tool) if tool else 0} chars | "
                f"Historial: {len(chat_messages)} mensajes | "
                f"Template: {SAI_TEMPLATE_ID} | "
                f"Auth: {auth_type}"
            )

        if VERBOSE_LOGGING:
            logger.debug(f"[{request_id}] Payload completo:\n{json.dumps(data, indent=2, ensure_ascii=False)}")

        # Ejecutar request con reintentos
        response, response_headers, auth_method_used = self._execute_request_with_retry(
            url, data, custom_cookie, api_key_to_use, user_api_key, request_id
        )

        # Manejar errores
        error_result = self._handle_error_response(response, auth_method_used, request_id, chat_messages, url)
        if error_result:
            return error_result

        # Actualizar datos de uso
        usage_data = self._update_usage_data(response_headers)

        # Log de respuesta exitosa
        self._log_successful_response(request_id, response, response_headers, usage_data)

        return response, "stop", usage_data

    def _make_request(self, url: str, data: dict, use_api_key: bool = False, timeout: int = None, request_id: str = "unknown", custom_api_key: Optional[str] = None, custom_cookie: Optional[str] = None) -> tuple[Optional[str], Optional[dict]]:
        resp = None
        request_timeout = timeout or REQUEST_TIMEOUT

        try:
            # Configurar headers de autenticación
            headers_result = self._setup_request_headers(use_api_key, custom_api_key, custom_cookie, request_id)
            if headers_result is None or headers_result[0] is None:
                return None, None
            headers, auth_method = headers_result

            # Logging del payload
            self._log_request_payload(data, auth_method, request_timeout, request_id)

            if VERBOSE_LOGGING:
                logger.debug(f"[{request_id}] [VERBOSE] Request URL: {url}")
                logger.debug(f"[{request_id}] [VERBOSE] Request headers (sin credenciales): {', '.join(k for k in headers.keys() if k not in ['X-Api-Key', 'Cookie'])}")

            # Ejecutar petición HTTP
            start_time = time.time()
            resp = self._execute_http_request(url, data, headers, request_timeout, request_id)
            response_time = time.time() - start_time

            # Extraer headers de respuesta
            response_headers = self._extract_response_headers(resp, response_time)

            return resp.text, response_headers

        except requests.RequestException as e:
            return self._handle_request_exceptions(e, resp, auth_method, url, request_timeout, request_id)


# ---------------- Instancia global ----------------
sai_llm = SAILLM()
"""Instancia global de SAILLM lista para usar."""

converter = OpenAiSAIConverter()
"""Instancia global del conversor de formatos OpenAI ↔ SAI."""

logger.info(
    f"✅ SAILLM inicializado correctamente | "
    f"Clase: {sai_llm.__class__.__name__} | "
    f"Métodos disponibles: completion, acompletion, astreaming | "
    f"Estado: Listo para recibir peticiones"
)