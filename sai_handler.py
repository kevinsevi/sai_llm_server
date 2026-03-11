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
from sai_exceptions import (
    SAIAPIError,
    SAIAuthenticationError,
    SAIRateLimitError,
    SAIPromptTooLongError,
    SAIServerError,
    SAIConnectionError,
    SAITimeoutError,
    SAIUpstreamInvalidResponseError,
)

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

class SAILLM(CustomLLM):
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
        headers = kwargs.get('headers', {})
        if not isinstance(headers, dict):
            return None, None

        user_api_key = headers.get('user_api_key', '')
        if user_api_key:
            return user_api_key, "headers"

        return None, None

    def _is_valid_api_key(self, api_key: str) -> bool:
        if not api_key:
            return False

        trimmed = str(api_key).strip()
        if not trimmed or trimmed.lower() == "raspberry":
            return False

        return True

    def _extract_user_api_key(self, kwargs: dict, request_id: str) -> Optional[str]:
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
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Cada mensaje debe ser un diccionario")
            if "role" not in msg or "content" not in msg:
                raise ValueError("Cada mensaje debe tener 'role' y 'content'")

    def _convert_to_sai_format(self, messages: list) -> list:
        return [{
            "content": msg.get("content", ""),
            "role": msg.get("role"),
            "id": int(time.time() * 1000) + idx
        } for idx, msg in enumerate(messages)]

    def _check_context_size(self, total_chars: int, request_id: str):
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

        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content")
            processed_messages = messages[1:]
            logger.info(
                f"📋 [{request_id}] System prompt detectado | "
                f"Rol: {messages[0].get('role')} | "
                f"Longitud: {len(system_prompt)} chars"
            )

        tool_call_id = None
        last_tool_idx = -1
        
        # Extraer user prompt (último mensaje que NO sea tool)
        user_prompt = ""
        type_prompt = ""  # ← INICIALIZAR AQUÍ para evitar UnboundLocalError

        # Buscar el último mensaje de usuario (ignorando mensajes tool)
        for idx in range(len(processed_messages) - 1, -1, -1):
            msg = processed_messages[idx]
            if msg.get("role") != "tool":
                user_prompt = msg.get("content", "")
                type_prompt = msg.get("type", "")
                logger.info(
                    f"📝 [{request_id}] User prompt detectado | "
                    f"Índice: {idx} | "
                    f"Longitud: {len(user_prompt)} chars"
                )
                break
        
        # Determinar qué mensajes van al historial
        # Excluir tanto el último user como el último tool
        indices_to_exclude = set()
        # TODO Analizar si se debe quitar el último mensaje de chat_messages
        # if last_user_idx >= 0:
        #    indices_to_exclude.add(last_user_idx)
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
            
            # Los mensajes tool que NO son el último se agregan al historial como tool
            if role == "tool":
                tc_id = msg.get("tool_call_id")
                tool_content = f"[Tool Response - ID: {tc_id}]\n{content}"
                
                chat_messages.append({
                    "content": tool_content,
                    "role": "tool",
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

        return system_prompt, user_prompt, tool_call_id, chat_messages, type_prompt

    def _normalize_tool_calls(self, tool_calls, request_id: str) -> Optional[list]:
        if not tool_calls:
            return None
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        normalized = []
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue

            # A) Estilo OpenAI-ish (Chat Completions): {"type":"function","function":{"name":...,"arguments":"{...}"}}
            if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
                fn = tc.get("function") or {}
                name = fn.get("name")
                arguments = fn.get("arguments", "{}")

            # A2) Estilo "function_call": {"type":"function_call","function":{"name":...,"arguments":{...}}}
            # (ej.: {"cmd":"ls -la"} como dict)
            elif tc.get("type") == "function_call" and isinstance(tc.get("function"), dict):
                fn = tc.get("function") or {}
                name = fn.get("name")
                arguments = fn.get("arguments", "{}")

            # 🆕 A3) Estilo custom_tool_call con function wrapper: {"type":"custom_tool_call","function":{"name":...,"arguments":"..."}}
            elif tc.get("type") == "custom_tool_call" and isinstance(tc.get("function"), dict):
                fn = tc.get("function") or {}
                name = fn.get("name")
                arguments = fn.get("arguments", "")
                logger.info(
                    f"🔧 [{request_id}] [TOOLS] custom_tool_call con function wrapper detectado | "
                    f"Name: {name} | "
                    f"Arguments length: {len(arguments) if isinstance(arguments, str) else 'N/A'}"
                )

            # B) Estilo compacto: {"name":...,"arguments":...}
            else:
                name = tc.get("name")
                arguments = tc.get("arguments", "{}")

            if not name:
                continue

            tc_type = tc.get("type")
            is_custom_tool_call = tc_type == "custom_tool_call"
            expects_json_arguments = not is_custom_tool_call

            # Normalizar argumentos
            if not isinstance(arguments, str):
                if expects_json_arguments:
                    arguments = json.dumps(arguments, ensure_ascii=False)
                else:
                    arguments = "" if arguments is None else str(arguments)
            else:
                # Solo validar JSON para function/function_call (NO para custom_tool_call/apply_patch)
                if expects_json_arguments:
                    try:
                        json.loads(arguments)
                    except json.JSONDecodeError:
                        repaired = arguments.replace('\\"', '"').replace('\\\\', '\\')
                        try:
                            json.loads(repaired)
                            logger.warning(
                                f"⚠️ [{request_id}] [TOOLS] arguments JSON inválido reparado | "
                                f"Name: {name} | Original len: {len(arguments)}"
                            )
                            arguments = repaired
                        except json.JSONDecodeError:
                            logger.warning(
                                f"⚠️ [{request_id}] [TOOLS] arguments JSON inválido no reparable, usando {{}} | "
                                f"Name: {name} | Preview: {arguments[:80]!r}"
                            )
                            arguments = "{}"

            # Normalizar type: "function_call" -> "function" (formato esperado aguas abajo)
            tc_type = tc.get("type")
            if tc_type == "custom":
                tc_type = "custom_tool_call"
            elif tc_type == "function_call":
                tc_type = "function_call"
            # 🆕 Mantener "custom_tool_call" tal cual
            elif tc_type == "custom_tool_call":
                tc_type = "custom_tool_call"

            normalized.append({
                "id": tc.get("id", f"call_{request_id}_{i}"),
                "type": tc_type,
                "function": {
                    "name": name,
                    "arguments": arguments
                }
            })
        return normalized or None

    def _try_parse_from(self, start_idx: int, request_id: str, trimmed, decoder) -> tuple[
        Optional[list], Optional[int], Optional[int], Optional[str]]:
        candidate = trimmed[start_idx:]
        try:
            obj, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError as e:
            # Log detallado del error de JSON
            err_pos = getattr(e, "pos", None)
            err_msg = getattr(e, "msg", str(e))
            err_line = getattr(e, "lineno", None)
            err_col = getattr(e, "colno", None)

            if isinstance(err_pos, int):
                start = max(0, err_pos - 120)
                end_snip = min(len(candidate), err_pos + 120)
                snippet = candidate[start:end_snip]
            else:
                snippet = candidate[:240]

            logger.warning(
                f"⚠️ [{request_id}] [TOOLS] JSONDecodeError en raw_decode | "
                f"msg={err_msg!r} pos={err_pos} line={err_line} col={err_col} | "
                f"start_idx={start_idx} candidate_len={len(candidate)} | "
                f"snippet={snippet!r}"
            )
            return None, None, None, None
        except Exception as e:
            logger.warning(
                f"⚠️ [{request_id}] [TOOLS] Error inesperado parseando JSON | "
                f"type={type(e).__name__} msg={str(e)} start_idx={start_idx}"
            )
            return None, None, None, None

        # Debe consumir todo el final salvo whitespace
        if candidate[end:].strip():
            return None, None, None, None

        # ✅ Caso 1 (existente): wrapper {"tool_calls": ...}
        if isinstance(obj, dict) and "tool_calls" in obj:
            tool_calls = self._normalize_tool_calls(obj.get("tool_calls"), request_id)
            if not tool_calls:
                return None, None, None, None

            json_raw = candidate[:end]
            return tool_calls, start_idx, start_idx + end, json_raw

        # ✅ Caso 2 (nuevo): tool call "suelto" al final
        # Ej: {"type":"function_call","name":"exec_command","arguments":{...}}
        if isinstance(obj, dict) and (obj.get("name") or obj.get("function")):
            tool_calls = self._normalize_tool_calls(obj, request_id)
            if not tool_calls:
                return None, None, None, None

            json_raw = candidate[:end]
            logger.info(
                f"🧪 [{request_id}] [TOOLS] extractor: tool_call JSON suelto detectado al final | "
                f"normalized_count={len(tool_calls)} | json_len={len(json_raw)}"
            )
            return tool_calls, start_idx, start_idx + end, json_raw

        return None, None, None, None

    def _extract_tool_calls_from_plain_text_end(
            self,
            response_text: Optional[str],
            request_id: str
    ) -> tuple[Optional[list], str]:
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

        # NUEVO: Detectar y extraer JSON de bloques markdown ```json\n...\n```
        import re
        markdown_json_pattern = r'```json\s*\n(.*?)\n```'
        markdown_match = re.search(markdown_json_pattern, trimmed, re.DOTALL)

        if markdown_match:
            json_content = markdown_match.group(1).strip()
            markdown_start = markdown_match.start()
            markdown_end = markdown_match.end()

            logger.info(
                f"🧪 [{request_id}] [TOOLS] Markdown JSON block detectado | "
                f"Start: {markdown_start} | End: {markdown_end} | "
                f"JSON length: {len(json_content)} chars"
            )

            # Intentar parsear el JSON extraído
            try:
                obj = json.loads(json_content)

                # ✅ Soportar wrapper {"tool_calls": ...}
                if isinstance(obj, dict) and "tool_calls" in obj:
                    tool_calls = self._normalize_tool_calls(obj.get("tool_calls"), request_id)
                    if tool_calls:
                        cleaned = trimmed[:markdown_start].rstrip()
                        after_tail = cleaned[-500:] if len(cleaned) > 500 else cleaned
                        logger.info(
                            f"🧪 [{request_id}] [TOOLS] AFTER extracted (from markdown) | "
                            f"cleaned_len={len(cleaned)} | "
                            f"tool_calls_count={len(tool_calls)} | "
                            f"markdown_removed_len={markdown_end - markdown_start} | "
                            f"cleaned_tail_preview={after_tail!r}"
                        )
                        return tool_calls, cleaned

                # ✅ Soportar tool_call suelto en markdown
                if isinstance(obj, dict) and (obj.get("name") or obj.get("function")):
                    tool_calls = self._normalize_tool_calls(obj, request_id)
                    if tool_calls:
                        cleaned = trimmed[:markdown_start].rstrip()
                        after_tail = cleaned[-500:] if len(cleaned) > 500 else cleaned
                        logger.info(
                            f"🧪 [{request_id}] [TOOLS] AFTER extracted (from markdown single tool_call) | "
                            f"cleaned_len={len(cleaned)} | "
                            f"tool_calls_count={len(tool_calls)} | "
                            f"markdown_removed_len={markdown_end - markdown_start} | "
                            f"cleaned_tail_preview={after_tail!r}"
                        )
                        return tool_calls, cleaned

            except json.JSONDecodeError as e:
                logger.warning(
                    f"⚠️ [{request_id}] [TOOLS] Markdown JSON block inválido | "
                    f"Error: {str(e)} | "
                    f"Continuando con extracción normal..."
                )

        decoder = json.JSONDecoder()

        # 1) Camino rápido: buscar el marcador {"tool_calls" desde el final
        marker_idx = trimmed.rfind('{"tool_calls"')
        if marker_idx != -1:
            tool_calls, js, je, json_raw = self._try_parse_from(marker_idx, request_id, trimmed, decoder)
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

                return tool_calls, cleaned
            else:
                # NUEVO: Log de diagnóstico cuando el marcador existe pero falla el parsing
                logger.warning(
                    f"⚠️ [{request_id}] [TOOLS] Marcador '{{\"tool_calls\"' encontrado en pos {marker_idx} pero parsing falló | "
                    f"Snippet: {trimmed[marker_idx:marker_idx+100]!r}"
                )

                # FAIL-FAST: si el upstream intentó devolver tool_calls pero el JSON es inválido,
                # no reenviar el JSON crudo como texto (rompe clientes) y no reintentar.
                upstream_preview = trimmed[marker_idx:]
                if len(upstream_preview) > 2000:
                    upstream_preview = upstream_preview[:2000] + "..."
                raise SAIUpstreamInvalidResponseError(
                    "Upstream devolvió tool_calls con JSON inválido (no parseable).",
                    upstream_preview=upstream_preview,
                    exception="JSONDecodeError"
                )

        # 2) Fallback: escanear todos los '{' hacia atrás (evita caer en '{' dentro de strings)
        brace_positions = [i for i, ch in enumerate(trimmed) if ch == "{"]

        logger.info(f"🧪 [{request_id}] [TOOLS] fallback scan: brace_positions={len(brace_positions)}")

        for start_idx in reversed(brace_positions):
            tool_calls, js, je, json_raw = self._try_parse_from(start_idx, request_id, trimmed, decoder)
            if tool_calls:
                cleaned = trimmed[:js].rstrip()

                after_tail = cleaned[-500:] if len(cleaned) > 500 else cleaned
                logger.info(
                    f"🧪 [{request_id}] [TOOLS] AFTER extracted | "
                    f"cleaned_len={len(cleaned)} | "
                    f"tool_calls_count={len(tool_calls)} | "
                    f"json_removed_len={len(json_raw)} | "
                    f"cleaned_tail_preview={after_tail!r}"
                )

                return tool_calls, cleaned

        # 3) Multi-JSON: modelo emitió varios {"tool_calls"...} + texto inventado al final
        EXAMPLE_MARKERS = (
            "se vería", "por ejemplo", "el formato", "como este", "así:", "ejemplo:",
            "sería así", "quedaría así", "como sigue", "a continuación", "siguiente forma",
            "de esta forma", "de este modo", "como muestra", "ilustración",
        )
        INVENTED_RESULT_MARKERS = (
            "- `", "**", "├", "└", "│", ".py`", ".java`", ".ts`", ".js`",
            ".json`", ".yaml`", ".yml`", ".md`", ".txt`",
        )
        all_tool_calls = []
        search_start = 0
        first_json_pos = None
        last_json_end = None

        while True:
            marker_pos = trimmed.find('{"tool_calls"', search_start)
            if marker_pos == -1:
                break
            candidate = trimmed[marker_pos:]
            try:
                obj, end = decoder.raw_decode(candidate)
            except Exception:
                search_start = marker_pos + 1
                continue

            if isinstance(obj, dict) and "tool_calls" in obj:
                tcs = self._normalize_tool_calls(obj.get("tool_calls"), request_id)
                if tcs:
                    if first_json_pos is None:
                        first_json_pos = marker_pos
                    last_json_end = marker_pos + end
                    all_tool_calls.extend(tcs)
            search_start = marker_pos + 1

        if all_tool_calls and first_json_pos is not None:
            text_before = trimmed[:first_json_pos].strip()
            text_after = trimmed[last_json_end:].strip()

            if text_before:
                text_before_lower = text_before.lower()
                is_example_intro = any(marker in text_before_lower for marker in EXAMPLE_MARKERS)
                if is_example_intro:
                    logger.info(
                        f"🧪 [{request_id}] [TOOLS] multi-JSON descartado: texto antes contiene marcador de ejemplo | "
                        f"text_before_preview={text_before[-200:]!r}"
                    )
                    return None, text

            text_after_looks_invented = any(m in text_after for m in INVENTED_RESULT_MARKERS)
            no_text_before = not text_before

            if no_text_before or text_after_looks_invented:
                logger.info(
                    f"🧪 [{request_id}] [TOOLS] multi-JSON ejecutable detectado | "
                    f"count={len(all_tool_calls)} | no_text_before={no_text_before} | "
                    f"text_after_looks_invented={text_after_looks_invented} | "
                    f"text_after_preview={text_after[:200]!r}"
                )
                return all_tool_calls, text_before

            logger.info(
                f"🧪 [{request_id}] [TOOLS] multi-JSON descartado: texto después no parece resultado inventado | "
                f"text_after_preview={text_after[:200]!r}"
            )

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
        api = kwargs["api"]
        # Omitir si es lista vacía
        if tools is not None and isinstance(tools, list) and len(tools) == 0:
            tools = None
            logger.info(f"🔧 [{request_id}] [TOOLS] tools es lista vacía [] - omitiendo para evitar falsos positivos")
        
        # NUEVA LÓGICA: Detectar si hay mensajes con role="tool"
        #has_tool_messages = any(msg.get("role") == "tool" for msg in messages if isinstance(msg, dict))
        #if has_tool_messages and tools is not None:
        #    logger.info(
        #        f"🔧 [{request_id}] [TOOLS] Mensajes con role='tool' detectados | "
        #        f"Acción: Eliminando tools de la entrada (se espera respuesta, no tool_calls)"
        #    )
        #    tools = None
        
        has_tools = bool(tools)  # Guardar si hay tools en la entrada
        if has_tools:
            print(f"[{request_id}] [TOOLS] completion(): tools recibidas count={len(tools)}")

        system, user_prompt, tool_call_id, chat_messages, type_prompt = self._prepare_messages(messages, request_id)

        response_text, finish_reason, usage_data = self._call_sai(
            system,
            user_prompt,
            tool_call_id,
            chat_messages,
            request_id,
            type_prompt,
            api,
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
            # Log para debug: imprimir tool_calls antes de retornar
            logger.info(
                f"🔍 [{request_id}] [DEBUG] tool_calls a retornar | "
                f"Count: {len(tool_calls)} | "
                f"Content: {json.dumps(tool_calls, indent=2, ensure_ascii=False)}"
            )

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
    async def acompletion(self, request_id, messages=None, **kwargs) -> ModelResponse:
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
        #has_tool_messages = any(msg.get("role") == "tool" for msg in messages if isinstance(msg, dict))
        #if has_tool_messages and tools is not None:
        #    logger.info(
        #        f"🔧 [{request_id}] [TOOLS] Mensajes con role='tool' detectados | "
        #        f"Acción: Eliminando tools de la entrada (se espera respuesta, no tool_calls)"
        #    )
        #    tools = None
        
        has_tools = bool(tools)  # Guardar si hay tools en la entrada
        logger.info(f"🧪 [{request_id}] [TOOLS] acompletion(): tools_present={has_tools} tools_type={type(tools).__name__ if tools else 'None'}")

        system, user_prompt, tool_call_id, chat_messages, type_prompt = self._prepare_messages(messages, request_id)

        loop = asyncio.get_running_loop()

        from functools import partial
        call_sai_with_params = partial(
            self._call_sai, 
            user_api_key=user_api_key, 
            model=model, 
            tools=tools,
            type=type_prompt,
            api=kwargs["api"]
        )

        response_text, finish_reason, usage_data = await loop.run_in_executor(
            None,
            call_sai_with_params,
            system,
            user_prompt,
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
            # Log para debug: imprimir tool_calls antes de retornar
            logger.info(
                f"🔍 [{request_id}] [DEBUG] tool_calls a retornar (async) | "
                f"Count: {len(tool_calls)} | "
                f"Content: {json.dumps(tool_calls, indent=2, ensure_ascii=False)}"
            )

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
    async def astreaming(self, request_id, messages=None, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        # Log detallado de kwargs solo si VERBOSE_LOGGING está activado
        if VERBOSE_LOGGING:
            logger.debug(f"⚙️ [{request_id}] kwargs recibidos en astreaming: {kwargs}")

        # Pasar el request_id y todos los kwargs a acompletion
        kwargs['_request_id'] = request_id

        response = await self.acompletion(request_id, messages, **kwargs)

        # Extraer texto y tool_calls de la respuesta
        text = None
        tool_calls = None

        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]

            if hasattr(choice, 'message'):
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    tool_calls = choice.message.tool_calls
                    logger.info(
                        f"🔧 [{request_id}] [STREAMING] Response contiene tool_calls | "
                        f"Count: {len(tool_calls)}"
                    )

                if hasattr(choice.message, 'content') and choice.message.content:
                    text = choice.message.content
                    logger.info(f"choice.message.content: {bool(text)}")
        elif hasattr(response, 'text') and response.text:
            text = response.text
            logger.info(f"response.text: {bool(text)}")

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

        # Convertir tool_calls a formato serializable (si aplica)
        tool_calls_list = None
        if tool_calls:
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

        # ✅ Emitir UN ÚNICO chunk que contenga texto + tool_use (si existen)
        final_text = text or ""
        logger.info(
            f"🧩 [{request_id}] [STREAMING] Emitiendo chunk único | "
            f"text_len={len(final_text)} | tool_calls={len(tool_calls_list) if tool_calls_list else 0}"
        )

        yield GenericStreamingChunk(
            text=final_text,
            index=0,
            is_finished=True,
            finish_reason=finish_reason,
            tool_use=tool_calls_list,
            usage=usage_dict
        )

    # ---------------- Métodos auxiliares para reducir complejidad ----------------
    def _determine_auth_method(self, user_api_key: Optional[str], request_id: str) -> tuple[Optional[str], Optional[str], str]:
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

    def _handle_http_401_error(self, resp, auth_method: str, url: str, request_id: str) -> None:
        """Maneja errores HTTP 401 Unauthorized lanzando SAIAuthenticationError."""
        logger.error(
            f"🔐 [{request_id}] [HTTP 401] Unauthorized | "
            f"Auth usado: {auth_method} | "
            f"Diagnóstico: Credencial rechazada por el servidor SAI | "
            f"URL: {url}"
        )
        error_message = self._build_auth_error_message(auth_method)
        raise SAIAuthenticationError(error_message)

    def _handle_http_429_error(self, resp, auth_method: str, request_id: str) -> None:
        """Maneja errores HTTP 429 Rate Limit lanzando SAIRateLimitError."""
        response_text = resp.text if resp else ""

        if "Test template usage limit exceeded" in response_text:
            logger.warning(
                f"⚠️ [{request_id}] [HTTP 429] Rate Limit - Test Template | "
                f"Auth usado: {auth_method} | "
                f"Diagnóstico: Límite de uso de template de prueba excedido"
            )
        else:
            logger.error(
                f"❌ [{request_id}] [HTTP 429] Rate Limit - Otro tipo | "
                f"Auth usado: {auth_method} | "
                f"Respuesta del servidor: {response_text[:200]}"
            )

        raise SAIRateLimitError("Se ha excedido el límite de uso de la API de SAI.")

    def _handle_http_500_error(self, resp, auth_method: str, request_id: str) -> None:
        """Maneja errores HTTP 500 Internal Server Error lanzando excepciones específicas."""
        response_text = resp.text if resp else ""
        lower_text = response_text.lower()

        if "prompt is too long" in lower_text or "openaicompatible" in lower_text:
            logger.warning(
                f"⚠️ [{request_id}] [HTTP 500] Prompt Too Long | "
                f"Auth usado: {auth_method} | "
                f"Diagnóstico: El contexto excede el límite del modelo | "
                f"Respuesta SAI (preview): {response_text[:200]}"
            )
            raise SAIPromptTooLongError(
                "El contexto o historial enviado a SAI excede el límite soportado por el modelo."
            )

        logger.error(
            f"❌ [{request_id}] [HTTP 500] Internal Server Error | "
            f"Auth usado: {auth_method} | "
            f"Diagnóstico: Error interno del servidor SAI (no relacionado con tamaño de prompt) | "
            f"Respuesta SAI (preview): {response_text[:200]}"
        )
        raise SAIServerError(
            "Error interno del servidor SAI al procesar la solicitud."
        )

    def _handle_other_http_errors(self, resp, auth_method: str, url: str, e: Exception, request_id: str) -> None:
        """Maneja otros errores HTTP no específicos lanzando SAIAPIError."""
        status_code = resp.status_code if resp else "N/A"
        response_text = resp.text[:200] if resp else ""

        logger.error(
            f"❌ [{request_id}] [HTTP {status_code}] Error HTTP no manejado específicamente | "
            f"Auth usado: {auth_method} | "
            f"URL: {url} | "
            f"Exception: {type(e).__name__}: {str(e)} | "
            f"Respuesta del servidor: {response_text}"
        )
        raise SAIAPIError(
            f"Error HTTP {status_code} al llamar a SAI: {response_text}"
        )

    def _handle_request_exceptions(
        self,
        e: Exception,
        resp,
        auth_method: str,
        url: str,
        request_timeout: int,
        request_id: str,
    ) -> None:
        """Traduce excepciones de requests a excepciones de dominio SAI."""
        if isinstance(e, requests.HTTPError):
            if resp is not None and resp.status_code == 401:
                self._handle_http_401_error(resp, auth_method, url, request_id)

            if resp is not None and resp.status_code == 429:
                self._handle_http_429_error(resp, auth_method, request_id)

            if resp is not None and resp.status_code == 500:
                self._handle_http_500_error(resp, auth_method, request_id)

            self._handle_other_http_errors(resp, auth_method, url, e, request_id)

        elif isinstance(e, requests.Timeout):
            logger.error(
                f"⏱️ [{request_id}] [TIMEOUT] Tiempo de espera agotado | "
                f"Timeout configurado: {request_timeout}s | "
                f"Auth usado: {auth_method} | "
                f"URL: {url} | "
                f"Diagnóstico: El servidor SAI no respondió en el tiempo esperado"
            )
            raise SAITimeoutError(
                f"Timeout al llamar a SAI tras {request_timeout} segundos."
            )

        elif isinstance(e, requests.RequestException):
            logger.error(
                f"❌ [{request_id}] [NETWORK ERROR] Error de conexión | "
                f"Auth usado: {auth_method} | "
                f"URL: {url} | "
                f"Exception: {type(e).__name__}: {str(e)} | "
                f"Diagnóstico: Problema de red o conectividad con SAI"
            )
            raise SAIConnectionError(
                f"Error de red al intentar conectar con SAI: {type(e).__name__}: {str(e)}"
            )

        else:
            logger.error(
                f"❌ [{request_id}] [ERROR] Excepción inesperada al llamar a SAI | "
                f"Tipo: {type(e).__name__}: {str(e)}"
            )
            raise SAIAPIError(str(e))

    # ---------------- Llamada privada a SAI (refactorizada) ----------------
    def _call_sai(self, system: str, user: str, tool_call_id: Optional[str],
                  chat_messages: list, request_id: str, type: str, api: str,
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
                "tools": None,  # tools definitions (se llenará después)
                "type": type,
                "api": api
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
                f"Historial: {len(chat_messages)} mensajes | "
                f"Template: {SAI_TEMPLATE_ID} | "
                f"Auth: {auth_type}"
            )

        if VERBOSE_LOGGING:
            logger.debug(f"[{request_id}] Payload completo:\n{json.dumps(data, indent=2, ensure_ascii=False)}")

        # Ejecutar request con reintentos (puede lanzar SAI*Error en caso de fallo)
        response, response_headers, auth_method_used = self._execute_request_with_retry(
            url, data, custom_cookie, api_key_to_use, user_api_key, request_id
        )

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
            # Traducir y relanzar como excepción de dominio
            self._handle_request_exceptions(e, resp, auth_method, url, request_timeout, request_id)
            # La línea siguiente no debería alcanzarse nunca, pero se deja por claridad tipada.
            raise


# ---------------- Instancia global ----------------
sai_llm = SAILLM()
"""Instancia global de SAILLM lista para usar."""

logger.info(
    f"✅ SAILLM inicializado correctamente | "
    f"Clase: {sai_llm.__class__.__name__} | "
    f"Métodos disponibles: completion, acompletion, astreaming | "
    f"Estado: Listo para recibir peticiones"
)
