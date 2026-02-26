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

        if messages and messages[0].get("role") in "system":
            system_prompt = messages[0].get("content")
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
        type_prompt = ""  # ← INICIALIZAR AQUÍ para evitar UnboundLocalError
        last_user_idx = -1
        
        # Buscar el último mensaje de usuario (ignorando mensajes tool)
        for idx in range(len(processed_messages) - 1, -1, -1):
            msg = processed_messages[idx]
            if msg.get("role") != "tool":
                user_prompt = msg.get("content", "")
                type_prompt = msg.get("type", "")
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
            
            # Los mensajes tool que NO son el último se agregan al historial como user
            if role == "tool":
                tc_id = msg.get("tool_call_id")
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

        return system_prompt, user_prompt, tool_prompt, tool_call_id, chat_messages, type_prompt

    def _normalize_tool_calls(self, tool_calls, request_id: str) -> Optional[list]:
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

    def _try_parse_from(self, start_idx: int, request_id: str, trimmed, decoder) -> tuple[Optional[list], Optional[int], Optional[int], Optional[str]]:
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

        tool_calls = self._normalize_tool_calls(obj.get("tool_calls"), request_id)
        if not tool_calls:
            return None, None, None, None

        json_raw = candidate[:end]
        return tool_calls, start_idx, start_idx + end, json_raw

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

        # NUEVO: Detectar formato {"cmd": ["tool_name", "args"]} de SAI
        # CRÍTICO: Buscar el patrón {"cmd": específicamente, no cualquier {
        import re

        # Buscar el último {"cmd": en el texto
        cmd_marker = '{"cmd":'
        last_cmd_idx = trimmed.rfind(cmd_marker)

        if last_cmd_idx != -1:
            # Intentar parsear desde ese punto hasta el final
            candidate = trimmed[last_cmd_idx:]
            try:
                # Parsear el JSON completo
                obj = json.loads(candidate)

                # Verificar si tiene la estructura {"cmd": [...]}
                if isinstance(obj, dict) and "cmd" in obj:
                    cmd_value = obj["cmd"]
                    if isinstance(cmd_value, list) and len(cmd_value) >= 2:
                        tool_name = cmd_value[0]
                        raw_args = cmd_value[1]

                        logger.info(
                            f"🔧 [{request_id}] [TOOLS] Detectado formato SAI cmd | "
                            f"tool_name={tool_name} | "
                            f"args_len={len(str(raw_args))} | "
                            f"args_preview={str(raw_args)[:100]!r}"
                        )

                        # Construir arguments según el tool
                        if tool_name == "apply_patch":
                            arguments = {"patch": raw_args}
                        elif tool_name == "exec_command":
                            arguments = {"command": raw_args}
                        else:
                            arguments = {"input": raw_args}

                        tool_calls = [{
                            "id": f"call_{request_id}_0",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments
                            }
                        }]

                        # Remover el JSON del texto
                        cleaned = trimmed[:last_cmd_idx].rstrip()

                        logger.info(
                            f"🔧 [{request_id}] [TOOLS] AFTER extracted (SAI cmd format) | "
                            f"tool_name={tool_name} | "
                            f"cleaned_len={len(cleaned)} | "
                            f"json_removed_len={len(candidate)} | "
                            f"arguments_keys={list(arguments.keys())}"
                        )

                        return tool_calls, cleaned
            except json.JSONDecodeError as e:
                logger.warning(
                    f"⚠️ [{request_id}] [TOOLS] JSON inválido después de {cmd_marker} | "
                    f"error={str(e)} | "
                    f"candidate_preview={candidate[:200]!r}"
                )

        # NUEVA LÓGICA: Detectar y extraer JSON dentro de bloques markdown ```json ... ```
        markdown_json_pattern = r'```json\s*\n(.+?)\n```\s*$'
        markdown_match = re.search(markdown_json_pattern, trimmed, re.DOTALL)
        
        if markdown_match:
            json_content = markdown_match.group(1).strip()
            logger.info(
                f"🧪 [{request_id}] [TOOLS] Detectado JSON en bloque markdown | "
                f"json_len={len(json_content)} | "
                f"preview={json_content[:100]!r}"
            )
            
            try:
                # Intentar parsear el JSON extraído
                obj = json.loads(json_content)
                
                # Verificar si es un dict válido
                if isinstance(obj, dict):
                    # Caso 1: El JSON tiene 'name' y opcionalmente 'parameters'
                    if "name" in obj:
                        tool_name = obj["name"]
                        # Si tiene 'parameters', usarlo; sino, usar el objeto completo sin 'name'
                        if "parameters" in obj:
                            arguments = obj["parameters"]
                        else:
                            # Remover 'name' y usar el resto como arguments
                            arguments = {k: v for k, v in obj.items() if k != "name"}
                    else:
                        # Caso 2: No tiene 'name' - asumir que TODO el JSON son los parámetros
                        tool_name = "exec_command"
                        arguments = obj  # El JSON completo son los argumentos
                    
                    # Convertir formato compacto a formato OpenAI
                    # IMPORTANTE: arguments debe ser dict, NO string JSON
                    tool_calls = [{
                        "id": f"call_{request_id}_0",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments  # ← Dict directo, NO json.dumps()
                        }
                    }]
                
                    # Remover el bloque markdown del texto
                    cleaned = trimmed[:markdown_match.start()].rstrip()
                
                    after_tail = cleaned[-500:] if len(cleaned) > 500 else cleaned
                    
                    # Log diferenciado según si se usó el default
                    if "name" not in obj:
                        logger.info(
                            f"🔧 [{request_id}] [TOOLS] AFTER extracted (markdown) | "
                            f"Campo 'name' ausente → usando default 'exec_command' | "
                            f"cleaned_len={len(cleaned)} | "
                            f"tool_calls_count=1 | "
                            f"arguments_keys={list(arguments.keys()) if isinstance(arguments, dict) else 'N/A'} | "
                            f"json_removed_len={len(markdown_match.group(0))} | "
                            f"cleaned_tail_preview={after_tail!r}"
                        )
                    else:
                        logger.info(
                            f"🧪 [{request_id}] [TOOLS] AFTER extracted (markdown) | "
                            f"cleaned_len={len(cleaned)} | "
                            f"tool_calls_count=1 | "
                            f"tool_name={tool_name} | "
                            f"arguments_keys={list(arguments.keys()) if isinstance(arguments, dict) else 'N/A'} | "
                            f"json_removed_len={len(markdown_match.group(0))} | "
                            f"cleaned_tail_preview={after_tail!r}"
                        )
                
                    return tool_calls, cleaned
                else:
                    # JSON válido pero no es un dict
                    logger.warning(
                        f"⚠️ [{request_id}] [TOOLS] JSON en markdown válido pero no es dict | "
                        f"Tipo: {type(obj).__name__} | "
                        f"Se retorna texto sin modificar"
                    )
                    return None, text
                    
            except json.JSONDecodeError as e:
                # JSON inválido en bloque markdown - ERROR CRÍTICO
                logger.error(
                    f"❌ [{request_id}] [TOOLS] JSON en markdown INVÁLIDO | "
                    f"error={str(e)} | "
                    f"json_preview={json_content[:200]!r} | "
                    f"Acción: Retornando texto sin modificar (NO se ejecutará fallback scan)"
                )
                # CAMBIO CRÍTICO: Retornar inmediatamente, NO continuar con fallback
                return None, text

        # NUEVO: Detectar JSON plano al final (sin markdown, sin wrapper tool_calls)
        if trimmed.endswith("}"):
            # MEJORA: Limpiar saltos de línea antes del JSON
            # Buscar el último bloque JSON válido, ignorando \n antes de él
            brace_positions = [i for i, ch in enumerate(trimmed) if ch == "{"]
            
            for start_idx in reversed(brace_positions):
                # Verificar si hay solo whitespace/newlines antes del '{'
                prefix = trimmed[:start_idx]
                # Si hay texto significativo muy cerca del JSON, no es un tool call aislado
                if prefix and not prefix[-20:].strip().endswith(('.', '!', '?', '\n', ':')):
                    continue

                candidate = trimmed[start_idx:]
                try:
                    obj = json.loads(candidate)
                    
                    # Verificar que sea un dict válido
                    if not isinstance(obj, dict):
                        continue
                    
                    # Verificar que el JSON consuma todo el final (sin texto después)
                    if trimmed[start_idx + len(json.dumps(obj, ensure_ascii=False)):].strip():
                        continue

                    # Caso 1: Tiene 'name' → es un tool call directo
                    if "name" in obj:
                        tool_name = obj["name"]
                        arguments = obj.get("parameters", {k: v for k, v in obj.items() if k != "name"})
                    # Caso 2: Tiene 'cmd' o 'command' → es exec_command
                    elif "cmd" in obj or "command" in obj:
                        tool_name = "exec_command"
                        arguments = obj
                    # Caso 3: Tiene 'path' o 'file' → podría ser apply_patch
                    elif "path" in obj or "file" in obj or "patch" in obj:
                        tool_name = "apply_patch"
                        arguments = obj
                    else:
                        # Caso 4: No tiene 'name' → asumir exec_command por defecto
                        tool_name = "exec_command"
                        arguments = obj

                    # Construir tool_call
                    tool_calls = [{
                        "id": f"call_{request_id}_0",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments
                        }
                    }]

                    # Remover el JSON del texto (y los \n antes de él)
                    cleaned = trimmed[:start_idx].rstrip()

                    logger.info(
                        f"🔧 [{request_id}] [TOOLS] AFTER extracted (plain JSON) | "
                        f"tool_name={tool_name} | "
                        f"cleaned_len={len(cleaned)} | "
                        f"json_removed_len={len(candidate)} | "
                        f"arguments_keys={list(arguments.keys()) if isinstance(arguments, dict) else 'N/A'}"
                    )

                    return tool_calls, cleaned
                    
                except json.JSONDecodeError:
                    continue

        # Lógica original: detectar JSON sin markdown (solo si NO hay bloque markdown)
        if not trimmed.endswith("}"):
            logger.info(f"[{request_id}] [TOOLS] extractor: no termina en '}}' -> no hay JSON final")
            return None, text

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

        # 2) Fallback: escanear todos los '{' hacia atrás (evita caer en '{' dentro de strings)
        brace_positions = [i for i, ch in enumerate(trimmed) if ch == "{"]

        logger.info(f"🧪 [{request_id}] [TOOLS] fallback scan: brace_positions={len(brace_positions)}")

        for start_idx in reversed(brace_positions):
            tool_calls, js, je, json_raw = self._try_parse_from(start_idx, request_id, trimmed, decoder)
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

        system, user_prompt, tool_prompt, tool_call_id, chat_messages, type_prompt = self._prepare_messages(messages, request_id)

        response_text, finish_reason, usage_data = self._call_sai(
            system,
            user_prompt,
            tool_prompt,
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

        system, user_prompt, tool_prompt, tool_call_id, chat_messages, type_prompt = self._prepare_messages(messages, request_id)

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
                "tool": tool if tool else None,  # Agregar tool message
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

logger.info(
    f"✅ SAILLM inicializado correctamente | "
    f"Clase: {sai_llm.__class__.__name__} | "
    f"Métodos disponibles: completion, acompletion, astreaming | "
    f"Estado: Listo para recibir peticiones"
)