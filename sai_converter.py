# sai_converter.py

from sai_handler import logger
from sai_extractor import text_extractor

# ---------------- Conversor de formatos OpenAI ----------------
class OpenAiSAIConverter:
    @staticmethod
    def openai_to_litellm(openai_request: dict) -> tuple[list, dict]:
        messages = []

        # Extraer system prompt si existe
        system_prompt = openai_request.get("system", "")
        instructions = openai_request.get("instructions")
        system_text = None
        if system_prompt:
            system_text = text_extractor.extract_text_recursive(system_prompt)

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
            text_content = text_extractor.extract_text_recursive(content)

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

# ---------------- Instancia global ----------------
converter = OpenAiSAIConverter()
"""Instancia global del conversor de formatos OpenAI ↔ SAI."""