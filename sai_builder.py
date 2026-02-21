# sai_builder.py

"""
Módulo para construcción de eventos SSE en formato OpenAI Responses API.
Contiene todos los métodos _build_* extraídos de sai_converter.py para mejor organización.
"""


class ResponseEventBuilder:
    """
    Clase que encapsula la construcción de eventos SSE para streaming
    en formato OpenAI Responses API.
    """

    @staticmethod
    def build_response_created_event(
            response_id: str,
            created_at: int,
            kwargs: dict,
            model: str,
            sequence_number: int = 0
    ) -> dict:
        """
        Construye el evento response.created para streaming SSE.

        Args:
            response_id: ID único de la respuesta
            created_at: Timestamp de creación
            kwargs: Parámetros de configuración del modelo
            model: Nombre del modelo
            sequence_number: Número de secuencia del evento (default: 0)

        Returns:
            dict: Evento response.created en formato OpenAI Responses API
        """
        return {
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
            "sequence_number": sequence_number
        }

    @staticmethod
    def build_response_in_progress_event(
            response_id: str,
            created_at: int,
            kwargs: dict,
            model: str,
            sequence_number: int = 1
    ) -> dict:
        """
        Construye el evento response.in_progress para streaming SSE.

        Args:
            response_id: ID único de la respuesta
            created_at: Timestamp de creación
            kwargs: Parámetros de configuración del modelo
            model: Nombre del modelo
            sequence_number: Número de secuencia del evento (default: 1)

        Returns:
            dict: Evento response.in_progress en formato OpenAI Responses API
        """
        return {
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
            "sequence_number": sequence_number
        }

    @staticmethod
    def build_response_output_item_added_event(
            item_id: str,
            item_type: str = "message",
            has_content: bool = True,
            call_id: str = None,
            name: str = None
    ) -> dict:
        """
        Construye el evento response.output_item.added para streaming SSE.

        Args:
            item_id: ID único del item de salida
            item_type: Tipo de item ("message", "reasoning", "function_call")
            has_content: Si el item debe incluir contenido inicial vacío (default: True)
            call_id: ID de la llamada para function_call (opcional)
            name: Nombre de la función para function_call (opcional)

        Returns:
            dict: Evento response.output_item.added en formato OpenAI Responses API
        """
        item = {
            "id": item_id,
            "type": item_type
        }

        # Para message y reasoning, agregar role y content
        if item_type in ["message", "reasoning"]:
            item["role"] = "assistant"
            if has_content:
                item["content"] = [
                    {
                        "type": "output_text",
                        "text": ""
                    }
                ]

        # Para function_call, agregar campos específicos
        elif item_type == "function_call":
            item.update({
                "status": "in_progress",
                "arguments": "",
                "call_id": call_id or "",
                "name": name or ""
            })

        return {
            "type": "response.output_item.added",
            "item": item
        }

    @staticmethod
    def build_response_output_item_done_event(
            item_id: str,
            item_type: str = "message",
            text: str = "",
            output_index: int = 0,
            sequence_number: int = 7,
            function_call_data: dict = None
    ) -> dict:
        """
        Construye el evento response.output_item.done para streaming SSE.

        Args:
            item_id: ID único del item de salida
            item_type: Tipo de item ("message", "reasoning", "function_call")
            text: Texto completo del mensaje (para message/reasoning)
            output_index: Índice del output en la respuesta (default: 0)
            sequence_number: Número de secuencia del evento (default: 7)
            function_call_data: Datos del function_call si aplica (dict con keys: call_id, name, arguments, status)

        Returns:
            dict: Evento response.output_item.done en formato OpenAI Responses API
        """
        # Estructura base del evento
        event = {
            "type": "response.output_item.done",
            "output_index": output_index,
            "sequence_number": sequence_number
        }

        # Construir el item según el tipo
        item = {
            "id": item_id,
            "type": item_type,
            "status": "completed"
        }

        # Construir según el tipo de item
        if item_type in ["message", "reasoning"]:
            item["role"] = "assistant"
            item["content"] = [
                {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": text
                }
            ]
        elif item_type == "function_call":
            # Soporte completo para function_call
            if function_call_data:
                item.update({
                    "arguments": function_call_data.get("arguments", "{}"),
                    "call_id": function_call_data.get("call_id", ""),
                    "name": function_call_data.get("name", ""),
                    "status": function_call_data.get("status", "completed")
                })
            else:
                # Valores por defecto si no se proporciona function_call_data
                item.update({
                    "arguments": "{}",
                    "call_id": "",
                    "name": "",
                    "status": "completed"
                })

        # Agregar el item al evento
        event["item"] = item

        return event

    @staticmethod
    def build_response_content_part_added_event(
            item_id: str,
            part_type: str = "output_text",
            text: str = "",
            index: int = 0
    ) -> dict:
        """
        Construye el evento response.content_part.added para streaming SSE.

        Args:
            item_id: ID del item al que pertenece este content part
            part_type: Tipo de contenido ("output_text", "audio", etc.)
            text: Texto inicial del contenido (default: "")
            index: Índice del content part (default: 0)

        Returns:
            dict: Evento response.content_part.added en formato OpenAI Responses API
        """
        return {
            "type": "response.content_part.added",
            "item_id": item_id,
            "part": {
                "type": part_type,
                "text": text
            },
            "index": index
        }

    @staticmethod
    def build_response_function_call_started_event(
            item_id: str,
            call_id: str,
            name: str
    ) -> dict:
        """
        Construye el evento response.function_call.started para streaming SSE.

        Args:
            item_id: ID del item de salida
            call_id: ID único de la llamada a función
            name: Nombre de la función que se está llamando

        Returns:
            dict: Evento response.function_call.started en formato OpenAI Responses API
        """
        return {
            "type": "response.function_call.started",
            "item_id": item_id,
            "call_id": call_id,
            "name": name
        }

    @staticmethod
    def build_response_function_call_arguments_delta_event(
            item_id: str,
            delta: str,
            call_id: str = None
    ) -> dict:
        """
        Construye el evento response.function_call.arguments.delta para streaming SSE.

        Args:
            item_id: ID del item de salida
            delta: Fragmento incremental de los argumentos JSON
            call_id: ID único de la llamada a función (opcional)

        Returns:
            dict: Evento response.function_call.arguments.delta en formato OpenAI Responses API
        """
        event = {
            "type": "response.function_call.arguments.delta",
            "item_id": item_id,
            "delta": delta
        }

        # Solo agregar call_id si se proporciona
        if call_id is not None:
            event["call_id"] = call_id

        return event

    @staticmethod
    def build_response_output_text_done_event(
            item_id: str,
            text: str,
            index: int = 0
    ) -> dict:
        """
        Construye el evento response.output_text.done para streaming SSE.

        Args:
            item_id: ID del item de salida al que pertenece este texto
            text: Texto completo generado
            index: Índice del content part (default: 0)

        Returns:
            dict: Evento response.output_text.done en formato OpenAI Responses API
        """
        return {
            "type": "response.output_text.done",
            "item_id": item_id,
            "index": index,
            "text": text
        }

    @staticmethod
    def build_response_function_call_arguments_done_event(
            arguments: str,
            item_id: str,
            output_index: int = 1,
            sequence_number: int = None
    ) -> dict:
        """
        Construye el evento response.function_call_arguments.done para streaming SSE.

        Args:
            arguments: Argumentos JSON completos de la función (o texto plano)
            item_id: ID del item de salida (function_call)
            output_index: Índice del output en la respuesta (default: 1)
            sequence_number: Número de secuencia del evento (opcional)

        Returns:
            dict: Evento response.function_call_arguments.done en formato OpenAI Responses API
        """
        event = {
            "type": "response.function_call_arguments.done",
            "arguments": arguments,
            "item_id": item_id,
            "output_index": output_index
        }

        # Solo agregar sequence_number si se proporciona
        if sequence_number is not None:
            event["sequence_number"] = sequence_number

        return event

    @staticmethod
    def build_response_function_call_completed_event(
            item_id: str,
            call_id: str,
            name: str,
            arguments: str
    ) -> dict:
        """
        Construye el evento response.function_call.completed para streaming SSE.

        Args:
            item_id: ID del item de salida
            call_id: ID único de la llamada a función
            name: Nombre de la función ejecutada
            arguments: Argumentos JSON completos de la función

        Returns:
            dict: Evento response.function_call.completed en formato OpenAI Responses API
        """
        return {
            "type": "response.function_call.completed",
            "item_id": item_id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments
        }

    @staticmethod
    def build_response_output_text_delta_event(
            item_id: str,
            delta: str
    ) -> dict:
        """
        Construye el evento response.output_text.delta para streaming SSE.

        Args:
            item_id: ID del item de salida al que pertenece este delta
            delta: Fragmento incremental de texto

        Returns:
            dict: Evento response.output_text.delta en formato OpenAI Responses API
        """
        return {
            "type": "response.output_text.delta",
            "item_id": item_id,
            "delta": delta
        }

    @staticmethod
    def build_response_content_part_done_event(
            item_id: str,
            text: str,
            index: int = 0,
            part_type: str = "output_text"
    ) -> dict:
        """
        Construye el evento response.content_part.done para streaming SSE.

        Args:
            item_id: ID del item al que pertenece este content part
            text: Texto completo del contenido
            index: Índice del content part (default: 0)
            part_type: Tipo de contenido (default: "output_text")

        Returns:
            dict: Evento response.content_part.done en formato OpenAI Responses API
        """
        return {
            "type": "response.content_part.done",
            "item_id": item_id,
            "index": index,
            "part": {
                "type": part_type,
                "text": text
            }
        }

    @staticmethod
    def build_response_completed_event(
            response_id: str,
            created_at: int,
            completed_at: int,
            kwargs: dict,
            model: str,
            output: list,
            input_tokens: int,
            output_tokens: int,
            sequence_number: int = 8,
            billing: dict = None,
            instructions: str = None
    ) -> dict:
        """
        Construye el evento response.completed para streaming SSE.

        Args:
            response_id: ID único de la respuesta
            created_at: Timestamp de creación
            completed_at: Timestamp de finalización
            kwargs: Parámetros de configuración del modelo
            model: Nombre del modelo
            output: Lista de items de salida (messages, function_calls, etc.)
            input_tokens: Tokens de entrada consumidos
            output_tokens: Tokens de salida generados
            sequence_number: Número de secuencia del evento (default: 8)
            billing: Información de facturación (opcional)
            instructions: Instrucciones del sistema (opcional, sobrescribe kwargs)

        Returns:
            dict: Evento response.completed en formato OpenAI Responses API
        """
        # Usar billing proporcionado o valor por defecto
        billing_info = billing if billing is not None else {"payer": "developer"}

        # Usar instructions proporcionado o del kwargs
        final_instructions = instructions if instructions is not None else kwargs.get("instructions")

        return {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "completed",
                "background": False,
                "billing": billing_info,
                "completed_at": completed_at,
                "error": None,
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "incomplete_details": None,
                "instructions": final_instructions,
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
            "sequence_number": sequence_number
        }


# Instancia global del builder
event_builder = ResponseEventBuilder()
"""Instancia global del constructor de eventos SSE."""
