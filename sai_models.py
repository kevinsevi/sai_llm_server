"""
sai_models.py - Módulo compartido para la definición de modelos disponibles en SAI.

Este módulo es usado tanto por sai_handler.py como por sai_gateway.py
para mantener una única fuente de verdad sobre los modelos soportados.
"""

# Lista de modelos disponibles en SAI
# Nota: el "id" es el que expone la OpenAI Models API (/v1/models).
# El identificador interno/provider (p.ej. "OpenAIAPI/sai-model") se puede guardar en metadata.
AVAILABLE_MODELS = [
    {
        "id": "claude-sonnet-4-5-20250929",
        "object": "model",
        "created": 1735689600,  # placeholder estable (UTC)
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-sonnet-4-5-20250929",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/claude-sonnet-4-5-20250929"
        }
    },
    {
        "id": "gpt-5.2-2025-12-11",
        "object": "model",
        "created": 1735689600,  # placeholder estable (UTC)
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5.2-2025-12-11",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/gpt-5.2-2025-12-11"
        }
    }
]

# Diccionario para búsqueda rápida por ID
MODELS_DICT = {model["id"]: model for model in AVAILABLE_MODELS}


def get_models_list() -> dict:
    """
    Retorna la lista de modelos en formato OpenAI Models API.

    Returns:
        dict: Respuesta en formato OpenAI con la lista de modelos
    """
    return {
        "object": "list",
        "data": AVAILABLE_MODELS
    }


def get_model_by_id(model_id: str) -> dict | None:
    """
    Busca un modelo específico por su ID.

    Args:
        model_id: ID del modelo a buscar

    Returns:
        dict | None: Información del modelo o None si no existe
    """
    return MODELS_DICT.get(model_id)


def is_valid_model(model_id: str) -> bool:
    """
    Verifica si un modelo ID es válido.

    Args:
        model_id: ID del modelo a verificar

    Returns:
        bool: True si el modelo existe, False en caso contrario
    """
    return model_id in MODELS_DICT