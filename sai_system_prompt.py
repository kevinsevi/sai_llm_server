# sai_system_prompt.py

"""
Módulo para gestionar system prompts adicionales desde archivos separados por endpoint.
"""
import os
from pathlib import Path
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

EndpointType = Literal["responses", "chat_completions"]


class SystemPromptManager:
    """Gestiona la carga y concatenación de system prompts adicionales por endpoint."""

    # Rutas por defecto para cada endpoint
    DEFAULT_PATHS = {
        "responses": "system_prompt_responses.txt",
        "chat_completions": "system_prompt_chat_completions.txt"
    }

    # Variables de entorno para rutas personalizadas
    ENV_VARS = {
        "responses": "SAI_SYSTEM_PROMPT_RESPONSES_FILE",
        "chat_completions": "SAI_SYSTEM_PROMPT_CHAT_COMPLETIONS_FILE"
    }

    def __init__(self):
        """Inicializa el gestor de system prompts."""
        # Cache de prompts por endpoint
        self._prompts: dict[EndpointType, Optional[str]] = {
            "responses": None,
            "chat_completions": None
        }
        # Timestamps de última modificación
        self._last_modified: dict[EndpointType, Optional[float]] = {
            "responses": None,
            "chat_completions": None
        }

    def _get_file_path(self, endpoint: EndpointType) -> str:
        """
        Obtiene la ruta del archivo para un endpoint específico.

        Args:
            endpoint: Tipo de endpoint ("responses" o "chat_completions")

        Returns:
            Ruta al archivo de system prompt.
        """
        # Intentar obtener desde variable de entorno
        env_path = os.getenv(self.ENV_VARS[endpoint])
        if env_path:
            return env_path

        # Usar ruta por defecto
        return self.DEFAULT_PATHS[endpoint]

    def _load_prompt_from_file(self, endpoint: EndpointType) -> Optional[str]:
        """
        Carga el contenido del archivo de system prompt para un endpoint.

        Args:
            endpoint: Tipo de endpoint

        Returns:
            Contenido del archivo o None si no existe/error.
        """
        try:
            file_path = self._get_file_path(endpoint)
            prompt_path = Path(file_path)

            if not prompt_path.exists():
                logger.debug(
                    f"📄 [{endpoint}] Archivo de system prompt no encontrado: {file_path}"
                )
                return None

            # Verificar si el archivo ha sido modificado
            current_mtime = prompt_path.stat().st_mtime

            if (self._last_modified[endpoint] is None or
                    current_mtime > self._last_modified[endpoint]):

                with open(prompt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                self._prompts[endpoint] = content if content else None
                self._last_modified[endpoint] = current_mtime

                if self._prompts[endpoint]:
                    logger.info(
                        f"✅ [{endpoint}] System prompt adicional cargado | "
                        f"Archivo: {file_path} | "
                        f"Tamaño: {len(content)} caracteres"
                    )
                else:
                    logger.warning(
                        f"⚠️ [{endpoint}] Archivo de system prompt vacío: {file_path}"
                    )

            return self._prompts[endpoint]

        except Exception as e:
            logger.error(
                f"❌ [{endpoint}] Error al cargar system prompt desde {file_path}: {e}"
            )
            return None

    def get_additional_prompt(self, endpoint: EndpointType) -> Optional[str]:
        """
        Obtiene el system prompt adicional para un endpoint específico.
        Recarga automáticamente si el archivo cambió.

        Args:
            endpoint: Tipo de endpoint

        Returns:
            System prompt adicional o None si no está disponible.
        """
        return self._load_prompt_from_file(endpoint)

    def merge_system_prompts(
            self,
            base_prompt: str,
            endpoint: EndpointType,
            separator: str = "\n\n"
    ) -> str:
        """
        Concatena el system prompt base con el adicional del endpoint.

        Args:
            base_prompt: System prompt principal.
            endpoint: Tipo de endpoint.
            separator: Separador entre prompts (por defecto: doble salto de línea).

        Returns:
            System prompt combinado.
        """
        additional = self.get_additional_prompt(endpoint)

        if not additional:
            return base_prompt

        return f"{base_prompt}{separator}{additional}"


# Instancia global (singleton)
_system_prompt_manager: Optional[SystemPromptManager] = None


def get_system_prompt_manager() -> SystemPromptManager:
    """
    Obtiene la instancia global del gestor de system prompts.

    Returns:
        Instancia de SystemPromptManager.
    """
    global _system_prompt_manager

    if _system_prompt_manager is None:
        _system_prompt_manager = SystemPromptManager()

    return _system_prompt_manager


def merge_system_prompts_for_responses(base_prompt: str) -> str:
    """
    Helper para concatenar system prompts en el endpoint /responses.

    Args:
        base_prompt: System prompt principal.

    Returns:
        System prompt combinado.
    """
    manager = get_system_prompt_manager()
    return manager.merge_system_prompts(base_prompt, "responses")


def merge_system_prompts_for_chat_completions(base_prompt: str) -> str:
    """
    Helper para concatenar system prompts en el endpoint /chat/completions.

    Args:
        base_prompt: System prompt principal.

    Returns:
        System prompt combinado.
    """
    manager = get_system_prompt_manager()
    return manager.merge_system_prompts(base_prompt, "chat_completions")