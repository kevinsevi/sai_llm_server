"""Excepciones personalizadas para el gateway SAI."""


class SAIAPIError(Exception):
    """Error base genérico al interactuar con SAI."""
    pass


class SAIAuthenticationError(SAIAPIError):
    """Error de autenticación (credenciales inválidas o expiradas)."""
    pass


class SAIRateLimitError(SAIAPIError):
    """Error de límite de tasa excedido."""
    pass


class SAIPromptTooLongError(SAIAPIError):
    """El contexto o prompt excede el límite soportado por el modelo."""
    pass


class SAIServerError(SAIAPIError):
    """Error interno del servidor SAI (HTTP 5xx no relacionado con longitud de prompt)."""
    pass


class SAIConnectionError(SAIAPIError):
    """Problema de red o conectividad con SAI."""
    pass


class SAITimeoutError(SAIConnectionError):
    """Timeout al esperar respuesta de SAI."""
    pass
