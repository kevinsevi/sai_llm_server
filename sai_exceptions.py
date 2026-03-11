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


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida o malformada recibida desde el upstream (SAI)."""

    def __init__(self, message: str, *, upstream_preview: str = "", exception: str = ""):
        super().__init__(message)
        self.upstream_preview = upstream_preview
        self.exception = exception


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida/malformada proveniente del upstream (SAI)."""

    def __init__(self, message: str, *, upstream_preview: str = "", exception: str = ""):
        super().__init__(message)
        self.upstream_preview = upstream_preview
        self.exception = exception


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida/malformada proveniente del upstream (SAI)."""

    def __init__(self, message: str, *, upstream_preview: str = "", exception: str = ""):
        super().__init__(message)
        self.upstream_preview = upstream_preview
        self.exception = exception


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida o no parseable proveniente del upstream (SAI)."""

    def __init__(self, message: str, *, upstream_preview: str = "", exception_type: str = ""):
        super().__init__(message)
        self.upstream_preview = upstream_preview
        self.exception_type = exception_type


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida o malformada recibida desde el upstream (SAI)."""

    def __init__(self, message: str, *, upstream_preview: str = "", exception_type: str = ""):
        super().__init__(message)
        self.upstream_preview = upstream_preview
        self.exception_type = exception_type


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida del upstream (p.ej. JSON truncado/malformado)."""

    def __init__(self, message: str, upstream_preview: str = "", request_id: str = ""):
        super().__init__(message)
        self.upstream_preview = upstream_preview
        self.request_id = request_id


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida/malformada recibida desde el upstream (SAI)."""

    def __init__(self, message: str, *, upstream_preview: str = "", details: dict | None = None):
        super().__init__(message)
        self.upstream_preview = upstream_preview
        self.details = details or {}


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida o no parseable proveniente del upstream (SAI)."""

    def __init__(self, message: str, *, upstream_preview: str = "", exception_type: str = ""):
        super().__init__(message)
        self.upstream_preview = upstream_preview
        self.exception_type = exception_type


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida o malformada recibida desde el upstream (SAI)."""

    def __init__(self, message: str, *, upstream_preview: str = "", exception_type: str = ""):
        super().__init__(message)
        self.upstream_preview = upstream_preview
        self.exception_type = exception_type


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida o malformada recibida desde SAI (upstream)."""
    pass


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida/malformada recibida desde el upstream (SAI)."""
    pass


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida o no parseable proveniente del upstream (SAI)."""
    pass


class SAIUpstreamInvalidResponseError(SAIAPIError):
    """Respuesta inválida o malformada recibida desde el upstream (SAI)."""
    pass
