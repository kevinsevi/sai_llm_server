# exceptions.py
"""Excepciones personalizadas para SAI Handler."""

class SAIAPIError(Exception):
    """Error base para la API de SAI."""
    pass


class SAIRateLimitError(SAIAPIError):
    """Error de límite de tasa excedido."""
    pass


class SAIAuthenticationError(SAIAPIError):
    """Error de autenticación."""
    pass