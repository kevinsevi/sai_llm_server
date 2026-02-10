"""
sai_gateway.py - Gateway FastAPI para exponer SAI con APIs compatibles con OpenAI.

Este servidor proporciona múltiples endpoints compatibles con diferentes
formatos de la API de OpenAI:

Endpoints principales:
- POST /v1/completions: OpenAI Completions API (legacy) con soporte streaming
- POST /v1/chat/completions: OpenAI Chat Completions API con soporte streaming
- POST /v1/messages: OpenAI Messages API (sin streaming)
- POST /v1/responses: OpenAI Responses API con streaming SSE (compatible con Codex CLI)
- GET /v1/models: Lista de modelos disponibles
- GET /v1/models/{model_id}: Detalles de un modelo específico
- GET /health: Health check

Características:
- Todos los endpoints funcionan con o sin el prefijo /v1
- Autenticación vía Authorization header (Bearer token) o x-api-key header
- Conversión automática entre formatos OpenAI y LiteLLM
- Logging detallado con request_id para trazabilidad
- Manejo robusto de errores con HTTPException

Versión: 1.0.4
"""

import json
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from sai_handler import sai_llm, logger, converter
from sai_models import get_models_list, get_model_by_id

app = FastAPI(title="OpenAI SAI Gateway", version="1.0.4")

def _normalize_bool(value) -> bool:
    """
    Normaliza un valor a booleano.
    
    Maneja múltiples tipos de entrada y convierte strings como "true"/"false"
    a sus equivalentes booleanos.

    Args:
        value: Valor a normalizar. Puede ser bool, str, int o None.
        
    Returns:
        bool: Valor normalizado a booleano.
        
    Examples:
        >>> _normalize_bool(True)
        True
        >>> _normalize_bool("true")
        True
        >>> _normalize_bool("1")
        True
        >>> _normalize_bool(0)
        False
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if isinstance(value, int):
        return value != 0
    return False


@app.post("/v1/completions")
@app.post("/completions")
async def completions_endpoint(request: Request):
    """
    Endpoint compatible con OpenAI Completions API (legacy).
    
    Acepta un prompt simple y retorna una completion. Soporta tanto
    modo streaming como no streaming.

    Request Body:
        prompt (str): Texto del prompt
        stream (bool, optional): Si es True, retorna streaming SSE. Default: False
        model (str, optional): ID del modelo. Default: "claude-sonnet-4-5-20250929"
        max_tokens (int, optional): Máximo de tokens a generar. Default: 4096
        temperature (float, optional): Temperatura de sampling
        top_p (float, optional): Nucleus sampling
        stop (str|list, optional): Secuencias de parada

    Headers:
        Authorization: Bearer token (opcional)
        x-api-key: API key alternativa (opcional)

    Returns:
        JSONResponse o StreamingResponse: Completion en formato OpenAI
        
    Raises:
        HTTPException: 400 si falta el prompt, 500 en errores internos
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        body = await request.json()
        logger.info(f"📨 [{request_id}] /v1/completions request recibido")
        logger.debug(f"📋 [{request_id}] Body completo: {json.dumps(body, indent=2)}")

        logger.debug(f"🔑 [{request_id}] Headers recibidos:")
        logger.debug(f"   - Authorization: {request.headers.get('authorization', 'NO PRESENTE')}")
        logger.debug(f"   - x-api-key: {request.headers.get('x-api-key', 'NO PRESENTE')}")

        # Extraer parámetros
        prompt = body.get("prompt", "")
        stream = _normalize_bool(body.get("stream", False))
        model = body.get("model", "claude-sonnet-4-5-20250929")
        max_tokens = body.get("max_tokens", 4096)
        temperature = body.get("temperature")
        top_p = body.get("top_p")
        stop = body.get("stop")

        logger.info(
            f"📊 [{request_id}] Model: {model} | "
            f"Prompt length: {len(prompt)} chars | "
            f"Stream: {stream}"
        )

        # Validar prompt
        if not prompt:
            logger.error(f"❌ [{request_id}] Request sin 'prompt'")
            raise HTTPException(
                status_code=400,
                detail="Se requiere 'prompt'"
            )

        # Convertir prompt a formato de mensajes para sai_handler
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Preparar kwargs para sai_llm
        kwargs = {
            "model": model,
            "max_tokens": max_tokens
        }

        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop is not None:
            kwargs["stop"] = stop

        # Extraer user_api_key si viene en headers
        user_api_key = request.headers.get("authorization", "").replace("Bearer ", "").strip()
        if not user_api_key:
            user_api_key = request.headers.get("x-api-key")

        if user_api_key:
            kwargs["headers"] = {"user_api_key": user_api_key}
            logger.info(f"🔑 [{request_id}] user_api_key detectada")

        # Decidir si es streaming o no
        if stream:
            logger.info(f"🌊 [{request_id}] Modo streaming activado")
            return await converter._completions_streaming(request_id, messages, kwargs, model)
        else:
            logger.info(f"📄 [{request_id}] Modo sin streaming")
            return await converter._completions_non_streaming(request_id, messages, kwargs, model)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Error en /v1/completions: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions_endpoint(request: Request):
    """
    Endpoint compatible con OpenAI Chat Completions API.
    
    Acepta una lista de mensajes en formato chat y retorna una completion.
    Soporta tanto modo streaming como no streaming.

    Request Body:
        messages (list): Lista de mensajes con 'role' y 'content'
        stream (bool, optional): Si es True, retorna streaming SSE. Default: False
        model (str, optional): ID del modelo. Default: "claude-sonnet-4-5-20250929"
        max_tokens (int, optional): Máximo de tokens a generar. Default: 4096
        temperature (float, optional): Temperatura de sampling
        top_p (float, optional): Nucleus sampling
        stop (str|list, optional): Secuencias de parada

    Headers:
        Authorization: Bearer token (opcional)
        x-api-key: API key alternativa (opcional)

    Returns:
        JSONResponse o StreamingResponse: Chat completion en formato OpenAI
        
    Raises:
        HTTPException: 400 si falta messages, 500 en errores internos
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        body = await request.json()
        logger.info(f"📨 [{request_id}] /v1/chat/completions request recibido")
        logger.debug(f"📋 [{request_id}] Body completo: {json.dumps(body, indent=2)}")

        logger.debug(f"🔑 [{request_id}] Headers recibidos:")
        logger.debug(f"   - Authorization: {request.headers.get('authorization', 'NO PRESENTE')}")
        logger.debug(f"   - x-api-key: {request.headers.get('x-api-key', 'NO PRESENTE')}")

        # Extraer parámetros de OpenAI format
        stream = _normalize_bool(body.get("stream", False))
        model = body.get("model", "claude-sonnet-4-5-20250929")
        messages_input = body.get("messages", [])
        tools = body.get("tools")  # Extraer tools si existen

        # Omitir tools si es una lista vacía
        if tools is not None and isinstance(tools, list) and len(tools) == 0:
            tools = None
            logger.info(f"🔧 [{request_id}] tools es lista vacía [] - omitiendo para evitar falsos positivos")

        logger.info(
            f"📊 [{request_id}] Model: {model} | "
            f"Messages: {len(messages_input)} | "
            f"Stream: {stream} | "
            f"Tools: {len(tools) if tools else 0}"
        )

        # Validar messages
        if not messages_input:
            logger.error(f"❌ [{request_id}] Request sin 'messages'")
            raise HTTPException(
                status_code=400,
                detail="Se requiere 'messages'"
            )

        # Convertir a formato interno
        openai_body = {
            "model": model,
            "messages": messages_input,
            "max_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "stop": body.get("stop")
        }

        # Agregar tools si existen
        if tools:
            openai_body["tools"] = tools

        messages, kwargs = converter.openai_to_litellm(openai_body)

        # Extraer user_api_key si viene en headers
        user_api_key = request.headers.get("authorization", "").replace("Bearer ", "").strip()
        if not user_api_key:
            user_api_key = request.headers.get("x-api-key")

        if user_api_key:
            kwargs["headers"] = {"user_api_key": user_api_key}
            logger.info(f"🔑 [{request_id}] user_api_key detectada")

        # Decidir si es streaming o no
        if stream:
            logger.info(f"🌊 [{request_id}] Modo streaming activado")
            return await converter._chat_completions_streaming(request_id, messages, kwargs, model)
        else:
            logger.info(f"📄 [{request_id}] Modo sin streaming")
            return await converter._chat_completions_non_streaming(request_id, messages, kwargs, model)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Error en /v1/chat/completions: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages")
@app.post("/messages")
async def messages_endpoint(request: Request):
    """
    Endpoint compatible con OpenAI Messages API (sin streaming).
    
    Acepta mensajes en formato OpenAI y retorna una respuesta completa
    sin streaming. Si necesitas streaming, usa /v1/responses.

    Request Body:
        messages (list): Lista de mensajes con 'role' y 'content'
        input (str, optional): Alternativa a messages para prompt simple
        model (str, optional): ID del modelo
        max_tokens (int, optional): Máximo de tokens a generar
        temperature (float, optional): Temperatura de sampling
        top_p (float, optional): Nucleus sampling

    Headers:
        Authorization: Bearer token (opcional)
        x-api-key: API key alternativa (opcional)

    Returns:
        JSONResponse: Respuesta completa en formato OpenAI Messages
        
    Raises:
        HTTPException: 400 si falta messages/input o si pide streaming,
                      500 en errores internos
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        body = await request.json()
        logger.info(f"📨 [{request_id}] /v1/messages request recibido")
        logger.debug(f"📋 [{request_id}] Body completo: {json.dumps(body, indent=2)}")

        logger.debug(f"🔑 [{request_id}] Headers recibidos:")
        logger.debug(f"   - Authorization: {request.headers.get('authorization', 'NO PRESENTE')}")
        logger.debug(f"   - x-api-key: {request.headers.get('x-api-key', 'NO PRESENTE')}")

        # Validar que no esté pidiendo streaming
        if _normalize_bool(body.get("stream", False)):
            raise HTTPException(
                status_code=400,
                detail="Para streaming, use el endpoint /v1/responses"
            )

        # Normalizar el formato
        if "input" in body and "messages" not in body:
            logger.info(f"🔄 [{request_id}] Formato OpenAI SDK detectado (input) - Convirtiendo")
            body["messages"] = [
                {
                    "role": "user",
                    "content": body["input"]
                }
            ]

        # Validar que tengamos messages
        if not body.get("messages"):
            logger.error(f"❌ [{request_id}] Request sin 'messages' ni 'input'")
            raise HTTPException(
                status_code=400,
                detail="Se requiere 'messages' (OpenAI format) o 'input' (OpenAI format)"
            )

        logger.info(f"📊 [{request_id}] Model: {body.get('model', 'N/A')} | Messages: {len(body.get('messages', []))}")

        # Convertir request de OpenAI a LiteLLM
        messages, kwargs = converter.openai_to_litellm(body)

        logger.info(
            f"🔄 [{request_id}] Convertido a formato LiteLLM | "
            f"Messages: {len(messages)} | "
            f"System: {'✓' if messages and messages[0]['role'] == 'system' else '✗'}"
        )

        # Extraer user_api_key
        user_api_key = request.headers.get("x-api-key")
        if not user_api_key:
            auth_header = request.headers.get("authorization", "")
            user_api_key = auth_header.replace("Bearer ", "").strip()

        if user_api_key:
            kwargs["headers"] = {"user_api_key": user_api_key}
            logger.info(f"🔑 [{request_id}] user_api_key detectada")

        # Llamar a SAI
        logger.info(f"🚀 [{request_id}] Llamando a sai_llm.acompletion()...")
        litellm_response = await sai_llm.acompletion(messages=messages, **kwargs)

        # Convertir respuesta a formato OpenAI
        model = body.get("model")
        openai_response = converter.litellm_to_openai_response(
            litellm_response, model, request_id
        )

        logger.info(
            f"✅ [{request_id}] Respuesta lista | "
            f"Output tokens: {openai_response['usage']['output_tokens']} | "
            f"Stop reason: {openai_response['stop_reason']}"
        )

        return JSONResponse(content=openai_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Error en /v1/messages: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/responses")
@app.post("/responses")
async def responses_endpoint(request: Request):
    """
    Endpoint compatible con OpenAI Responses API (con streaming SSE).
    
    Implementa el formato oficial de OpenAI Responses API utilizado por
    herramientas como Codex CLI. Emite eventos SSE durante el streaming.

    Eventos SSE emitidos:
        - response.created: Inicio del response
        - response.output_text.delta: Deltas de texto incrementales
        - response.output_item.done: Item de salida completado
        - response.done: Response finalizado (evento final correcto según spec)

    Request Body:
        messages (list): Lista de mensajes con 'role' y 'content'
        input (str, optional): Alternativa a messages para prompt simple
        stream (bool, optional): Si es True, retorna streaming SSE. Default: False
        model (str, optional): ID del modelo
        max_tokens (int, optional): Máximo de tokens a generar
        temperature (float, optional): Temperatura de sampling
        top_p (float, optional): Nucleus sampling

    Headers:
        Authorization: Bearer token (opcional)
        x-api-key: API key alternativa (opcional)

    Returns:
        StreamingResponse o JSONResponse: Response en formato OpenAI Responses API
        
    Raises:
        HTTPException: 400 si falta messages/input, 500 en errores internos
        
    Note:
        v1.0.4: Corregido evento final de 'response.completed' a 'response.done'
        según especificación oficial de OpenAI.
    """
    request_id = str(uuid.uuid4())[:8]
    response_id = f"msg_{request_id}"
    output_item_id = f"item_{request_id}"

    try:
        body = await request.json()

        logger.info(f"📨 [{request_id}] /v1/responses request recibido")
        logger.debug(f"📋 [{request_id}] Body completo: {json.dumps(body, indent=2)}")

        logger.debug(f"🔑 [{request_id}] Headers recibidos:")
        logger.debug(f"   - Authorization: {request.headers.get('authorization', 'NO PRESENTE')}")
        logger.debug(f"   - x-api-key: {request.headers.get('x-api-key', 'NO PRESENTE')}")

        # Extraer parámetro stream
        stream = _normalize_bool(body.get("stream", False))
        logger.info(f"🔀 [{request_id}] Modo: {'Streaming' if stream else 'No streaming'}")

        # Normalizar formato
        if "input" in body and "messages" not in body:
            logger.info(f"🔄 [{request_id}] Formato OpenAI SDK detectado (input) - Convirtiendo")
            body["messages"] = [
                {
                    "role": "user",
                    "content": body["input"]
                }
            ]

        # Validar messages
        if not body.get("messages"):
            logger.error(f"❌ [{request_id}] Request sin 'messages' ni 'input'")
            raise HTTPException(
                status_code=400,
                detail="Se requiere 'messages' (OpenAI format) o 'input' (OpenAI format)"
            )

        logger.info(f"📊 [{request_id}] Model: {body.get('model', 'N/A')} | Messages: {len(body.get('messages', []))}")

        # Convertir request
        messages, kwargs = converter.openai_to_litellm(body)

        logger.info(
            f"🔄 [{request_id}] Convertido a formato LiteLLM | "
            f"Messages: {len(messages)} | "
            f"System: {'✓' if messages and messages[0]['role'] == 'system' else '✗'}"
        )

        # Extraer user_api_key
        user_api_key = request.headers.get("x-api-key")
        if not user_api_key:
            auth_header = request.headers.get("authorization", "")
            user_api_key = auth_header.replace("Bearer ", "").strip()

        if user_api_key:
            kwargs["headers"] = {"user_api_key": user_api_key}
            logger.info(f"🔑 [{request_id}] user_api_key detectada")

        model = body.get("model")

        # Decidir entre streaming y no streaming
        if stream:
            logger.info(f"🌊 [{request_id}] Modo streaming activado")
            return await converter._responses_streaming(request_id, response_id, output_item_id, messages, kwargs, model)
        else:
            logger.info(f"📄 [{request_id}] Modo sin streaming")
            return await converter._responses_non_streaming(request_id, response_id, output_item_id, messages, kwargs, model)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Error en /v1/responses: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
@app.get("/models")
async def models_endpoint():
    """
    Lista los modelos disponibles en formato OpenAI Models API.
    
    Retorna una lista de todos los modelos compatibles con SAI.
    No requiere autenticación.

    Returns:
        JSONResponse: Lista de modelos en formato OpenAI con estructura:
            {
                "object": "list",
                "data": [
                    {
                        "id": "model-id",
                        "object": "model",
                        "created": timestamp,
                        "owned_by": "organization"
                    },
                    ...
                ]
            }
    """
    logger.info("📋 GET /v1/models request recibido")

    models_data = get_models_list()

    logger.info(f"✅ Retornando {len(models_data['data'])} modelos disponibles")
    return JSONResponse(content=models_data)


@app.get("/v1/models/{model_id}")
@app.get("/models/{model_id}")
async def model_detail_endpoint(model_id: str):
    """
    Retorna detalles de un modelo específico.
    
    Consulta información detallada de un modelo por su ID.
    No requiere autenticación.

    Args:
        model_id (str): ID del modelo a consultar
        
    Returns:
        JSONResponse: Información del modelo en formato OpenAI
        
    Raises:
        HTTPException: 404 si el modelo no existe
    """
    logger.info(f"📋 GET /v1/models/{model_id} request recibido")

    model_info = get_model_by_id(model_id)

    if model_info:
        logger.info(f"✅ Modelo '{model_id}' encontrado")
        return JSONResponse(content=model_info)
    else:
        logger.warning(f"⚠️ Modelo '{model_id}' no encontrado")
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"The model '{model_id}' does not exist",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "model_not_found"
                }
            }
        )


@app.get("/")
async def root():
    """
    Documentación y metadata del gateway.
    
    Retorna información sobre el servicio, versión, endpoints disponibles
    y notas de uso.
    
    Returns:
        dict: Información completa del gateway incluyendo changelog y endpoints
    """
    return {
        "service": "OpenAI SAI Gateway",
        "version": "1.0.4",
        "description": "Gateway que expone SAI con APIs compatibles con OpenAI",
        "changelog": {
            "1.0.4": "Corregido evento final de 'response.completed' a 'response.done' según spec oficial",
            "1.0.3": "Cierre inmediato de stream después de completed",
            "1.0.2": "Intento de fix con keepalives",
            "1.0.1": "Intento de fix con delay",
            "1.0.0": "Release inicial"
        },
        "endpoints": {
            "completions": {
                "paths": ["/v1/completions", "/completions"],
                "method": "POST",
                "format": "OpenAI Completions (legacy)",
                "streaming": True,
                "description": "Endpoint compatible con OpenAI Completions API"
            },
            "chat_completions": {
                "paths": ["/v1/chat/completions", "/chat/completions"],
                "method": "POST",
                "format": "OpenAI Chat Completions",
                "streaming": True,
                "description": "Endpoint compatible con OpenAI Chat Completions API"
            },
            "messages": {
                "paths": ["/v1/messages", "/messages"],
                "method": "POST",
                "format": "OpenAI Messages",
                "streaming": False,
                "description": "Endpoint compatible con OpenAI Messages API (sin streaming)"
            },
            "responses": {
                "paths": ["/v1/responses", "/responses"],
                "method": "POST",
                "format": "OpenAI Responses API",
                "streaming": True,
                "description": "Endpoint compatible con OpenAI Responses API y Codex CLI"
            },
            "models": {
                "paths": ["/v1/models", "/models"],
                "method": "GET",
                "description": "Lista de modelos disponibles"
            },
            "model_detail": {
                "paths": ["/v1/models/{model_id}", "/models/{model_id}"],
                "method": "GET",
                "description": "Detalles de un modelo específico"
            },
            "health": {
                "paths": ["/health"],
                "method": "GET",
                "description": "Health check endpoint"
            }
        },
        "authentication": {
            "methods": ["Authorization header (Bearer token)", "x-api-key header"],
            "required": False,
            "note": "Opcional pero recomendado para uso en producción"
        },
        "notes": [
            "Todos los endpoints funcionan con o sin el prefijo /v1",
            "El endpoint /v1/responses emite eventos SSE: response.created, response.output_text.delta, response.output_item.done, response.done",
            "Para streaming, usar /v1/responses o /v1/chat/completions con stream=true",
            "El endpoint /v1/messages NO soporta streaming"
        ]
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint para monitoreo.
    
    Retorna el estado del servicio. Útil para load balancers y
    sistemas de monitoreo.
    
    Returns:
        dict: Estado del servicio con versión
    """
    return {
        "status": "healthy",
        "service": "openai-sai-gateway",
        "version": "1.0.4"
    }


if __name__ == "__main__":
    import signal
    import sys

    print("\n" + "="*80)
    print("🚀 OPENAI SAI GATEWAY v1.0.4")
    print("="*80)
    logger.info("📍 Endpoints disponibles:")
    logger.info("   - POST /v1/completions  (o /completions)")
    logger.info("           OpenAI Completions format (legacy) - streaming y no streaming")
    logger.info("   - POST /v1/chat/completions  (o /chat/completions)")
    logger.info("           OpenAI Chat Completions format - streaming y no streaming")
    logger.info("   - POST /v1/messages  (o /messages)")
    logger.info("           OpenAI Messages format - sin streaming")
    logger.info("   - POST /v1/responses  (o /responses)")
    logger.info("           OpenAI Responses API format - con streaming SSE")
    logger.info("   - GET /v1/models")
    logger.info("           Lista de modelos disponibles")
    logger.info("   - GET /health")
    logger.info("")
    logger.info("💡 Todos los endpoints funcionan con o sin el prefijo /v1")
    logger.info("🔧 FIX v1.0.4: Evento final correcto: response.done (no completed)")
    logger.info("🌐 Servidor iniciando en http://0.0.0.0:8000")
    logger.info("💡 Presiona Ctrl+C para detener")
    print("="*80 + "\n")

    def signal_handler(sig, frame):
        """
        Manejador de señal SIGINT (Ctrl+C).
        
        Permite detener el servidor limpiamente cuando se presiona Ctrl+C.
        
        Args:
            sig: Señal recibida
            frame: Frame actual de ejecución
        """
        print("\n" + "="*80)
        logger.info("👋 Deteniendo gateway... (Ctrl+C recibido)")
        logger.info("✅ Gateway detenido exitosamente")
        print("="*80 + "\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            timeout_keep_alive=75,
            timeout_graceful_shutdown=30
        )
    except KeyboardInterrupt:
        print("\n" + "="*80)
        logger.info("👋 Gateway detenido limpiamente")
        print("="*80 + "\n")
        sys.exit(0)