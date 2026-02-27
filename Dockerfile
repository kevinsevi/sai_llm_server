FROM python:3.12-slim

# Evitar buffering
ENV PYTHONUNBUFFERED=1

# Establecer zona horaria
ENV TZ=America/Guayaquil

# Instalar dependencias necesarias para tzdata
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -snf /usr/share/zoneinfo/"$TZ" /etc/localtime && \
    echo "$TZ" > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo
WORKDIR /app

# Instalar dependencias
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    starlette \
    requests \
    python-dotenv \
    litellm

# Copiar el código (todas las clases sai_*.py)
COPY sai_*.py ./

# (Opcional) si aún usas config.yaml para otra cosa, mantenlo; si no, puedes borrarlo
COPY config.yaml ./

# Puerto del servicio
EXPOSE 4000

# Arrancar el gateway
CMD ["uvicorn", "sai_gateway:app", "--host", "0.0.0.0", "--port", "4000"]
