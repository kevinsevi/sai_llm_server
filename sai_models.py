# sai_models.py

# Lista de modelos disponibles en SAI
# Nota: el "id" es el que expone la OpenAI Models API (/v1/models).
# El identificador interno/provider (p.ej. "OpenAIAPI/sai-model") se puede guardar en metadata.
AVAILABLE_MODELS = [
    {
        "id": "amazon.nova-canvas-v1:0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "amazon.nova-canvas-v1:0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/amazon.nova-canvas-v1:0"
        }
    },
    {
        "id": "gemini-3-pro-preview",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-3-pro-preview",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-3-pro-preview"
        }
    },
    {
        "id": "gemini-3-pro-image-preview",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-3-pro-image-preview",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-3-pro-image-preview"
        }
    },
    {
        "id": "gpt-4.1",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4.1",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/EMEA/gpt-4.1"
        }
    },
    {
        "id": "gpt-5.2-chat",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5.2-chat",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/EMEA/gpt-5.2-chat"
        }
    },
    {
        "id": "gpt-5-mini",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-mini",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/EMEA/gpt-5-mini"
        }
    },
    {
        "id": "gpt-image-1.5",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-image-1.5",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-image-1.5"
        }
    },
    {
        "id": "gpt-image-1",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-image-1",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-image-1"
        }
    },
    {
        "id": "dall-e-3",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "dall-e-3",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/dall-e-3"
        }
    },
    {
        "id": "dall-e-2",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "dall-e-2",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/dall-e-2"
        }
    },
    {
        "id": "gpt-image-1",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-image-1",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-image-1"
        }
    },
    {
        "id": "gpt-4-turbo-2024-04-09",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4-turbo-2024-04-09",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4-turbo-2024-04-09"
        }
    },
    {
        "id": "gpt-4o-mini-2024-07-18",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4o-mini-2024-07-18",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/EMEA/gpt-4o-mini-2024-07-18"
        }
    },
    {
        "id": "gpt-3.5-turbo-0125",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-3.5-turbo-0125",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-3.5-turbo-0125"
        }
    },
    {
        "id": "gpt-4o-2024-05-13",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4o-2024-05-13",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4o-2024-05-13"
        }
    },
    {
        "id": "gpt-4o-2024-08-06",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4o-2024-08-06",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4o-2024-08-06"
        }
    },
    {
        "id": "gpt-4o-search-preview-2025-03-11",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4o-search-preview-2025-03-11",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4o-search-preview-2025-03-11"
        }
    },
    {
        "id": "gpt-4o-mini-search-preview-2025-03-11",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4o-mini-search-preview-2025-03-11",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4o-mini-search-preview-2025-03-11"
        }
    },
    {
        "id": "o1-2024-12-17",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "o1-2024-12-17",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/o1-2024-12-17"
        }
    },
    {
        "id": "o3-mini-2025-01-31",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "o3-mini-2025-01-31",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/o3-mini-2025-01-31"
        }
    },
    {
        "id": "o3-mini-2025-01-31",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "o3-mini-2025-01-31",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/o3-mini-2025-01-31"
        }
    },
    {
        "id": "o4-mini-2025-04-16",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "o4-mini-2025-04-16",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/o4-mini-2025-04-16"
        }
    },
    {
        "id": "o4-mini-2025-04-16",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "o4-mini-2025-04-16",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/o4-mini-2025-04-16"
        }
    },
    {
        "id": "gpt-4o-2024-11-20",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4o-2024-11-20",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4o-2024-11-20"
        }
    },
    {
        "id": "gpt-4o-mini-2024-07-18",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4o-mini-2024-07-18",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4o-mini-2024-07-18"
        }
    },
    {
        "id": "o3-2025-04-16",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "o3-2025-04-16",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/o3-2025-04-16"
        }
    },
    {
        "id": "gpt-4.1-2025-04-14",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4.1-2025-04-14",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4.1-2025-04-14"
        }
    },
    {
        "id": "gpt-4.1-mini-2025-04-14",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4.1-mini-2025-04-14",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4.1-mini-2025-04-14"
        }
    },
    {
        "id": "gpt-3.5-turbo-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-3.5-turbo-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-3.5-turbo-instruct"
        }
    },
    {
        "id": "gpt-4-0125-preview",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4-0125-preview",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4-0125-preview"
        }
    },
    {
        "id": "gpt-4.1-nano-2025-04-14",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4.1-nano-2025-04-14",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4.1-nano-2025-04-14"
        }
    },
    {
        "id": "gpt-4",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4"
        }
    },
    {
        "id": "gpt-5-codex",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-codex",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5-codex"
        }
    },
    {
        "id": "gpt-5.1-codex",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5.1-codex",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5.1-codex"
        }
    },
    {
        "id": "r1-1776",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "r1-1776",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/PERPLEXITY (NO WEB SEARCH)/r1-1776"
        }
    },
    {
        "id": "gpt-5.2-codex",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5.2-codex",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5.2-codex"
        }
    },
    {
        "id": "sonar-pro",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "sonar-pro",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/PERPLEXITY/sonar-pro"
        }
    },
    {
        "id": "gpt-4.5-preview-2025-02-27",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4.5-preview-2025-02-27",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4.5-preview-2025-02-27"
        }
    },
    {
        "id": "llama-2-70b-chat",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-2-70b-chat",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-2-70b-chat"
        }
    },
    {
        "id": "o1-mini-2024-09-12",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "o1-mini-2024-09-12",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/o1-mini-2024-09-12"
        }
    },
    {
        "id": "sonar",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "sonar",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/PERPLEXITY/sonar"
        }
    },
    {
        "id": "gpt-4-1106-preview",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-4-1106-preview",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-4-1106-preview"
        }
    },
    {
        "id": "llama-2-13b-chat",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-2-13b-chat",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-2-13b-chat"
        }
    },
    {
        "id": "mistral-small-2503-ekfbe",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "mistral-small-2503-ekfbe",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/mistral-small-2503-ekfbe"
        }
    },
    {
        "id": "gpt-3.5-turbo",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-3.5-turbo",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-3.5-turbo"
        }
    },
    {
        "id": "llama-2-7b-chat",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-2-7b-chat",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-2-7b-chat"
        }
    },
    {
        "id": "deepseek-r1",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "deepseek-r1",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/deepseek-r1"
        }
    },
    {
        "id": "phi-3-small-128k-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "phi-3-small-128k-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/phi-3-small-128k-instruct"
        }
    },
    {
        "id": "gpt-5-2025-08-07",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-2025-08-07",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5-2025-08-07"
        }
    },
    {
        "id": "gpt-3.5-turbo-1106",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-3.5-turbo-1106",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-3.5-turbo-1106"
        }
    },
    {
        "id": "mistral-large",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "mistral-large",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/mistral-large"
        }
    },
    {
        "id": "phi-3-medium-128k-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "phi-3-medium-128k-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/phi-3-medium-128k-instruct"
        }
    },
    {
        "id": "cohere.command-r-v1:0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "cohere.command-r-v1:0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/cohere.command-r-v1:0"
        }
    },
    {
        "id": "phi-3-5-mini-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "phi-3-5-mini-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/phi-3-5-mini-instruct"
        }
    },
    {
        "id": "gpt-5.1-2025-11-13",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5.1-2025-11-13",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5.1-2025-11-13"
        }
    },
    {
        "id": "claude-3-opus-20240229",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-3-opus-20240229",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-3-opus-20240229"
        }
    },
    {
        "id": "cohere.command-r-plus-v1:0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "cohere.command-r-plus-v1:0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/cohere.command-r-plus-v1:0"
        }
    },
    {
        "id": "mistral-small",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "mistral-small",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/mistral-small"
        }
    },
    {
        "id": "mistral-large-2407",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "mistral-large-2407",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/mistral-large-2407"
        }
    },
    {
        "id": "azure-mistral-large-2411",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "azure-mistral-large-2411",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/azure-mistral-large-2411"
        }
    },
    {
        "id": "gpt-5-mini-2025-08-07",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-mini-2025-08-07",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5-mini-2025-08-07"
        }
    },
    {
        "id": "claude-2.1",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-2.1",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-2.1"
        }
    },
    {
        "id": "ai21.j2-ultra-v1",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "ai21.j2-ultra-v1",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/ai21.j2-ultra-v1"
        }
    },
    {
        "id": "mistral-nemo",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "mistral-nemo",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/mistral-nemo"
        }
    },
    {
        "id": "o1-preview-2024-09-12",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "o1-preview-2024-09-12",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/o1-preview-2024-09-12"
        }
    },
    {
        "id": "google/gemini-2.5-flash-preview-04-17",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "google/gemini-2.5-flash-preview-04-17",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/google/gemini-2.5-flash-preview-04-17"
        }
    },
    {
        "id": "claude-3-5-sonnet-20240620",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-3-5-sonnet-20240620",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-3-5-sonnet-20240620"
        }
    },
    {
        "id": "llama-3-8b-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-3-8b-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-3-8b-instruct"
        }
    },
    {
        "id": "llama-3-70b-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-3-70b-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-3-70b-instruct"
        }
    },
    {
        "id": "ai21.j2-mid-v1",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "ai21.j2-mid-v1",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/ai21.j2-mid-v1"
        }
    },
    {
        "id": "google/gemini-2.5-pro-preview-05-06",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "google/gemini-2.5-pro-preview-05-06",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/google/gemini-2.5-pro-preview-05-06"
        }
    },
    {
        "id": "gpt-5-nano-2025-08-07",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-nano-2025-08-07",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5-nano-2025-08-07"
        }
    },
    {
        "id": "claude-2.0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-2.0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-2.0"
        }
    },
    {
        "id": "ai21.jamba-instruct-v1:0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "ai21.jamba-instruct-v1:0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/ai21.jamba-instruct-v1:0"
        }
    },
    {
        "id": "llama-3-1-405b-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-3-1-405b-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-3-1-405b-instruct"
        }
    },
    {
        "id": "o1-pro-2025-03-19",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "o1-pro-2025-03-19",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/o1-pro-2025-03-19"
        }
    },
    {
        "id": "gpt-3.5-turbo-16k",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-3.5-turbo-16k",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-3.5-turbo-16k"
        }
    },
    {
        "id": "gemini-1.5-pro-preview-0409",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-1.5-pro-preview-0409",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-1.5-pro-preview-0409"
        }
    },
    {
        "id": "llama-3-1-8b-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-3-1-8b-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-3-1-8b-instruct"
        }
    },
    {
        "id": "ai21.jamba-1-5-large-v1:0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "ai21.jamba-1-5-large-v1:0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/ai21.jamba-1-5-large-v1:0"
        }
    },
    {
        "id": "claude-3-5-sonnet-20241022",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-3-5-sonnet-20241022",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-3-5-sonnet-20241022"
        }
    },
    {
        "id": "grok-2-vision-1212",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "grok-2-vision-1212",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/XAI/grok-2-vision-1212"
        }
    },
    {
        "id": "gemini-pro",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-pro",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-pro"
        }
    },
    {
        "id": "claude-instant-1.2",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-instant-1.2",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-instant-1.2"
        }
    },
    {
        "id": "llama-3-1-70b-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-3-1-70b-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-3-1-70b-instruct"
        }
    },
    {
        "id": "phi-3-5-vision-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "phi-3-5-vision-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/phi-3-5-vision-instruct"
        }
    },
    {
        "id": "ai21.jamba-1-5-mini-v1:0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "ai21.jamba-1-5-mini-v1:0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/ai21.jamba-1-5-mini-v1:0"
        }
    },
    {
        "id": "gemini-1.0-pro-001",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-1.0-pro-001",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-1.0-pro-001"
        }
    },
    {
        "id": "grok-2-1212",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "grok-2-1212",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/XAI/grok-2-1212"
        }
    },
    {
        "id": "gpt-5.2-2025-12-11",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5.2-2025-12-11",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5.2-2025-12-11"
        }
    },
    {
        "id": "gemini-1.5-pro-002",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-1.5-pro-002",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-1.5-pro-002"
        }
    },
    {
        "id": "gemini-1.5-pro-001",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-1.5-pro-001",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-1.5-pro-001"
        }
    },
    {
        "id": "amazon.titan-text-express-v1",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "amazon.titan-text-express-v1",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/amazon.titan-text-express-v1"
        }
    },
    {
        "id": "google/gemini-1.5-pro-002",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "google/gemini-1.5-pro-002",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/google/gemini-1.5-pro-002"
        }
    },
    {
        "id": "llama-3-2-90b-vision-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-3-2-90b-vision-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-3-2-90b-vision-instruct"
        }
    },
    {
        "id": "claude-3-7-sonnet-20250219",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-3-7-sonnet-20250219",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-3-7-sonnet-20250219"
        }
    },
    {
        "id": "imagen-3.0-generate-002",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "imagen-3.0-generate-002",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/imagen-3.0-generate-002"
        }
    },
    {
        "id": "google/gemini-2.5-pro-preview-03-25",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "google/gemini-2.5-pro-preview-03-25",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/google/gemini-2.5-pro-preview-03-25"
        }
    },
    {
        "id": "grok-3-beta",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "grok-3-beta",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/XAI/grok-3-beta"
        }
    },
    {
        "id": "gpt-5-2025-08-07",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-2025-08-07",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5-2025-08-07"
        }
    },
    {
        "id": "gemini-1.5-flash-001",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-1.5-flash-001",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-1.5-flash-001"
        }
    },
    {
        "id": "amazon.titan-text-lite-v1",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "amazon.titan-text-lite-v1",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/amazon.titan-text-lite-v1"
        }
    },
    {
        "id": "google/gemini-1.5-flash-002",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "google/gemini-1.5-flash-002",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/google/gemini-1.5-flash-002"
        }
    },
    {
        "id": "llama-3-2-11b-vision-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-3-2-11b-vision-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-3-2-11b-vision-instruct"
        }
    },
    {
        "id": "grok-3-mini-beta",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "grok-3-mini-beta",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/XAI/grok-3-mini-beta"
        }
    },
    {
        "id": "claude-opus-4-20250514",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-opus-4-20250514",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-opus-4-20250514"
        }
    },
    {
        "id": "imagen-4.0-generate-001",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "imagen-4.0-generate-001",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/imagen-4.0-generate-001"
        }
    },
    {
        "id": "gpt-5-mini-2025-08-07",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-mini-2025-08-07",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5-mini-2025-08-07"
        }
    },
    {
        "id": "claude-3-sonnet-20240229",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-3-sonnet-20240229",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-3-sonnet-20240229"
        }
    },
    {
        "id": "amazon.titan-text-premier-v1:0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "amazon.titan-text-premier-v1:0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/amazon.titan-text-premier-v1:0"
        }
    },
    {
        "id": "gemini-1.0-pro-002",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-1.0-pro-002",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-1.0-pro-002"
        }
    },
    {
        "id": "phi-4",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "phi-4",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/phi-4"
        }
    },
    {
        "id": "google/gemini-2.0-flash-001",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "google/gemini-2.0-flash-001",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/google/gemini-2.0-flash-001"
        }
    },
    {
        "id": "grok-code-fast-1",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "grok-code-fast-1",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/XAI/grok-code-fast-1"
        }
    },
    {
        "id": "gpt-5-nano-2025-08-07",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-nano-2025-08-07",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5-nano-2025-08-07"
        }
    },
    {
        "id": "amazon.nova-lite-v1:0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "amazon.nova-lite-v1:0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/amazon.nova-lite-v1:0"
        }
    },
    {
        "id": "llama-3-3-70b-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-3-3-70b-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-3-3-70b-instruct"
        }
    },
    {
        "id": "google/gemini-2.0-flash-lite-001",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "google/gemini-2.0-flash-lite-001",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/google/gemini-2.0-flash-lite-001"
        }
    },
    {
        "id": "claude-sonnet-4-20250514",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-sonnet-4-20250514",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-sonnet-4-20250514"
        }
    },
    {
        "id": "grok-4-fast-reasoning",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "grok-4-fast-reasoning",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/XAI/grok-4-fast-reasoning"
        }
    },
    {
        "id": "gpt-5-2025-08-07",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-2025-08-07",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5-2025-08-07"
        }
    },
    {
        "id": "amazon.nova-pro-v1:0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "amazon.nova-pro-v1:0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/amazon.nova-pro-v1:0"
        }
    },
    {
        "id": "llama-4-maverick-17b-128e-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-4-maverick-17b-128e-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-4-maverick-17b-128e-instruct"
        }
    },
    {
        "id": "gemini-2.5-pro",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-2.5-pro",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-2.5-pro"
        }
    },
    {
        "id": "grok-4-fast-non-reasoning",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "grok-4-fast-non-reasoning",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/XAI/grok-4-fast-non-reasoning"
        }
    },
    {
        "id": "claude-sonnet-4-5-20250929",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-sonnet-4-5-20250929",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-sonnet-4-5-20250929"
        }
    },
    {
        "id": "gpt-5-nano-2025-08-07",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-nano-2025-08-07",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5-nano-2025-08-07"
        }
    },
    {
        "id": "amazon.nova-micro-v1:0",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "amazon.nova-micro-v1:0",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/BEDROCK/amazon.nova-micro-v1:0"
        }
    },
    {
        "id": "llama-4-scout-17b-16e-instruct",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "llama-4-scout-17b-16e-instruct",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/AZURE/llama-4-scout-17b-16e-instruct"
        }
    },
    {
        "id": "gemini-2.5-flash",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gemini-2.5-flash",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/GOOGLE/gemini-2.5-flash"
        }
    },
    {
        "id": "grok-4-0709",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "grok-4-0709",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/XAI/grok-4-0709"
        }
    },
    {
        "id": "gpt-5-mini-2025-08-07",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "gpt-5-mini-2025-08-07",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/OPENAI/gpt-5-mini-2025-08-07"
        }
    },
    {
        "id": "claude-sonnet-4-6",
        "object": "model",
        "created": 1735689600,
        "owned_by": "stefanini",
        "permission": [],
        "root": "claude-sonnet-4-6",
        "parent": None,
        "metadata": {
            "provider_model": "SAI/ANTHROPIC/claude-sonnet-4-6"
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