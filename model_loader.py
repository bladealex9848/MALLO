"""
Módulo para cargar modelos desde APIs externas como OpenRouter y Groq.
"""

import requests
import logging
import time
import json
import os
from typing import List, Dict, Any, Optional
from load_secrets import get_secret, secrets

# Configuración de caché
CACHE_DIR = "cache"
CACHE_DURATION = 3600  # 1 hora en segundos

# Asegurar que el directorio de caché existe
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def get_cache_path(provider: str) -> str:
    """Obtiene la ruta del archivo de caché para un proveedor específico."""
    return os.path.join(CACHE_DIR, f"{provider}_models_cache.json")


def is_cache_valid(provider: str) -> bool:
    """Verifica si la caché para un proveedor específico es válida."""
    cache_path = get_cache_path(provider)
    if not os.path.exists(cache_path):
        return False

    # Verificar si la caché ha expirado
    cache_time = os.path.getmtime(cache_path)
    current_time = time.time()
    return (current_time - cache_time) < CACHE_DURATION


def load_from_cache(provider: str) -> List[Dict[str, Any]]:
    """Carga modelos desde la caché para un proveedor específico."""
    cache_path = get_cache_path(provider)
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error al cargar caché para {provider}: {str(e)}")
        return []


def save_to_cache(provider: str, models: List[Dict[str, Any]]) -> None:
    """Guarda modelos en la caché para un proveedor específico."""
    cache_path = get_cache_path(provider)
    try:
        with open(cache_path, "w") as f:
            json.dump(models, f, indent=2)
    except Exception as e:
        logging.error(f"Error al guardar caché para {provider}: {str(e)}")


def load_models_from_openrouter() -> List[Dict[str, Any]]:
    """
    Carga los modelos disponibles desde la API de OpenRouter.

    Returns:
        List[Dict[str, Any]]: Lista de modelos disponibles en OpenRouter.
    """
    # Verificar si hay una caché válida
    if is_cache_valid("openrouter"):
        return load_from_cache("openrouter")

    api_key = get_secret("OPENROUTER_API_KEY")
    if not api_key:
        logging.warning("OpenRouter API key no configurada")
        return []

    try:
        # Realizar solicitud a la API de OpenRouter
        response = requests.get(
            url="https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": get_secret("YOUR_SITE_URL", "http://localhost:8501"),
                "X-Title": "MALLO",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

        # Verificar si hay error
        if not response.ok:
            logging.error(
                f"Error al cargar modelos desde OpenRouter API: {response.status_code} {response.reason}"
            )
            return []

        # Obtener datos
        data = response.json()

        # Verificar si hay modelos
        if not data.get("data") or not isinstance(data["data"], list):
            logging.warning(
                "No se encontraron modelos en la respuesta de OpenRouter API"
            )
            return []

        # Filtrar modelos gratuitos y de OpenRouter
        api_models = []
        for model in data["data"]:
            # Verificar si el modelo es gratuito (contiene ":free" en el ID o tiene context_length_free > 0)
            is_free = ":free" in model.get("id", "") or (
                model.get("context_length_free", 0) > 0
            )

            # Incluir el modelo si es gratuito
            if is_free:
                api_models.append(
                    {
                        "id": model.get("id", ""),
                        "name": model.get("name", model.get("id", "").split("/")[-1]),
                        "provider": (
                            model.get("id", "").split("/")[0]
                            if "/" in model.get("id", "")
                            else "Desconocido"
                        ),
                        "specialty": get_model_specialty(model),
                        "capabilities": get_model_capabilities(model),
                        "context_length": model.get("context_length", 0),
                        "context_length_free": model.get("context_length_free", 0),
                        "pricing": {
                            "prompt": model.get("pricing", {}).get("prompt", 0),
                            "completion": model.get("pricing", {}).get("completion", 0),
                        },
                        "source": "openrouter",
                    }
                )

        # Guardar en caché
        save_to_cache("openrouter", api_models)

        return api_models

    except Exception as e:
        logging.error(f"Error al cargar modelos desde OpenRouter API: {str(e)}")
        return []


def load_models_from_groq() -> List[Dict[str, Any]]:
    """
    Carga los modelos disponibles desde la API de Groq.

    Returns:
        List[Dict[str, Any]]: Lista de modelos disponibles en Groq.
    """
    # Verificar si hay una caché válida
    if is_cache_valid("groq"):
        return load_from_cache("groq")

    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        logging.warning("Groq API key no configurada")
        return []

    try:
        # Realizar solicitud a la API de Groq
        response = requests.get(
            url="https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

        # Verificar si hay error
        if not response.ok:
            logging.error(
                f"Error al cargar modelos desde Groq API: {response.status_code} {response.reason}"
            )
            return []

        # Obtener datos
        data = response.json()

        # Verificar si hay modelos
        if not data.get("data") or not isinstance(data["data"], list):
            logging.warning("No se encontraron modelos en la respuesta de Groq API")
            return []

        # Transformar modelos al formato de la aplicación
        api_models = []
        for model in data["data"]:
            api_models.append(
                {
                    "id": model.get("id", ""),
                    "name": model.get("id", ""),
                    "provider": "groq",
                    "specialty": get_groq_model_specialty(model),
                    "capabilities": get_groq_model_capabilities(model),
                    "context_length": model.get("context_window", 0),
                    "source": "groq",
                }
            )

        # Guardar en caché
        save_to_cache("groq", api_models)

        return api_models

    except Exception as e:
        logging.error(f"Error al cargar modelos desde Groq API: {str(e)}")
        return []


def load_models_from_ollama() -> List[Dict[str, Any]]:
    """
    Carga los modelos disponibles desde Ollama local.

    Returns:
        List[Dict[str, Any]]: Lista de modelos disponibles en Ollama.
    """
    # Verificar si hay una caché válida
    if is_cache_valid("ollama"):
        return load_from_cache("ollama")

    # Intentar obtener la URL base de Ollama desde la configuración
    import yaml

    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            base_url = config.get("ollama", {}).get(
                "base_url", "http://localhost:11434"
            )
    except Exception:
        base_url = "http://localhost:11434"

    try:
        # Realizar solicitud a la API de Ollama
        response = requests.get(url=f"{base_url}/api/tags", timeout=5)

        # Verificar si hay error
        if not response.ok:
            logging.error(
                f"Error al cargar modelos desde Ollama API: {response.status_code} {response.reason}"
            )
            return get_ollama_models_from_command()

        # Obtener datos
        data = response.json()

        # Verificar si hay modelos
        if not data.get("models") or not isinstance(data["models"], list):
            logging.warning("No se encontraron modelos en la respuesta de Ollama API")
            return get_ollama_models_from_command()

        # Transformar modelos al formato de la aplicación
        api_models = []
        for model in data["models"]:
            api_models.append(
                {
                    "id": model.get("name", ""),
                    "name": model.get("name", ""),
                    "provider": "ollama",
                    "specialty": get_ollama_model_specialty(model.get("name", "")),
                    "capabilities": get_ollama_model_capabilities(
                        model.get("name", "")
                    ),
                    "size": model.get("size", 0),
                    "modified": model.get("modified_at", ""),
                    "source": "ollama",
                }
            )

        # Guardar en caché
        save_to_cache("ollama", api_models)

        return api_models

    except Exception as e:
        logging.error(f"Error al cargar modelos desde Ollama API: {str(e)}")
        # Intentar obtener modelos mediante el comando ollama list
        return get_ollama_models_from_command()


def get_ollama_models_from_command() -> List[Dict[str, Any]]:
    """
    Obtiene los modelos de Ollama mediante el comando 'ollama list'.

    Returns:
        List[Dict[str, Any]]: Lista de modelos disponibles en Ollama.
    """
    try:
        import subprocess

        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Error al ejecutar 'ollama list': {result.stderr}")
            return []

        # Parsear la salida del comando
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:  # Solo encabezado o vacío
            return []

        models = []
        # Saltar la primera línea (encabezado)
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                size = parts[2] + " " + parts[3] if len(parts) > 3 else parts[2]

                models.append(
                    {
                        "id": name,
                        "name": name,
                        "provider": "ollama",
                        "specialty": get_ollama_model_specialty(name),
                        "capabilities": get_ollama_model_capabilities(name),
                        "size": size,
                        "source": "ollama",
                    }
                )

        # Guardar en caché
        save_to_cache("ollama", models)

        return models

    except Exception as e:
        logging.error(f"Error al obtener modelos de Ollama mediante comando: {str(e)}")
        return []


def get_model_specialty(model: Dict[str, Any]) -> str:
    """
    Determina la especialidad del modelo de OpenRouter basado en sus características.

    Args:
        model (Dict[str, Any]): Modelo de la API.

    Returns:
        str: Especialidad del modelo.
    """
    model_id = model.get("id", "").lower()

    # Verificar si el modelo admite imágenes
    if model.get("multimodal", False):
        return "multimodal"

    # Verificar si el modelo es especializado en código
    if "code" in model_id or "coder" in model_id:
        return "coding"

    # Verificar si el modelo es creativo
    if "creative" in model_id or "story" in model_id:
        return "creative"

    # Verificar si el modelo es para audio
    if "whisper" in model_id or "audio" in model_id or "tts" in model_id:
        return "audio"

    # Por defecto, es un modelo general
    return "general"


def get_model_capabilities(model: Dict[str, Any]) -> List[str]:
    """
    Determina las capacidades del modelo de OpenRouter basado en sus características.

    Args:
        model (Dict[str, Any]): Modelo de la API.

    Returns:
        List[str]: Capacidades del modelo.
    """
    capabilities = []
    model_id = model.get("id", "").lower()

    # Verificar capacidades basadas en características del modelo
    if model.get("multimodal", False):
        capabilities.append("vision_understanding")
        capabilities.append("image_reasoning")

    if model.get("context_length", 0) > 16000:
        capabilities.append("large_context")

    if model.get("context_length_free", 0) > 0:
        capabilities.append("free_tier_available")

    # Añadir capacidades basadas en el ID del modelo
    if "instruct" in model_id:
        capabilities.append("instruction_optimized")

    if "code" in model_id or "coder" in model_id:
        capabilities.append("code_generation")

    # Añadir capacidades basadas en el proveedor
    provider = model_id.split("/")[0] if "/" in model_id else ""

    if provider == "anthropic":
        capabilities.append("advanced_reasoning")

    if provider == "google":
        capabilities.append("multilingual_support")

    if provider == "openai":
        capabilities.append("advanced_reasoning")

    if provider == "meta-llama":
        capabilities.append("multilingual_support")

    if provider == "openrouter":
        capabilities.append("proprietary_architecture")

    return capabilities


def get_groq_model_specialty(model: Dict[str, Any]) -> str:
    """
    Determina la especialidad del modelo de Groq basado en sus características.

    Args:
        model (Dict[str, Any]): Modelo de la API.

    Returns:
        str: Especialidad del modelo.
    """
    model_id = model.get("id", "").lower()

    # Verificar si el modelo es para audio
    if "whisper" in model_id:
        return "audio_transcription"

    # Verificar si el modelo es para texto a voz
    if "tts" in model_id:
        return "text_to_speech"

    # Verificar si el modelo es para moderación
    if "guard" in model_id:
        return "content_moderation"

    # Verificar si el modelo es especializado en código
    if "coder" in model_id:
        return "coding"

    # Por defecto, es un modelo general
    return "general"


def get_groq_model_capabilities(model: Dict[str, Any]) -> List[str]:
    """
    Determina las capacidades del modelo de Groq basado en sus características.

    Args:
        model (Dict[str, Any]): Modelo de la API.

    Returns:
        List[str]: Capacidades del modelo.
    """
    capabilities = []
    model_id = model.get("id", "").lower()

    # Capacidades basadas en el ID del modelo
    if "whisper" in model_id:
        capabilities.append("speech_to_text")
        if "large" in model_id:
            capabilities.append("multilingual_audio")
        if "turbo" in model_id:
            capabilities.append("faster_processing")

    if "tts" in model_id:
        capabilities.append("voice_synthesis")
        capabilities.append("text_to_audio")
        if "arabic" in model_id:
            capabilities.append("arabic_language")

    if "llama" in model_id:
        if "70b" in model_id or "70-b" in model_id:
            capabilities.append("advanced_reasoning")
            capabilities.append("large_context")
        if "8b" in model_id or "8-b" in model_id:
            capabilities.append("fast_response")
            capabilities.append("medium_context")
        if "instant" in model_id:
            capabilities.append("instant_response")
            capabilities.append("low_latency")

    if "guard" in model_id:
        capabilities.append("content_filtering")
        capabilities.append("safety_evaluation")

    if "gemma" in model_id:
        capabilities.append("instruction_following")
        capabilities.append("medium_size")

    if "qwen" in model_id:
        if "coder" in model_id:
            capabilities.append("code_generation")
            capabilities.append("technical_analysis")
        else:
            capabilities.append("advanced_reasoning")
            capabilities.append("large_context")

    if "qwq" in model_id:
        capabilities.append("creative_writing")
        capabilities.append("storytelling")

    if "mistral" in model_id:
        capabilities.append("advanced_reasoning")
        capabilities.append("diverse_knowledge")

    if "deepseek" in model_id:
        capabilities.append("instruction_following")
        capabilities.append("large_context")
        capabilities.append("efficiency")

    # Añadir capacidad de contexto si está disponible
    context_window = model.get("context_window", 0)
    if context_window > 0:
        capabilities.append(f"context_window_{context_window}")

    return capabilities


def get_ollama_model_specialty(model_name: str) -> str:
    """
    Determina la especialidad del modelo de Ollama basado en su nombre.

    Args:
        model_name (str): Nombre del modelo.

    Returns:
        str: Especialidad del modelo.
    """
    model_name = model_name.lower()

    # Verificar si el modelo es multimodal
    if "llava" in model_name or "minicpm-v" in model_name:
        return "multimodal"

    # Verificar si el modelo es especializado en código
    if "code" in model_name or "coder" in model_name or "codegemma" in model_name:
        return "coding"

    # Por defecto, es un modelo general
    return "general"


def get_ollama_model_capabilities(model_name: str) -> List[str]:
    """
    Determina las capacidades del modelo de Ollama basado en su nombre.

    Args:
        model_name (str): Nombre del modelo.

    Returns:
        List[str]: Capacidades del modelo.
    """
    capabilities = []
    model_name = model_name.lower()

    # Capacidades basadas en el nombre del modelo
    if "llava" in model_name:
        capabilities.append("vision_understanding")
        capabilities.append("image_reasoning")

    if "minicpm-v" in model_name:
        capabilities.append("vision_understanding")
        capabilities.append("compact_size")

    if "llama" in model_name:
        if "3" in model_name:
            capabilities.append("advanced_reasoning")
        if "3.2" in model_name or "3.3" in model_name:
            capabilities.append("improved_instruction_following")

    if "gemma" in model_name:
        capabilities.append("instruction_following")
        if "3" in model_name:
            capabilities.append("advanced_reasoning")

    if "phi" in model_name:
        capabilities.append("efficient_inference")
        capabilities.append("compact_size")

    if "qwen" in model_name:
        capabilities.append("multilingual_support")
        if "2.5" in model_name:
            capabilities.append("improved_reasoning")

    if "deepseek" in model_name:
        capabilities.append("instruction_following")
        if "r1" in model_name:
            capabilities.append("research_optimized")

    if "codegemma" in model_name:
        capabilities.append("code_generation")
        capabilities.append("code_completion")

    # Añadir capacidad basada en el tamaño del modelo
    if "0.5b" in model_name or "1b" in model_name or "1.5b" in model_name:
        capabilities.append("very_compact")
    elif "2b" in model_name or "3b" in model_name:
        capabilities.append("compact_size")
    elif "7b" in model_name or "8b" in model_name:
        capabilities.append("medium_size")
    elif "13b" in model_name or "14b" in model_name:
        capabilities.append("large_size")
    elif "70b" in model_name:
        capabilities.append("very_large_size")

    return capabilities


def get_all_available_models() -> Dict[str, List[Dict[str, Any]]]:
    """
    Obtiene todos los modelos disponibles de todas las fuentes.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Diccionario con los modelos disponibles por proveedor.
    """
    models = {
        "openrouter": load_models_from_openrouter(),
        "groq": load_models_from_groq(),
        "ollama": load_models_from_ollama(),
    }

    return models


def get_model_info(provider: str, model_id: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene información detallada sobre un modelo específico.

    Args:
        provider (str): Proveedor del modelo (openrouter, groq, ollama).
        model_id (str): ID del modelo.

    Returns:
        Optional[Dict[str, Any]]: Información del modelo o None si no se encuentra.
    """
    models = get_all_available_models().get(provider, [])
    for model in models:
        if model.get("id") == model_id:
            return model
    return None


if __name__ == "__main__":
    # Prueba de las funciones
    print("Cargando modelos de OpenRouter...")
    openrouter_models = load_models_from_openrouter()
    print(f"Modelos de OpenRouter: {len(openrouter_models)}")

    print("\nCargando modelos de Groq...")
    groq_models = load_models_from_groq()
    print(f"Modelos de Groq: {len(groq_models)}")

    print("\nCargando modelos de Ollama...")
    ollama_models = load_models_from_ollama()
    print(f"Modelos de Ollama: {len(ollama_models)}")
