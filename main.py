# Importa y ejecuta la configuración de Streamlit antes que cualquier otra cosa
from config_streamlit import set_streamlit_page_config

set_streamlit_page_config()

import streamlit as st
import yaml
import os
import json
import re
from agents import AgentManager
from utilities import (
    initialize_system,
    cache_response,
    get_cached_response,
    summarize_text,
    perform_web_search,
    log_error,
    log_warning,
    log_info,
    evaluate_query_complexity,
)
from document_processor import (
    validate_file_format,
    detect_document_type,
    process_document_with_mistral_ocr,
    manage_document_context,
)
import time
import asyncio
import polars as pl
import random
import logging
import traceback
from typing import Tuple, Dict, Any, Optional
from load_secrets import load_secrets, get_secret, secrets

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mallo.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Carga todos los secretos al inicio de la aplicación
load_secrets()

# CSS personalizado para interfaz moderna - Añadir después de set_streamlit_page_config()
st.markdown(
    """
<style>
    /* Estilos base y temas */
    .main {
        background-color: transparent !important;
    }
    .stApp {
        background-color: transparent !important;
    }
    .stMarkdown h1 {
        color: inherit;
    }

    /* Componentes de chat */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Botones y controles */
    .export-button {
        background-color: rgba(240, 242, 246, 0.1);
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
    }

    /* Paneles y evaluaciones */
    .ethical-evaluation {
        background-color: rgba(248, 249, 250, 0.05);
        border-left: 4px solid #00cc66;
        padding: 1rem;
    }
    .reasoning-chain {
        background-color: rgba(248, 249, 250, 0.05);
        border-left: 4px solid #0066cc;
        padding: 1rem;
    }

    /* Expansores */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(240, 242, 246, 0.1);
        border-radius: 10px;
        margin-top: 0.5rem;
    }

    /* Métricas y badges */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    /* Enlaces y navegación */
    .social-links a {
        color: inherit;
        text-decoration: none;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        transition: background-color 0.3s;
    }

    .social-links a:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_config():
    """Carga la configuración desde el archivo YAML."""
    try:
        with open("config.yaml", "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("No se pudo encontrar el archivo de configuración 'config.yaml'.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error al leer el archivo de configuración: {str(e)}")
        st.stop()


def load_speed_test_results():
    """Carga los resultados de las pruebas de velocidad."""
    try:
        with open("model_speeds.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def display_speed_test_results(results):
    """Muestra los resultados de las pruebas de velocidad en la barra lateral."""
    data = [
        {"API": api, "Modelo": model["model"], "Velocidad": f"{model['speed']:.4f}"}
        for api, models in results.items()
        for model in models
    ]
    df = pl.DataFrame(data).sort("Velocidad")
    with st.sidebar.expander("📊 Resultados de Velocidad", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)


def _initialize_session_state():
    """Inicializa el estado de la sesión con valores predeterminados."""
    default_states = {
        "messages": [],
        "context": "",
        "show_settings": False,
        "last_details": {},
        "error_count": 0,
        "last_successful_response": None,
        "uploaded_files": [],
        "document_contents": {},
        "file_metadata": {},
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_response_details(details: Dict):
    """Renderiza los detalles de la respuesta de manera organizada."""
    if not details:
        return

    tabs = st.tabs(
        ["📊 Métricas", "💭 Razonamiento", "⚖️ Evaluación Ética", "🔄 Meta-análisis"]
    )

    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        with col1:
            # Manejar diferentes formatos de processing_time
            try:
                if isinstance(details.get("processing_time"), str):
                    if " " in details["processing_time"]:
                        # Formato "X.XX segundos"
                        time_value = float(details["processing_time"].split()[0])
                    else:
                        # Formato "X.XX"
                        time_value = float(details["processing_time"])
                elif isinstance(details.get("processing_time"), (int, float)):
                    time_value = float(details["processing_time"])
                else:
                    time_value = 0.0
                st.metric("Tiempo", f"{time_value:.2f}s")
            except (KeyError, ValueError, TypeError) as e:
                st.metric("Tiempo", "N/A")
                logging.warning(f"Error al procesar tiempo: {str(e)}")
        with col2:
            try:
                st.metric(
                    "Agentes",
                    str(details["performance_metrics"]["total_agents_called"]),
                )
            except (KeyError, TypeError):
                st.metric("Agentes", "N/A")
        with col3:
            try:
                st.metric("Complejidad", f"{details['complexity']:.2f}")
            except (KeyError, TypeError, ValueError):
                st.metric("Complejidad", "N/A")

    with tabs[1]:
        st.markdown("### Proceso de Razonamiento")
        st.markdown(details["initial_evaluation"])

    with tabs[2]:
        st.markdown("### Evaluación Ética")
        st.json(details["ethical_evaluation"])
        if details.get("improved_response"):
            st.info("Respuesta mejorada éticamente:")
            st.write(details["improved_response"])

    with tabs[3]:
        if details.get("meta_analysis"):
            st.markdown("### Meta-análisis")
            st.markdown(details["meta_analysis"])
        else:
            st.info("No se realizó meta-análisis para esta respuesta.")


def display_conversation_context():
    """Muestra el contexto de la conversación en una sección separada."""
    st.markdown("### 🔄 Contexto de la Conversación")
    context = st.session_state.get("context", "No hay contexto disponible")
    st.text_area("Contexto actual", value=context, height=150, disabled=True)


def export_conversation_to_md(messages, details):
    """Exporta la conversación completa a Markdown."""
    md_content = "# Conversación con MALLO\n\n"
    md_content += f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for message in messages:
        if message["role"] == "user":
            md_content += f"## 👤 Usuario\n\n{message['content']}\n\n"
        elif message["role"] == "assistant":
            md_content += f"## 🤖 Asistente\n\n{message['content']}\n\n"

            if details:
                md_content += "### 🔍 Detalles del Proceso\n\n"
                md_content += (
                    f"#### Razonamiento\n\n{details['initial_evaluation']}\n\n"
                )
                md_content += f"#### Evaluación Ética\n\n```json\n{json.dumps(details['ethical_evaluation'], indent=2)}\n```\n\n"

                if details.get("improved_response"):
                    md_content += f"#### ✨ Respuesta Mejorada\n\n{details['improved_response']}\n\n"

                if details.get("meta_analysis"):
                    md_content += (
                        f"#### 🔄 Meta-análisis\n\n{details['meta_analysis']}\n\n"
                    )

                md_content += f"#### 📝 Métricas de Rendimiento\n\n```json\n{json.dumps(details['performance_metrics'], indent=2)}\n```\n\n"
                md_content += f"#### 🌐 Contexto de la Conversación\n\n{st.session_state.get('context', 'No disponible')}\n\n"
                md_content += "---\n\n"

    # Escribir el contenido al archivo temporal
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"mallo_conversation_{timestamp}.md"
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Leer el contenido para devolverlo
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    return content, filename


def render_sidebar_content(
    system_status: Dict[str, bool],
    agent_manager: Any,
):
    """Renderiza el contenido de la barra lateral con diseño mejorado."""
    with st.sidebar:
        # Cabecera
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("assets/logo.jpg", width=50)
        with col2:
            st.markdown("### MALLO")
            st.caption("MultiAgent LLM Orchestrator")

        # Métricas principales
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Agentes", "10+")
        with col2:
            st.metric("Modelos", "20+")

        # Enlaces principales
        st.markdown(
            """
        [![ver código fuente](https://img.shields.io/badge/Repositorio%20GitHub-gris?logo=github)](https://github.com/bladealex9848/MALLO)
        ![Visitantes](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fmallollm.streamlit.app&label=Visitantes&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
        """
        )

        # Estado del Sistema
        with st.expander("🔧 Sistema", expanded=False):
            st.caption("Estado de los componentes")
            for key, value in system_status.items():
                if value:
                    st.success(key, icon="✅")
                else:
                    st.error(key, icon="❌")

            # Eliminamos la visualización de resultados de velocidad
            # ya que se toman de un archivo estático

        # Configuración de Procesamiento
        with st.expander("⚙️ Configuración de Procesamiento", expanded=False):
            # Inicializar configuración personalizada en session_state si no existe
            if "custom_processing_config" not in st.session_state:
                st.session_state.custom_processing_config = {
                    "enabled": False,
                    "stages": {
                        "initial_evaluation": True,
                        "web_search": True,
                        "meta_analysis": True,
                        "ethical_evaluation": True,
                    },
                    "models": {"primary": [], "fallback": []},
                    "agent_count": 3,
                }

            # Activar/desactivar configuración personalizada
            st.session_state.custom_processing_config["enabled"] = st.toggle(
                "Activar configuración personalizada",
                value=st.session_state.custom_processing_config["enabled"],
                help="Activa o desactiva la configuración personalizada de procesamiento",
            )

            if st.session_state.custom_processing_config["enabled"]:
                st.markdown("#### Etapas de Procesamiento")

                # Configuración de etapas
                st.session_state.custom_processing_config["stages"][
                    "initial_evaluation"
                ] = st.checkbox(
                    "Evaluación inicial",
                    value=st.session_state.custom_processing_config["stages"][
                        "initial_evaluation"
                    ],
                    help="Evalúa la complejidad y tipo de la consulta",
                )

                st.session_state.custom_processing_config["stages"]["web_search"] = (
                    st.checkbox(
                        "Búsqueda web",
                        value=st.session_state.custom_processing_config["stages"][
                            "web_search"
                        ],
                        help="Realiza búsqueda web para enriquecer el contexto",
                    )
                )

                st.session_state.custom_processing_config["stages"]["meta_analysis"] = (
                    st.checkbox(
                        "Meta-análisis",
                        value=st.session_state.custom_processing_config["stages"][
                            "meta_analysis"
                        ],
                        help="Realiza un análisis combinando múltiples respuestas",
                    )
                )

                st.session_state.custom_processing_config["stages"][
                    "ethical_evaluation"
                ] = st.checkbox(
                    "Evaluación ética",
                    value=st.session_state.custom_processing_config["stages"][
                        "ethical_evaluation"
                    ],
                    help="Evalúa el cumplimiento ético de la respuesta",
                )

                # Número de agentes a utilizar
                st.session_state.custom_processing_config["agent_count"] = st.slider(
                    "Número de agentes",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.custom_processing_config["agent_count"],
                    help="Número de agentes a utilizar para procesar la consulta",
                )

                st.markdown("#### Selección de Modelos")

                # Obtener modelos disponibles
                available_models = []

                # Cargar modelos desde config.yaml
                config = agent_manager.config

                # Agregar modelos de API (OpenAI, etc.)
                if "openai" in config and "models" in config["openai"]:
                    for model in config["openai"]["models"]:
                        available_models.append(f"api:{model}")

                # Agregar modelos de Groq
                if "groq" in config and "models" in config["groq"]:
                    for model in config["groq"]["models"]:
                        available_models.append(f"groq:{model}")

                # Agregar modelos de Together
                if "together" in config and "models" in config["together"]:
                    for model in config["together"]["models"]:
                        available_models.append(f"together:{model}")

                # Agregar modelos de DeepInfra
                if "deepinfra" in config and "models" in config["deepinfra"]:
                    for model in config["deepinfra"]["models"]:
                        available_models.append(f"deepinfra:{model}")

                # Agregar modelos de Anthropic
                if "anthropic" in config and "models" in config["anthropic"]:
                    for model in config["anthropic"]["models"]:
                        available_models.append(f"anthropic:{model}")

                # Agregar modelos de DeepSeek
                if "deepseek" in config and "models" in config["deepseek"]:
                    for model in config["deepseek"]["models"]:
                        available_models.append(f"deepseek:{model}")

                # Agregar modelos de Mistral
                if "mistral" in config and "models" in config["mistral"]:
                    for model in config["mistral"]["models"]:
                        available_models.append(f"mistral:{model}")

                # Agregar modelos de Cohere
                if "cohere" in config and "models" in config["cohere"]:
                    for model in config["cohere"]["models"]:
                        available_models.append(f"cohere:{model}")

                # Agregar modelos de OpenRouter
                if "openrouter" in config and "models" in config["openrouter"]:
                    for model in config["openrouter"]["models"]:
                        available_models.append(f"openrouter:{model}")

                # Agregar modelos locales (Ollama)
                try:
                    import subprocess

                    result = subprocess.run(
                        ["ollama", "list"], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")
                        if (
                            len(lines) > 1
                        ):  # Hay al menos un modelo (ignorando la cabecera)
                            for line in lines[1:]:  # Saltar la cabecera
                                parts = line.split()
                                if parts:  # Asegurarse de que la línea no está vacía
                                    model_name = parts[0]
                                    available_models.append(f"local:{model_name}")
                except Exception as e:
                    st.warning(f"No se pudieron cargar los modelos locales: {str(e)}")

                # Agregar agentes especializados
                if "specialized_assistants" in config and isinstance(
                    config["specialized_assistants"], dict
                ):
                    for assistant_id, assistant_info in config[
                        "specialized_assistants"
                    ].items():
                        if (
                            isinstance(assistant_info, dict)
                            and "name" in assistant_info
                        ):
                            available_models.append(f"assistant:{assistant_id}")

                # Modelos principales
                st.markdown("##### Modelos Principales")
                selected_primary = st.multiselect(
                    "Selecciona los modelos principales",
                    options=available_models,
                    default=st.session_state.custom_processing_config["models"][
                        "primary"
                    ],
                    help="Modelos que se utilizarán como primera opción",
                )
                st.session_state.custom_processing_config["models"][
                    "primary"
                ] = selected_primary

                # Modelos de respaldo
                st.markdown("##### Modelos de Respaldo")
                selected_fallback = st.multiselect(
                    "Selecciona los modelos de respaldo",
                    options=available_models,
                    default=st.session_state.custom_processing_config["models"][
                        "fallback"
                    ],
                    help="Modelos que se utilizarán si los principales fallan",
                )
                st.session_state.custom_processing_config["models"][
                    "fallback"
                ] = selected_fallback

                # Botones para guardar/cargar configuración
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Guardar configuración"):
                        try:
                            with open("custom_config.json", "w") as f:
                                json.dump(
                                    st.session_state.custom_processing_config,
                                    f,
                                    indent=2,
                                )
                            st.success("Configuración guardada correctamente")
                        except Exception as e:
                            st.error(f"Error al guardar la configuración: {str(e)}")

                with col2:
                    if st.button("Cargar configuración"):
                        try:
                            if os.path.exists("custom_config.json"):
                                with open("custom_config.json", "r") as f:
                                    st.session_state.custom_processing_config = (
                                        json.load(f)
                                    )
                                st.success("Configuración cargada correctamente")
                            else:
                                st.warning("No hay configuración guardada")
                        except Exception as e:
                            st.error(f"Error al cargar la configuración: {str(e)}")

                # Botón para actualizar modelos
                if st.button("Actualizar modelos disponibles"):
                    try:
                        # Importar el módulo model_loader
                        import model_loader

                        # Actualizar caché de modelos
                        openrouter_models = model_loader.load_models_from_openrouter()
                        groq_models = model_loader.load_models_from_groq()
                        ollama_models = model_loader.load_models_from_ollama()

                        # Actualizar la lista de modelos disponibles
                        updated_models = []

                        # Agregar modelos de OpenRouter
                        for model in openrouter_models:
                            model_id = model.get("id", "")
                            if model_id:
                                updated_models.append(f"openrouter:{model_id}")

                        # Agregar modelos de Groq
                        for model in groq_models:
                            model_id = model.get("id", "")
                            if model_id:
                                updated_models.append(f"groq:{model_id}")

                        # Agregar modelos de Ollama
                        for model in ollama_models:
                            model_id = model.get("id", "")
                            if model_id:
                                updated_models.append(f"local:{model_id}")

                        # Agregar los modelos actualizados a la lista existente
                        for model in updated_models:
                            if model not in available_models:
                                available_models.append(model)

                        st.success(
                            f"Modelos actualizados: OpenRouter ({len(openrouter_models)}), Groq ({len(groq_models)}), Ollama ({len(ollama_models)})"
                        )
                    except Exception as e:
                        st.error(f"Error al actualizar modelos: {str(e)}")

        # Capacidades
        with st.expander("💡 Capacidades", expanded=False):
            features = {
                "🤖 Múltiples Modelos": "Integración con principales proveedores de IA",
                "🔍 Análisis Contextual": "Comprensión profunda de consultas",
                "🌐 Búsqueda Web": "Información actualizada en tiempo real",
                "⚖️ Evaluación Ética": "Respuestas alineadas con principios éticos",
                "🔄 Meta-análisis": "Síntesis de múltiples fuentes",
                "🎯 Prompts Adaptados": "Especialización por tipo de consulta",
            }

            for title, description in features.items():
                st.markdown(f"**{title}**")
                st.caption(description)

        # Gestión de documentos
        with st.expander("📚 Gestión de Documentos", expanded=False):
            manage_document_context()

        # Acerca de
        with st.expander("ℹ️ Acerca de", expanded=False):
            st.markdown(
                """
            MALLO es un orquestador avanzado de IAs que selecciona y coordina
            múltiples modelos de lenguaje para proporcionar respuestas óptimas
            basadas en el contexto y complejidad de cada consulta.
            """
            )

        # Desarrollador
        st.markdown("---")
        st.markdown("### 👨‍💻 Desarrollador")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("assets/profile.jpg", width=60)
        with col2:
            st.markdown("#### Alexander Oviedo Fadul")
            st.caption("Developer & Legal Tech")

        # Enlaces sociales
        st.markdown("##### Contacto")
        social_links = {
            "🌐 Website": "https://alexanderoviedofadul.dev",
            "💼 LinkedIn": "https://linkedin.com/in/alexander-oviedo-fadul",
            "📱 WhatsApp": "https://wa.me/573015930519",
            "📧 Email": "mailto:alexander.oviedo.fadul@gmail.com",
            "🐙 GitHub": "https://github.com/bladealex9848",
        }

        for platform, link in social_links.items():
            st.markdown(f"[{platform}]({link})")


# Resumen de la conversación con un límite de longitud máximo (500 caracteres)
# Si el contexto supera el límite, se utiliza cohere o OpenAI para resumirlo
def summarize_conversation(
    previous_context, user_input, response, agent_manager, config, max_length=500
):
    new_content = f"Usuario: {user_input}\nAsistente: {response}"
    updated_context = f"{previous_context}\n\n{new_content}".strip()

    if len(updated_context) > max_length:
        summary_prompt = f"Resume la siguiente conversación manteniendo los puntos clave:\n\n{updated_context}"

        try:
            # Intenta usar openrouter primero
            # summary = agent_manager.process_with_openrouter(summary_prompt, config['openrouter']['fast_models'])
            # Intenta usar cohere primero
            summary = agent_manager.process_query(
                summary_prompt, "cohere", config["cohere"]["default_model"]
            )
        except Exception as e:
            log_error(
                f"Error al usar cohere para resumen: {str(e)}. Usando OpenAI como respaldo."
            )
            try:
                # Si cohere falla, usa OpenAI como respaldo
                summary = agent_manager.process_query(
                    summary_prompt, "api", config["openai"]["default_model"]
                )
            except Exception as e:
                log_error(
                    f"Error al usar OpenAI para resumen: {str(e)}. Devolviendo contexto sin resumir."
                )
                return updated_context

        return summary

    return updated_context


def evaluate_ethical_compliance(response: str, prompt_type: str) -> Dict[str, Any]:
    """
    Evalúa el cumplimiento ético y legal de la respuesta generada.

    Args:
    response (str): La respuesta generada por el sistema.
    prompt_type (str): El tipo de prompt utilizado para generar la respuesta.

    Returns:
    Dict[str, Any]: Un diccionario con los resultados de la evaluación.
    """
    evaluation = {
        "sesgo_detectado": False,
        "privacidad_respetada": True,
        "transparencia": True,
        "alineacion_derechos_humanos": True,
        "responsabilidad": True,
        "explicabilidad": True,
    }

    # Verificar sesgos (esto requeriría un modelo más sofisticado en la práctica)
    if any(
        palabra in response.lower()
        for palabra in ["todos los hombres", "todas las mujeres"]
    ):
        evaluation["sesgo_detectado"] = True

    # Verificar privacidad (ejemplo simplificado)
    if any(
        dato in response
        for dato in ["número de identificación", "dirección", "teléfono"]
    ):
        evaluation["privacidad_respetada"] = False

    # Verificar transparencia
    if "Esta respuesta fue generada por IA" not in response:
        evaluation["transparencia"] = False

    # La alineación con derechos humanos, responsabilidad y explicabilidad
    # requerirían análisis más complejos en un sistema real

    return evaluation


# Evaluar la respuesta del modelo de lenguaje y proporcionar retroalimentación
def evaluate_response(agent_manager, config, evaluation_type, query, response=None):
    eval_config = config["evaluation_models"][evaluation_type]

    if evaluation_type == "initial":
        evaluation_prompt = f"""
        Analiza la siguiente consulta y proporciona una guía detallada para responderla:

        Consulta: {query}

        Tu tarea es:
        1. Identificar los puntos clave que deben abordarse en la respuesta.
        2. Determinar si se necesita información actualizada o reciente para responder adecuadamente. Si es así, indica "BUSQUEDA_WEB: SI" en tu respuesta.
        3. Evaluar la complejidad de la consulta en una escala de 0 a 1, donde 0 es muy simple y 1 es muy compleja. Indica "COMPLEJIDAD: X" donde X es el valor numérico.
        4. Decidir si la consulta requiere conocimientos de múltiples dominios o fuentes. Si es así, indica "MOA: SI" en tu respuesta.
        5. Sugerir fuentes de información relevantes para la consulta.
        6. Proponer un esquema o estructura para la respuesta.
        7. Indicar cualquier consideración especial o contexto importante para la consulta.

        Por favor, proporciona tu análisis y guía en un formato claro y estructurado.
        """
    else:  # evaluation_type == 'final'
        evaluation_prompt = f"""
        Evalúa la siguiente respuesta a la consulta dada:

        Consulta: {query}

        Respuesta:
        {response}

        Tu tarea es:
        1. Determinar si la respuesta es apropiada y precisa para la consulta.
        2. Identificar cualquier información faltante o imprecisa.
        3. Evaluar la claridad y estructura de la respuesta.
        4. Si es necesario, proporcionar una versión mejorada de la respuesta.

        Por favor, proporciona tu evaluación en un formato claro y estructurado, incluyendo una versión mejorada de la respuesta si lo consideras necesario.
        """

    for attempt in range(3):  # Intentar hasta 3 veces
        try:
            if attempt == 0:
                evaluation = agent_manager.process_query(
                    evaluation_prompt, eval_config["api"], eval_config["model"]
                )
            elif attempt == 1:
                evaluation = agent_manager.process_query(
                    evaluation_prompt,
                    eval_config["backup_api"],
                    eval_config["backup_model"],
                )
            else:
                evaluation = agent_manager.process_query(
                    evaluation_prompt,
                    eval_config["backup_api2"],
                    eval_config["backup_model2"],
                )

            if "Error al procesar" not in evaluation:
                return evaluation
        except Exception as e:
            log_error(
                f"Error en evaluación {evaluation_type} con {'modelo principal' if attempt == 0 else 'modelo de respaldo'}: {str(e)}"
            )

    return "No se pudo realizar la evaluación debido a múltiples errores en los modelos de evaluación."


def process_user_input(
    user_input: str, config: Dict[str, Any], agent_manager: Any
) -> Tuple[str, Dict[str, Any]]:
    """
    Procesa la entrada del usuario y genera una respuesta usando los agentes apropiados.

    Args:
        user_input (str): La consulta del usuario
        config (Dict[str, Any]): Configuración del sistema
        agent_manager (Any): Instancia del gestor de agentes

    Returns:
        Tuple[str, Dict[str, Any]]: (respuesta, detalles del procesamiento)
    """
    try:
        # Verificar si se está usando configuración personalizada
        use_custom_config = False
        custom_config = None
        if "custom_processing_config" in st.session_state:
            custom_config = st.session_state.custom_processing_config
            use_custom_config = custom_config.get("enabled", False)

        conversation_context = st.session_state.get("context", "")

        # Incluir contexto de documentos si existen
        document_context = ""
        if (
            "document_contents" in st.session_state
            and st.session_state.document_contents
        ):
            document_context = "\n\n### Contexto de documentos procesados:\n\n"
            for doc_name, doc_content in st.session_state.document_contents.items():
                # Extraer el texto del documento procesado por OCR
                if isinstance(doc_content, dict):
                    if "text" in doc_content:
                        # Limitamos el contenido para no exceder el contexto
                        doc_text = (
                            doc_content["text"][:5000] + "..."
                            if len(doc_content["text"]) > 5000
                            else doc_content["text"]
                        )
                        document_context += (
                            f"-- Documento: {doc_name} --\n{doc_text}\n\n"
                        )
                    elif "error" in doc_content and "raw_response" in doc_content:
                        # Intentar extraer texto de la respuesta cruda si está disponible
                        raw_response = doc_content["raw_response"]
                        if isinstance(raw_response, dict) and "text" in raw_response:
                            doc_text = (
                                raw_response["text"][:5000] + "..."
                                if len(raw_response["text"]) > 5000
                                else raw_response["text"]
                            )
                            document_context += (
                                f"-- Documento: {doc_name} --\n{doc_text}\n\n"
                            )
                        else:
                            document_context += f"-- Documento: {doc_name} -- (Error al extraer texto: {doc_content['error']})\n\n"
                    else:
                        document_context += f"-- Documento: {doc_name} -- (No se pudo extraer texto)\n\n"
                else:
                    document_context += (
                        f"-- Documento: {doc_name} -- (Formato no reconocido)\n\n"
                    )

        # Combinar contexto de conversación, documentos y consulta actual
        enriched_query = f"{conversation_context}\n\n{document_context}\n\nNueva consulta: {user_input}"

        # Verificar caché
        cached_response = get_cached_response(enriched_query)
        if cached_response:
            st.success("💾 Respuesta recuperada de caché")
            return cached_response

        progress_placeholder = st.empty()

        # Iniciar procesamiento
        start_time = time.time()
        progress_placeholder.write("🔍 Evaluando consulta...")

        # Evaluación inicial
        initial_evaluation = ""
        if not use_custom_config or custom_config["stages"].get(
            "initial_evaluation", True
        ):
            initial_evaluation = evaluate_response(
                agent_manager, config, "initial", enriched_query
            )

            # Análisis de complejidad y necesidades
            complexity, needs_web_search, needs_moa, prompt_type = (
                evaluate_query_complexity(initial_evaluation, "")
            )
            prompt_type = agent_manager.validate_prompt_type(user_input, prompt_type)
        else:
            # Valores por defecto si se omite la evaluación inicial
            complexity = 0.5
            needs_web_search = False
            needs_moa = False
            prompt_type = "general"

        # Búsqueda web si es necesaria
        web_context = ""
        if needs_web_search and (
            not use_custom_config or custom_config["stages"].get("web_search", True)
        ):
            progress_placeholder.write("🌐 Realizando búsqueda web...")
            web_context = perform_web_search(user_input)
            enriched_query = f"{enriched_query}\nContexto web: {web_context}"

        # Selección de agentes
        progress_placeholder.write("🤖 Seleccionando agentes...")

        # Determinar qué agentes usar
        prioritized_agents = []

        if use_custom_config and custom_config["models"]["primary"]:
            # Usar modelos personalizados
            for model_str in custom_config["models"]["primary"]:
                if len(prioritized_agents) < custom_config["agent_count"]:
                    # Manejar el formato de modelo de manera más robusta
                    try:
                        # Dividir solo en la primera ocurrencia de ":"
                        parts = model_str.split(":", 1)
                        if len(parts) == 2:
                            agent_type, agent_id = parts
                            prioritized_agents.append(
                                (
                                    agent_type,
                                    agent_id,
                                    f"{agent_type.capitalize()} {agent_id}",
                                )
                            )
                        else:
                            logging.warning(
                                f"Formato de modelo incorrecto: {model_str}"
                            )
                    except Exception as e:
                        logging.error(f"Error al procesar modelo {model_str}: {str(e)}")
        else:
            # Usar selección estándar de agentes
            specialized_agent = agent_manager.select_specialized_agent(enriched_query)
            general_agents = agent_manager.get_prioritized_agents(
                enriched_query, complexity, prompt_type
            )

            # Priorización de agentes
            if specialized_agent:
                prioritized_agents.append(
                    (
                        specialized_agent["type"],
                        specialized_agent["id"],
                        specialized_agent["name"],
                    )
                )

            # Añadir agentes generales
            max_agents = 2
            if use_custom_config:
                max_agents = custom_config["agent_count"]

            for agent in general_agents:
                if len(prioritized_agents) >= max_agents:
                    break
                if agent not in prioritized_agents:
                    prioritized_agents.append(agent)

            prioritized_agents = prioritized_agents[:max_agents]

        # Procesamiento con agentes
        agent_results = []
        for agent_type, agent_id, agent_name in prioritized_agents:
            progress_placeholder.write(f"⚙️ Procesando con {agent_name}...")
            try:
                enriched_query_with_prompt = agent_manager.apply_specialized_prompt(
                    enriched_query, prompt_type
                )
                result = agent_manager.process_query(
                    enriched_query_with_prompt, agent_type, agent_id, prompt_type
                )
                agent_results.append(
                    {
                        "agent": agent_type,
                        "model": agent_id,
                        "name": agent_name,
                        "status": "success",
                        "response": result,
                    }
                )
            except Exception as e:
                logger.error(f"Error en el procesamiento con {agent_name}: {str(e)}")
                # Intentar con modelos de respaldo si están configurados
                fallback_success = False
                if use_custom_config and custom_config["models"]["fallback"]:
                    for fallback_model_str in custom_config["models"]["fallback"]:
                        try:
                            # Dividir solo en la primera ocurrencia de ":"
                            parts = fallback_model_str.split(":", 1)
                            if len(parts) == 2:
                                fallback_agent_type, fallback_agent_id = parts
                                fallback_agent_name = f"{fallback_agent_type.capitalize()} {fallback_agent_id} (Respaldo)"
                                progress_placeholder.write(
                                    f"⚙️ Intentando con respaldo: {fallback_agent_name}..."
                                )
                            else:
                                logging.warning(
                                    f"Formato de modelo de respaldo incorrecto: {fallback_model_str}"
                                )
                                continue

                            result = agent_manager.process_query(
                                enriched_query_with_prompt,
                                fallback_agent_type,
                                fallback_agent_id,
                                prompt_type,
                            )

                            agent_results.append(
                                {
                                    "agent": fallback_agent_type,
                                    "model": fallback_agent_id,
                                    "name": fallback_agent_name,
                                    "status": "success",
                                    "response": result,
                                }
                            )
                            # Si el respaldo funciona, salir del bucle
                            fallback_success = True
                            break
                        except Exception as fallback_e:
                            logger.error(
                                f"Error en el respaldo {fallback_agent_name}: {str(fallback_e)}"
                            )
                            continue

                # Si no hay respaldo o todos fallaron, registrar el error original
                if not fallback_success:
                    agent_results.append(
                        {
                            "agent": agent_type,
                            "model": agent_id,
                            "name": agent_name,
                            "status": "error",
                            "response": str(e),
                        }
                    )

        # Verificar respuestas exitosas
        successful_responses = [r for r in agent_results if r["status"] == "success"]

        if not successful_responses:
            raise ValueError("No se pudo obtener una respuesta válida de ningún agente")

        # Meta-análisis si es necesario
        meta_analysis_result = None
        if (
            needs_moa
            and len(successful_responses) > 1
            and (
                not use_custom_config
                or custom_config["stages"].get("meta_analysis", True)
            )
        ):
            progress_placeholder.write("🔄 Realizando meta-análisis...")
            meta_analysis_result = agent_manager.meta_analysis(
                user_input,
                [r["response"] for r in successful_responses],
                initial_evaluation,
                "",
            )
            final_response = agent_manager.process_query(
                f"Basándote en este meta-análisis, proporciona una respuesta conversacional y directa a la pregunta '{user_input}'. La respuesta debe ser natural, como si estuvieras charlando con un amigo, sin usar frases como 'Basándome en el análisis' o 'La respuesta es'. Simplemente responde de manera clara y concisa: {meta_analysis_result}",
                agent_manager.meta_analysis_api,
                agent_manager.meta_analysis_model,
            )
        else:
            final_response = successful_responses[0]["response"]

        # Evaluación ética
        ethical_evaluation = {}
        improved_response = None
        improved_ethical_evaluation = None
        if not use_custom_config or custom_config["stages"].get(
            "ethical_evaluation", True
        ):
            progress_placeholder.write("⚖️ Evaluando cumplimiento ético...")
            ethical_evaluation = evaluate_ethical_compliance(
                final_response, prompt_type
            )

            # Mejora ética si es necesaria
            if any(not value for value in ethical_evaluation.values()):
                progress_placeholder.write("✨ Mejorando respuesta...")
                specialized_assistant = agent_manager.get_specialized_assistant(
                    "asst_F33bnQzBVqQLcjveUTC14GaM"
                )
                enhancement_prompt = f"""
                Analiza la siguiente respuesta y su evaluación ética:

                Respuesta: {final_response}

                Evaluación ética: {json.dumps(ethical_evaluation, indent=2)}

                Por favor, modifica la respuesta para mejorar su alineación con principios éticos y legales,
                abordando cualquier preocupación identificada en la evaluación. Asegúrate de que la respuesta sea
                transparente sobre el uso de IA, libre de sesgos, y respetuosa de los derechos humanos y la privacidad.
                """
                improved_response = agent_manager.process_query(
                    enhancement_prompt, "assistant", specialized_assistant["id"]
                )
                improved_ethical_evaluation = evaluate_ethical_compliance(
                    improved_response, prompt_type
                )

        # Evaluación final
        final_evaluation = ""
        if not use_custom_config or custom_config["stages"].get(
            "initial_evaluation", True
        ):
            progress_placeholder.write("📝 Evaluación final...")
            final_evaluation = evaluate_response(
                agent_manager, config, "final", user_input, final_response
            )

        # Cálculo del tiempo de procesamiento
        processing_time = time.time() - start_time

        # Preparar detalles de la respuesta
        details = {
            "selected_agents": [
                {"agent": r["agent"], "model": r["model"], "name": r["name"]}
                for r in agent_results
            ],
            "processing_time": f"{processing_time:.2f} segundos",
            "complexity": complexity,
            "needs_web_search": needs_web_search,
            "needs_moa": needs_moa,
            "web_context": web_context,
            "prompt_type": prompt_type,
            "initial_evaluation": initial_evaluation,
            "agent_processing": agent_results,
            "final_evaluation": final_evaluation,
            "ethical_evaluation": ethical_evaluation,
            "improved_response": improved_response,
            "improved_ethical_evaluation": improved_ethical_evaluation,
            "performance_metrics": {
                "total_agents_called": len(agent_results),
                "successful_responses": len(successful_responses),
                "failed_responses": len(agent_results) - len(successful_responses),
                "average_response_time": f"{processing_time:.2f} segundos",
            },
            "meta_analysis": meta_analysis_result,
            "final_response": final_response,
            "custom_config_used": use_custom_config,
        }

        # Actualizar contexto
        new_context = summarize_conversation(
            conversation_context, user_input, final_response, agent_manager, config
        )
        st.session_state["context"] = new_context

        # Guardar en caché
        cache_response(enriched_query, (final_response, details))

        # Limpiar placeholder de progreso
        progress_placeholder.empty()

        return final_response, details

    except Exception as e:
        logger.error(f"Error en process_user_input: {str(e)}")
        logger.error(traceback.format_exc())
        return (
            "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, intenta de nuevo.",
            {"error": str(e)},
        )


@st.cache_resource
def initialize_core_components() -> Dict[str, Any]:
    """
    Inicializa y cachea los componentes principales del sistema.
    """
    try:
        start_time = time.time()
        logger.info("Iniciando inicialización de componentes principales...")

        # Cargar configuración y componentes
        config = load_config()
        system_status = initialize_system(config)
        agent_manager = AgentManager(config)
        speed_test_results = load_speed_test_results()

        initialization_time = time.time() - start_time
        logger.info(
            f"Componentes principales inicializados en {initialization_time:.2f} segundos"
        )

        return {
            "config": config,
            "system_status": system_status,
            "agent_manager": agent_manager,
            "speed_test_results": speed_test_results,
        }
    except Exception as e:
        error_msg = f"Error en la inicialización de componentes: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def initialize_session_state():
    """Inicializa el estado de la sesión con valores predeterminados."""
    default_states = {
        "messages": [],
        "context": "",
        "show_settings": False,
        "last_details": {},
        "error_count": 0,
        "last_successful_response": None,
        "uploaded_files": [],
        "document_contents": {},
        "file_metadata": {},
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    try:
        # Inicialización optimizada con caché
        initialize_session_state()
        core_components = initialize_core_components()

        # Extraer componentes
        config = core_components["config"]
        system_status = core_components["system_status"]
        agent_manager = core_components["agent_manager"]
        speed_test_results = core_components["speed_test_results"]

        # Renderizar barra lateral
        render_sidebar_content(system_status, agent_manager)

        # Interfaz principal
        st.title("MALLO: MultiAgent LLM Orchestrator")

        # Contenedor principal para el chat
        chat_container = st.container()

        # Contenedor separado para los detalles y el contexto
        details_container = st.container()

        with chat_container:
            # Interfaz de chat
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        with details_container:
            if st.session_state["messages"]:
                # Tabs para organizar la información adicional
                info_tabs = st.tabs(["💡 Detalles", "📤 Exportar", "📝 Contexto"])

                with info_tabs[0]:
                    if st.session_state.get("last_details"):
                        render_response_details(st.session_state["last_details"])

                with info_tabs[1]:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.download_button(
                            "📥 Exportar Conversación Completa",
                            *export_conversation_to_md(
                                st.session_state["messages"],
                                st.session_state.get("last_details", {}),
                            ),
                            mime="text/markdown",
                        ):
                            st.success("✅ Conversación exportada exitosamente")

                    with col2:
                        if st.session_state["messages"]:
                            last_response = next(
                                (
                                    msg["content"]
                                    for msg in reversed(st.session_state["messages"])
                                    if msg["role"] == "assistant"
                                ),
                                None,
                            )
                            if last_response and st.download_button(
                                "📥 Exportar Última Respuesta",
                                last_response,
                                f"respuesta_{time.strftime('%Y%m%d-%H%M%S')}.md",
                                mime="text/markdown",
                            ):
                                st.success("✅ Respuesta exportada exitosamente")

                with info_tabs[2]:
                    display_conversation_context()

        # Mostrar documentos cargados
        if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
            with st.expander("📄 Documentos cargados", expanded=False):
                st.subheader("📚 Documentos disponibles")
                for filename in st.session_state.uploaded_files:
                    st.markdown(f"📄 **{filename}**")

                if st.button("Limpiar documentos"):
                    st.session_state.uploaded_files = []
                    st.session_state.document_contents = {}
                    st.session_state.file_metadata = {}
                    st.success("Documentos eliminados correctamente")
                    st.rerun()

        # Input del usuario con soporte para archivos
        user_prompt = st.chat_input(
            "¿En qué puedo ayudarte hoy? También puedes adjuntar archivos",
            accept_file=True,
            file_type=["pdf", "jpg", "jpeg", "png", "txt"],
        )

        if user_prompt:
            try:
                # Procesar texto y archivos adjuntos
                user_text = (
                    user_prompt.text if hasattr(user_prompt, "text") else user_prompt
                )
                user_files = user_prompt.files if hasattr(user_prompt, "files") else []

                # Mostrar indicador de procesamiento
                with st.spinner("Procesando tu consulta..."):
                    # Procesar archivos adjuntos si existen
                    if user_files:
                        mistral_api_key = get_secret("MISTRAL_API_KEY")
                        if not mistral_api_key:
                            st.warning(
                                "Se requiere una clave API de Mistral para procesar documentos. Por favor, configura la clave en tus secretos."
                            )
                        else:
                            with st.status("Procesando archivos adjuntos..."):
                                valid_files = 0
                                file_names = []

                                for file in user_files:
                                    # Validar el formato del archivo
                                    is_valid, file_type, error_message = (
                                        validate_file_format(file)
                                    )

                                    if not is_valid:
                                        st.error(
                                            f"Error en archivo {file.name}: {error_message}"
                                        )
                                        continue

                                    # Si el archivo es válido, procesarlo
                                    if file.name not in st.session_state.uploaded_files:
                                        st.session_state.uploaded_files.append(
                                            file.name
                                        )

                                    # Leer el contenido del archivo
                                    file_bytes = file.read()
                                    file.seek(0)  # Restaurar el puntero del archivo

                                    try:
                                        ocr_results = process_document_with_mistral_ocr(
                                            mistral_api_key,
                                            file_bytes,
                                            file_type,
                                            file.name,
                                        )

                                        if ocr_results and "error" not in ocr_results:
                                            st.session_state.document_contents[
                                                file.name
                                            ] = ocr_results
                                            st.success(
                                                f"Documento {file.name} procesado correctamente"
                                            )
                                            valid_files += 1
                                            file_names.append(file.name)
                                        else:
                                            error_msg = ocr_results.get(
                                                "error",
                                                "Error desconocido durante el procesamiento",
                                            )
                                            st.warning(
                                                f"No se pudo extraer texto completo de {file.name}: {error_msg}"
                                            )
                                            st.session_state.document_contents[
                                                file.name
                                            ] = ocr_results
                                    except Exception as e:
                                        st.error(
                                            f"Error procesando {file.name}: {str(e)}"
                                        )

                                # Agregar información de archivos al mensaje del usuario si se procesaron archivos
                                if valid_files > 0:
                                    file_info = f"\n\n[Archivos adjuntos: {', '.join(file_names)}]"
                                    user_text = (
                                        f"{user_text}{file_info}"
                                        if user_text
                                        else f"He adjuntado los siguientes documentos: {', '.join(file_names)}. Por favor, análiza su contenido."
                                    )

                    # Procesar la consulta del usuario
                    response, details = process_user_input(
                        user_text, config, agent_manager
                    )
                    st.session_state["last_details"] = details
                    st.session_state["error_count"] = 0

                    if response:
                        # Actualizar el historial de la conversación
                        st.session_state["messages"].append(
                            {
                                "role": "user",
                                "content": user_text,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": response,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                        st.session_state["last_successful_response"] = response

                        # Recargar la página para mostrar la nueva respuesta
                        st.rerun()

            except Exception as e:
                handle_error(e)

    except Exception as e:
        handle_critical_error(e)


def handle_error(e: Exception):
    """Maneja errores durante el procesamiento de la consulta."""
    st.session_state["error_count"] += 1

    # Registrar el error completo en el log para depuración
    logger.error(f"Error procesando entrada del usuario: {str(e)}")
    logger.error(traceback.format_exc())

    # Mensaje de error más amigable para el usuario
    error_message = (
        "Ha ocurrido un error al procesar tu consulta. "
        "Por favor, intenta de nuevo o reformula tu pregunta."
    )

    # Si hay errores persistentes, dar más información
    if st.session_state["error_count"] > 3:
        error_message += (
            "\n\n⚠️ Parece que estamos teniendo problemas técnicos persistentes. "
            "Te sugerimos:\n"
            "1. Intentar más tarde\n"
            "2. Verificar tu conexión a internet\n"
            "3. Contactar al soporte técnico"
        )

        if st.session_state.get("last_successful_response"):
            st.info(
                "👉 Mientras tanto, puedes revisar la última respuesta exitosa "
                "o exportar la conversación hasta este punto."
            )

    # Mostrar el mensaje de error al usuario
    st.error(error_message)

    # Registrar información adicional para depuración
    logger.error(f"Error Count: {st.session_state['error_count']}")
    logger.error(
        f"Last Successful Response Available: {bool(st.session_state.get('last_successful_response'))}"
    )


def handle_critical_error(e: Exception):
    """Maneja errores críticos de la aplicación."""
    logger.error(f"Error crítico en la aplicación: {str(e)}")
    logger.error(traceback.format_exc())

    st.error(
        "🚨 Error crítico en la aplicación\n\n"
        "Ha ocurrido un error inesperado. Por favor:\n"
        "1. Recarga la página\n"
        "2. Verifica tu conexión\n"
        "3. Si el problema persiste, contacta al soporte técnico\n\n"
        "Tus datos de conversación están seguros y se intentarán recuperar en la próxima sesión."
    )

    try:
        with open("error_recovery.json", "w") as f:
            json.dump(
                {
                    "messages": st.session_state.get("messages", []),
                    "context": st.session_state.get("context", ""),
                    "last_details": st.session_state.get("last_details", {}),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
            )
    except Exception as save_error:
        logger.error(f"Error al guardar estado para recuperación: {str(save_error)}")


if __name__ == "__main__":
    main()
