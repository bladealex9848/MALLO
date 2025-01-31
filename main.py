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
import time
import asyncio
import polars as pl
import random
import logging
import traceback
from typing import Tuple, Dict, Any, Optional

from load_secrets import load_secrets, get_secret, secrets

# CSS personalizado para interfaz moderna - Añadir después de set_streamlit_page_config()
st.markdown(
    """
<style>
    .main {
        background-color: transparent !important;
    }
    .stApp {
        background-color: transparent !important;
    }
    .stMarkdown h1 {
        color: inherit;
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .export-button {
        background-color: rgba(240, 242, 246, 0.1);
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
    }
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
    div[data-testid="stExpander"] {
        border: 1px solid rgba(240, 242, 246, 0.1);
        border-radius: 10px;
        margin-top: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Carga todos los secretos al inicio de la aplicación
load_secrets()


def load_config():
    try:
        with open("config.yaml", "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("No se pudo encontrar el archivo de configuración 'config.yaml'.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error al leer el archivo de configuración: {str(e)}")
        st.stop()


def load_speed_test_results():
    try:
        with open("model_speeds.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def display_speed_test_results(results):
    st.sidebar.title("Resultados del Test de Velocidad")
    data = [
        {
            "API": api,
            "Modelo": model["model"],
            "Velocidad": f"{model['speed']:.4f}",
        }
        for api, models in results.items()
        for model in models
    ]
    df = pl.DataFrame(data).sort("Velocidad")
    with st.sidebar.expander("Mostrar Resultados"):
        st.dataframe(df, use_container_width=True, hide_index=True)


async def process_with_agent(agent_manager, query, agent_type, agent_id):
    try:
        response = await asyncio.to_thread(
            agent_manager.process_query, query, agent_type, agent_id
        )
        return {
            "agent": agent_type,
            "model": agent_id,
            "status": "success",
            "response": response,
        }
    except Exception as e:
        log_error(f"Error processing query with {agent_type}:{agent_id}: {str(e)}")
        return {
            "agent": agent_type,
            "model": agent_id,
            "status": "error",
            "response": str(e),
        }


async def process_with_multiple_agents(query, agent_manager, num_agents=3):
    agents = agent_manager.get_agent_priority()
    random.shuffle(agents)
    tasks = [
        process_with_agent(agent_manager, query, agent_type, agent_id)
        for agent_type, agent_id in agents[:num_agents]
    ]
    results = await asyncio.gather(*tasks)
    valid_responses = [
        r for r in results if r["status"] == "success" and r["response"]
    ]
    best_response = (
        max(valid_responses, key=lambda x: len(x["response"]))
        if valid_responses
        else None
    )
    return best_response, results


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


def process_user_input(
    user_input: str, config: Dict[str, Any], agent_manager: Any
) -> Tuple[str, Dict[str, Any]]:
    try:
        conversation_context = st.session_state.get("context", "")
        enriched_query = f"{conversation_context}\n\nNueva consulta: {user_input}"

        cached_response = get_cached_response(enriched_query)
        if cached_response:
            st.success("Respuesta obtenida de la caché")
            return cached_response

        with st.spinner("Procesando tu consulta..."):
            progress_placeholder = st.empty()
            start_time = time.time()  # Definimos start_time aquí

            # Mostrar etapas del proceso
            progress_placeholder.write("Evaluando consulta...")
            initial_evaluation = evaluate_response(
                agent_manager, config, "initial", enriched_query
            )

            (
                complexity,
                needs_web_search,
                needs_moa,
                prompt_type,
            ) = evaluate_query_complexity(initial_evaluation, "")
            prompt_type = agent_manager.validate_prompt_type(user_input, prompt_type)

            if needs_web_search:
                progress_placeholder.write("Realizando búsqueda web...")
                web_context = perform_web_search(user_input)
                enriched_query = f"{enriched_query}\nContexto web: {web_context}"
            else:
                web_context = ""

            progress_placeholder.write("Seleccionando agentes...")
            specialized_agent = agent_manager.select_specialized_agent(enriched_query)
            general_agents = agent_manager.get_prioritized_agents(
                enriched_query, complexity, prompt_type
            )

            prioritized_agents = []
            if specialized_agent:
                prioritized_agents.append(
                    (
                        specialized_agent["type"],
                        specialized_agent["id"],
                        specialized_agent["name"],
                    )
                )

            for agent in general_agents:
                if len(prioritized_agents) >= 2:
                    break
                if agent not in prioritized_agents:
                    prioritized_agents.append(agent)

            prioritized_agents = prioritized_agents[:2]

            agent_results = []
            for agent_type, agent_id, agent_name in prioritized_agents:
                progress_placeholder.write(f"Procesando con {agent_name}...")
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
                    logging.error(f"Error processing with {agent_name}: {str(e)}")
                    agent_results.append(
                        {
                            "agent": agent_type,
                            "model": agent_id,
                            "name": agent_name,
                            "status": "error",
                            "response": str(e),
                        }
                    )

            successful_responses = [
                r for r in agent_results if r["status"] == "success"
            ]

            if not successful_responses:
                raise ValueError(
                    "No se pudo obtener una respuesta válida de ningún agente"
                )

            if needs_moa and len(successful_responses) > 1:
                progress_placeholder.write("Realizando meta-análisis...")
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

            progress_placeholder.write("Evaluando cumplimiento ético...")
            ethical_evaluation = evaluate_ethical_compliance(
                final_response, prompt_type
            )

            if any(not value for value in ethical_evaluation.values()):
                progress_placeholder.write("Mejorando respuesta...")
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
            else:
                improved_response = None
                improved_ethical_evaluation = None

            progress_placeholder.write("Evaluación final...")
            final_evaluation = evaluate_response(
                agent_manager, config, "final", user_input, final_response
            )

            processing_time = time.time() - start_time

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
                "meta_analysis": meta_analysis_result
                if needs_moa and len(successful_responses) > 1
                else None,
                "final_response": final_response,
            }

            new_context = summarize_conversation(
                conversation_context,
                user_input,
                final_response,
                agent_manager,
                config,
            )
            st.session_state["context"] = new_context

            cache_response(enriched_query, (final_response, details))

            progress_placeholder.empty()
            return final_response, details

    except Exception as e:
        logging.error(f"Se ha producido un error inesperado: {str(e)}")
        logging.error(traceback.format_exc())
        return (
            "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, intenta de nuevo.",
            {"error": str(e)},
        )


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


# Resumen de la conversación con un límite de longitud máximo (500 caracteres)
# Si el contexto supera el límite, se utiliza cohere o OpenAI para resumirlo
def summarize_conversation(
    previous_context, user_input, response, agent_manager, config, max_length=500
):
    new_content = f"Usuario: {user_input}\nAsistente: {response}"
    updated_context = f"{previous_context}\n\n{new_content}".strip()

    if len(updated_context) > max_length:
        summary_prompt = (
            f"Resume la siguiente conversación manteniendo los puntos clave:\n\n{updated_context}"
        )

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


def initialize_session_state():
    """
    Inicializa el estado de la sesión con valores predeterminados.
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "context" not in st.session_state:
        st.session_state["context"] = ""
    if "show_settings" not in st.session_state:
        st.session_state["show_settings"] = False


def render_sidebar_content(system_status: Dict[str, bool], speed_test_results: Optional[Dict]):
    """
    Renderiza el contenido de la barra lateral con diseño mejorado.
    """
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

            if speed_test_results:
                st.markdown("##### Rendimiento")
                display_speed_test_results(speed_test_results)

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
            st.image(
                "assets/profile.jpg",
                width=60,
            )  # Asegúrate de tener esta imagen de perfil en la carpeta assets
        with col2:
            st.markdown("#### Alexander Oviedo Fadul")
            st.caption("Developer & Legal Tech")

        # Enlaces sociales
        social_links = {
            "🌐 Website": "https://alexanderoviedofadul.dev",
            "💼 LinkedIn": "https://linkedin.com/in/alexander-oviedo-fadul",
            "📱 WhatsApp": "https://wa.me/573015930519",
        }

        for platform, link in social_links.items():
            st.markdown(f"[{platform}]({link})")


def render_response_details(details: Dict):
    """
    Renderiza los detalles de la respuesta en expansores organizados.
    """
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("💭 Proceso de Razonamiento", expanded=False):
            st.markdown(details["initial_evaluation"])
            st.download_button(
                "📥 Exportar Razonamiento",
                details["initial_evaluation"],
                "razonamiento.md",
                mime="text/markdown",
            )

    with col2:
        with st.expander("⚖️ Evaluación Ética", expanded=False):
            st.json(details["ethical_evaluation"])
            if details.get("improved_response"):
                st.info("Respuesta mejorada éticamente:")
                st.write(details["improved_response"])


def export_conversation_to_md(messages, details):
    """
    Exporta la conversación actual a un archivo Markdown, incluyendo detalles.
    """
    md_content = "# Conversación con MALLO\n\n"

    for message in messages:
        if message["role"] == "user":
            md_content += f"## Usuario\n\n{message['content']}\n\n"
        elif message["role"] == "assistant":
            md_content += f"## Asistente\n\n{message['content']}\n\n"

            # Añadir detalles al archivo Markdown
            if details:
                md_content += "### Detalles de la Respuesta\n\n"
                md_content += f"#### Proceso de Razonamiento\n\n{details['initial_evaluation']}\n\n"
                md_content += f"#### Evaluación Ética\n\n"
                md_content += f"{json.dumps(details['ethical_evaluation'], indent=2)}\n\n"

                if details.get("improved_response"):
                    md_content += f"#### Respuesta Mejorada Éticamente\n\n{details['improved_response']}\n\n"

                md_content += f"#### Contexto de la Conversación\n\n{st.session_state.get('context', 'No disponible')}\n\n"

    # Usar el timestamp actual para el nombre del archivo
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"conversacion_{timestamp}.md"

    # Crear un directorio temporal para guardar el archivo
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    return filepath, filename


def main():
    try:
        # Inicialización
        initialize_session_state()
        config = load_config()
        system_status = initialize_system(config)
        agent_manager = AgentManager(config)
        speed_test_results = load_speed_test_results()

        # Renderizar barra lateral
        render_sidebar_content(system_status, speed_test_results)

        # Interfaz principal minimalista
        st.title("MALLO: MultiAgent LLM Orchestrator")

        # Chat interface
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Incluir botón de exportación en mensajes del asistente
                if message["role"] == "assistant":
                    with st.expander("💡 Opciones", expanded=False):
                        # Botón para exportar la conversación actual
                        if st.download_button(
                            label="📥 Exportar Conversación",
                            data=export_conversation_to_md(
                                st.session_state["messages"],
                                st.session_state.get("last_details", {}),
                            )[0],
                            file_name=export_conversation_to_md(
                                st.session_state["messages"],
                                st.session_state.get("last_details", {}),
                            )[1],
                            mime="text/markdown",
                        ):
                            pass

        user_input = st.chat_input("¿En qué puedo ayudarte hoy?")

        if user_input:
            response, details = process_user_input(user_input, config, agent_manager)
            st.session_state["last_details"] = details

            if response:
                st.session_state["messages"].append(
                    {"role": "user", "content": user_input}
                )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )

                with st.chat_message("assistant"):
                    st.markdown(response)
                    render_response_details(details)

    except Exception as e:
        logging.error(f"Error en la aplicación principal: {str(e)}")
        logging.error(traceback.format_exc())
        st.error(
            "Ha ocurrido un error. Por favor, recarga la página o contacta al soporte."
        )

if __name__ == "__main__":
    main()