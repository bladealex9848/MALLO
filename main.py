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
    evaluate_response,
    evaluate_ethical_compliance,
    summarize_conversation,
    load_config,
    load_speed_test_results,
    render_sidebar_content,
    render_response_details,
    export_conversation_to_md,
    display_speed_test_results,
    load_speed_test_results,        
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mallo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Carga todos los secretos al inicio de la aplicación
load_secrets()

# CSS personalizado para interfaz moderna - Añadir después de set_streamlit_page_config()
st.markdown("""
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
""", unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa el estado de la sesión con valores predeterminados."""
    default_states = {
        "messages": [],
        "context": "",
        "show_settings": False,
        "last_details": {},
        "error_count": 0,
        "last_successful_response": None
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

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

        # Interfaz principal
        st.title("MALLO: MultiAgent LLM Orchestrator")

        # Interfaz de chat
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    with st.expander("💡 Opciones", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.download_button(
                                "📥 Exportar Conversación Completa",
                                *export_conversation_to_md(
                                    st.session_state["messages"],
                                    st.session_state.get("last_details", {})
                                ),
                                mime="text/markdown"
                            ):
                                st.success("Conversación exportada exitosamente")
                        with col2:
                            if st.download_button(
                                "📥 Exportar Última Respuesta",
                                message["content"],
                                f"respuesta_{time.strftime('%Y%m%d-%H%M%S')}.md",
                                mime="text/markdown"
                            ):
                                st.success("Respuesta exportada exitosamente")
                        render_response_details(st.session_state["last_details"])

        # Input del usuario y manejo de respuestas
        user_input = st.chat_input("¿En qué puedo ayudarte hoy?")

        if user_input:
            try:
                # Mostrar indicador de procesamiento
                with st.spinner("Procesando tu consulta..."):
                    response, details = process_user_input(user_input, config, agent_manager)
                    st.session_state["last_details"] = details
                    st.session_state["error_count"] = 0  # Resetear contador de errores

                    if response:
                        # Actualizar el historial de la conversación
                        st.session_state["messages"].append({
                            "role": "user",
                            "content": user_input,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.session_state["last_successful_response"] = response

                        # Mostrar la respuesta con sus detalles
                        with st.chat_message("assistant"):
                            st.markdown(response)
                            
                            # Expandir para mostrar detalles y opciones
                            with st.expander("🔍 Detalles y Opciones", expanded=False):
                                # Métricas de rendimiento
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Tiempo de Proceso", 
                                            f"{float(details['processing_time'].split()[0]):.2f}s")
                                with col2:
                                    st.metric("Agentes Usados", 
                                            str(details['performance_metrics']['total_agents_called']))
                                with col3:
                                    st.metric("Complejidad", 
                                            f"{details['complexity']:.2f}")

                                # Botones de exportación
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.download_button(
                                        "📥 Exportar Conversación Completa",
                                        *export_conversation_to_md(
                                            st.session_state["messages"],
                                            st.session_state.get("last_details", {})
                                        ),
                                        mime="text/markdown"
                                    ):
                                        st.success("✅ Conversación exportada exitosamente")
                                
                                with col2:
                                    if st.download_button(
                                        "📥 Exportar Última Respuesta",
                                        response,
                                        f"respuesta_{time.strftime('%Y%m%d-%H%M%S')}.md",
                                        mime="text/markdown"
                                    ):
                                        st.success("✅ Respuesta exportada exitosamente")

                                # Mostrar detalles adicionales
                                st.markdown("#### 📊 Detalles del Procesamiento")
                                render_response_details(details)

                                # Información sobre el contexto
                                st.markdown("#### 🔄 Contexto de la Conversación")
                                with st.expander("Ver contexto actual", expanded=False):
                                    st.text(st.session_state.get("context", "No hay contexto disponible"))

            except Exception as e:
                st.session_state["error_count"] += 1
                logger.error(f"Error procesando entrada del usuario: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Mensaje de error adaptativo
                error_message = (
                    "Ha ocurrido un error al procesar tu consulta. "
                    "Por favor, intenta de nuevo o reformula tu pregunta."
                )
                
                # Mensajes adicionales basados en el contador de errores
                if st.session_state["error_count"] > 3:
                    error_message += (
                        "\n\n⚠️ Parece que estamos teniendo problemas técnicos persistentes. "
                        "Te sugerimos:\n"
                        "1. Intentar más tarde\n"
                        "2. Verificar tu conexión a internet\n"
                        "3. Contactar al soporte técnico\n"
                        "\nPuedes continuar la conversación con el último contexto exitoso."
                    )
                    
                    # Intentar recuperar el último estado exitoso
                    if st.session_state.get("last_successful_response"):
                        st.info(
                            "👉 Mientras tanto, puedes revisar la última respuesta exitosa "
                            "o exportar la conversación hasta este punto."
                        )
                
                st.error(error_message)
                
                # Log del error para análisis
                logger.error(f"Error Count: {st.session_state['error_count']}")
                logger.error(f"Last Successful Response Available: {bool(st.session_state.get('last_successful_response'))}")

    except Exception as e:
        logger.error(f"Error crítico en la aplicación: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Mensaje de error amigable pero informativo
        st.error(
            "🚨 Error crítico en la aplicación\n\n"
            "Ha ocurrido un error inesperado. Por favor:\n"
            "1. Recarga la página\n"
            "2. Verifica tu conexión\n"
            "3. Si el problema persiste, contacta al soporte técnico\n\n"
            "Tus datos de conversación están seguros y se intentarán recuperar en la próxima sesión."
        )
        
        # Intentar guardar el estado actual para recuperación
        try:
            with open('error_recovery.json', 'w') as f:
                json.dump({
                    "messages": st.session_state.get("messages", []),
                    "context": st.session_state.get("context", ""),
                    "last_details": st.session_state.get("last_details", {}),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f)
        except Exception as save_error:
            logger.error(f"Error al guardar estado para recuperación: {str(save_error)}")

if __name__ == "__main__":
    main()