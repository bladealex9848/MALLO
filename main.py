# Importa y ejecuta la configuración de Streamlit antes que cualquier otra cosa
from config_streamlit import set_streamlit_page_config
set_streamlit_page_config()

# Resto de las importaciones
import streamlit as st
import yaml
import os
from agents import AgentManager
from utilities import (
    initialize_system, evaluate_query, process_query, 
    cache_response, get_cached_response, summarize_text
)
import time

def main():
    try:
        # Cargar configuración
        config = load_config()

        # Inicialización del sistema
        system_status = initialize_system(config)

        # Título y descripción
        st.title("MALLO: MultiAgent LLM Orchestrator")
        st.write("""
        MALLO es un sistema avanzado de orquestación de múltiples agentes de Modelos de Lenguaje de Gran Escala (LLMs).
        Proporciona respuestas precisas y contextuales utilizando una variedad de agentes y fuentes de información.
        """)

        # Sidebar con estado del sistema e información
        st.sidebar.title("Estado del Sistema")
        for key, value in system_status.items():
            st.sidebar.text(f"{key}: {'✅' if value else '❌'}")

        # Inicialización de la sesión
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        if 'context' not in st.session_state:
            st.session_state['context'] = ""

        # Mostrar mensajes anteriores
        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Entrada del usuario
        user_input = st.chat_input("¿En qué puedo ayudarte hoy?")

        if user_input:
            process_user_input(user_input, config)

        # Información adicional en la barra lateral
        st.sidebar.title("Acerca de MALLO")
        st.sidebar.info("""
        MALLO utiliza una variedad de agentes, búsqueda web y modelos para procesar consultas.
        La selección del agente se basa en la naturaleza de la consulta y la disponibilidad de recursos.
        """)

        # Footer
        st.sidebar.markdown('---')
        st.sidebar.subheader('Creado por:')
        st.sidebar.markdown('Alexander Oviedo Fadul')
        st.sidebar.markdown(
            "[GitHub](https://github.com/bladealex9848) | "
            "[Website](https://alexander.oviedo.isabellaea.com/) | "
            "[Instagram](https://www.instagram.com/alexander.oviedo.fadul) | "
            "[Twitter](https://twitter.com/alexanderofadul) | "
            "[Facebook](https://www.facebook.com/alexanderof/) | "
            "[WhatsApp](https://api.whatsapp.com/send?phone=573015930519&text=Hola%20!Quiero%20conversar%20contigo!%20)"
        )

    except Exception as e:
        st.error(f"Se ha producido un error inesperado: {str(e)}")
        st.error("Por favor, contacta al soporte técnico con los detalles del error.")

def load_config():
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("No se pudo encontrar el archivo de configuración 'config.yaml'.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error al leer el archivo de configuración: {str(e)}")
        st.stop()

def process_user_input(user_input, config):
    try:
        # Verificar caché
        cached_response = get_cached_response(user_input)
        if cached_response:
            st.success("Respuesta obtenida de la caché")
            response, details = cached_response
        else:
            with st.spinner("Procesando tu consulta..."):
                start_time = time.time()
                
                # Resumen del contexto y la conversación
                context_summary = summarize_text(st.session_state['context'], max_length=200)
                conversation_summary = summarize_text("\n".join([m["content"] for m in st.session_state['messages'][-5:]]), max_length=200)
                full_prompt = f"Nueva consulta: {user_input}\n\nContexto previo: {context_summary}\nConversación previa: {conversation_summary}"
                
                # Evaluar la consulta
                query_analysis = evaluate_query(full_prompt, config)
                
                # Procesar la consulta
                response, details = process_query(full_prompt, query_analysis, config)
                
                processing_time = time.time() - start_time
                details['processing_time'] = f"{processing_time:.2f} segundos"
                
                # Guardar en caché
                cache_response(user_input, (response, details))

        # Mostrar respuesta
        st.session_state['messages'].append({"role": "user", "content": user_input})
        st.session_state['messages'].append({"role": "assistant", "content": response})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Mostrar detalles del proceso
        with st.expander("Detalles del proceso"):
            st.json(details)
        
        # Actualizar el contexto
        st.session_state['context'] += f"\n{user_input}\n{response}"

    except Exception as e:
        st.error(f"Error al procesar la consulta: {str(e)}")
        st.error("Por favor, intenta reformular tu pregunta o contacta al soporte si el problema persiste.")

if __name__ == "__main__":
    main()