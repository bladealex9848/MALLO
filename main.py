# Importa y ejecuta la configuración de Streamlit antes que cualquier otra cosa
from config_streamlit import set_streamlit_page_config
set_streamlit_page_config()

import streamlit as st
import yaml
import os
import json
from agents import AgentManager
from utilities import (
    initialize_system, evaluate_query, process_query, 
    cache_response, get_cached_response, summarize_text,
    select_fastest_model, select_most_capable_model,
    perform_web_search, evaluate_query_complexity
)
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import polars as pl

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

def load_speed_test_results():
    try:
        with open('model_speeds.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def display_speed_test_results(results):
    st.sidebar.title("Resultados del Test de Velocidad")
    
    # Crear un DataFrame de Polars con los resultados
    data = []
    for api, models in results.items():
        for model in models:
            data.append({"API": api, "Modelo": model['model'], "Velocidad": f"{model['speed']:.4f}"})
    
    df = pl.DataFrame(data)
    
    # Obtener la lista única de APIs
    apis = df['API'].unique().to_list()
    
    # Crear un menú desplegable para seleccionar la API
    selected_api = st.sidebar.selectbox("Seleccionar API", apis)
    
    # Filtrar el DataFrame por la API seleccionada
    filtered_df = df.filter(pl.col('API') == selected_api)
    
    # Mostrar la tabla filtrada
    st.sidebar.table(filtered_df)

async def process_with_multiple_agents(user_input, agent_manager, num_agents=3):
    tasks = []
    for agent_type, agent_id in agent_manager.get_agent_priority()[:num_agents]:
        tasks.append(asyncio.create_task(
            asyncio.to_thread(agent_manager.process_query, user_input, agent_type, agent_id)
        ))
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    valid_responses = [r for r in responses if not isinstance(r, Exception) and r is not None]
    return max(valid_responses, key=len) if valid_responses else None

def process_user_input(user_input, config, agent_manager):
    try:
        cached_response = get_cached_response(user_input)
        if cached_response:
            st.success("Respuesta obtenida de la caché")
            return cached_response

        with st.spinner("Procesando tu consulta..."):
            start_time = time.time()
            
            complexity = evaluate_query_complexity(user_input)
            
            if complexity < 0.3:  # Umbral reducido para consultas simples
                response = agent_manager.process_query(user_input, 'deepinfra', 'meta-llama/Meta-Llama-3-8B-Instruct')
                details = {
                    "selected_agent": ('deepinfra', 'meta-llama/Meta-Llama-3-8B-Instruct'),
                    "complexity": complexity,
                    "processing_time": f"{time.time() - start_time:.2f} segundos"
                }
                return response, details
                        
            initial_evaluation = agent_manager.process_query(
                f"Evalúa esta consulta y proporciona un plan de acción: {user_input}",
                config['evaluation_models']['initial']['api'],
                config['evaluation_models']['initial']['model']
            )
            
            query_analysis = evaluate_query(user_input, config, initial_evaluation)
            
            if query_analysis['requires_web_search']:
                web_context = perform_web_search(user_input)
                enriched_query = f"{user_input}\nContexto web: {web_context}"
            else:
                enriched_query = user_input
                web_context = ""
            
            # Procesar con múltiples agentes en paralelo
            response = asyncio.run(process_with_multiple_agents(enriched_query, agent_manager))
            
            if not response:
                raise Exception("Todos los agentes fallaron al procesar la consulta.")

            final_evaluation = agent_manager.process_query(
                f"Evalúa si esta respuesta es apropiada y precisa para la consulta original. Si no lo es, proporciona una respuesta mejorada:\n\nConsulta: {user_input}\n\nRespuesta: {response}",
                config['evaluation_models']['final']['api'],
                config['evaluation_models']['final']['model']
            )

            if "no es apropiada" in final_evaluation.lower() or "no es precisa" in final_evaluation.lower():
                response = final_evaluation

            processing_time = time.time() - start_time
            details = {
                "selected_agent": 'multiple',
                'processing_time': f"{processing_time:.2f} segundos",
                'initial_evaluation': initial_evaluation,
                'final_evaluation': final_evaluation,
                'query_analysis': query_analysis,
                'web_search_performed': query_analysis['requires_web_search'],
                'web_context': web_context
            }

            cache_response(user_input, (response, details))

            return response, details

    except Exception as e:
        st.error(f"Se ha producido un error inesperado: {str(e)}")
        st.error("Por favor, intenta reformular tu pregunta o contacta al soporte si el problema persiste.")
        return None, None

def main():
    try:
        config = load_config()
        system_status = initialize_system(config)
        agent_manager = AgentManager(config)

        st.title("MALLO: MultiAgent LLM Orchestrator")
        st.write("""
        MALLO es un sistema avanzado de orquestación de múltiples agentes de Modelos de Lenguaje de Gran Escala (LLMs).
        Proporciona respuestas precisas y contextuales utilizando una variedad de agentes y fuentes de información.
        """)

        st.sidebar.title("Estado del Sistema")
        for key, value in system_status.items():
            st.sidebar.text(f"{key}: {'✅' if value else '❌'}")
            
        # Cargar y mostrar resultados del test de velocidad si están disponibles
        speed_test_results = load_speed_test_results()
        if speed_test_results:
            display_speed_test_results(speed_test_results)
        else:
            st.sidebar.warning("No se encontraron resultados de pruebas de velocidad.")

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        if 'context' not in st.session_state:
            st.session_state['context'] = ""

        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("¿En qué puedo ayudarte hoy?")

        if user_input:
            response, details = process_user_input(user_input, config, agent_manager)
            if response:
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": response})

                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    st.markdown(response)

                with st.expander("Detalles del proceso"):
                    st.json(details)

                st.session_state['context'] += f"\n{user_input}\n{response}"

        st.sidebar.title("Acerca de MALLO")
        st.sidebar.info("""
        MALLO utiliza una variedad de agentes, búsqueda web y modelos para procesar consultas.
        La selección del agente se basa en la naturaleza de la consulta y la disponibilidad de recursos.
        """)

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

if __name__ == "__main__":
    main()