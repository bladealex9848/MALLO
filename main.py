# Importa y ejecuta la configuración de Streamlit antes que cualquier otra cosa
from config_streamlit import set_streamlit_page_config
set_streamlit_page_config()

import streamlit as st
import yaml
import os
import json
from agents import AgentManager, evaluate_query_complexity
from utilities import (
    initialize_system, cache_response, get_cached_response, summarize_text,
    perform_web_search, log_error, log_warning, log_info
)
import time
import asyncio
import polars as pl
import random

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
    data = [{"API": api, "Modelo": model['model'], "Velocidad": f"{model['speed']:.4f}"}
            for api, models in results.items() for model in models]
    df = pl.DataFrame(data).sort("Velocidad")
    with st.sidebar.expander("Mostrar Resultados"):
        st.dataframe(df, use_container_width=True, hide_index=True)

async def process_with_agent(agent_manager, query, agent_type, agent_id):
    try:
        response = await asyncio.to_thread(agent_manager.process_query, query, agent_type, agent_id)
        return {
            "agent": agent_type,
            "model": agent_id,
            "status": "success",
            "response": response
        }
    except Exception as e:
        log_error(f"Error processing query with {agent_type}:{agent_id}: {str(e)}")
        return {
            "agent": agent_type,
            "model": agent_id,
            "status": "error",
            "response": str(e)
        }

async def process_with_multiple_agents(query, agent_manager, num_agents=3):
    agents = agent_manager.get_agent_priority()
    random.shuffle(agents)
    tasks = [process_with_agent(agent_manager, query, agent_type, agent_id) 
             for agent_type, agent_id in agents[:num_agents]]
    results = await asyncio.gather(*tasks)
    valid_responses = [r for r in results if r["status"] == "success" and r["response"]]
    best_response = max(valid_responses, key=lambda x: len(x["response"])) if valid_responses else None
    return best_response, results

def process_user_input(user_input, config, agent_manager):
    try:
        cached_response = get_cached_response(user_input)
        if cached_response:
            st.success("Respuesta obtenida de la caché")
            return cached_response

        with st.spinner("Procesando tu consulta..."):
            start_time = time.time()
            
            complexity = evaluate_query_complexity(user_input)
            
            needs_web_search = "actualidad" in user_input.lower() or "reciente" in user_input.lower()
            needs_moa = complexity > 0.7
            
            if needs_web_search:
                web_context = perform_web_search(user_input)
                enriched_query = f"{user_input}\nContexto web: {web_context}"
            else:
                enriched_query = user_input
                web_context = ""
            
            if needs_moa:
                response, agent_results = asyncio.run(process_with_multiple_agents(enriched_query, agent_manager))
            else:
                agent_type, agent_id = agent_manager.get_appropriate_agent(enriched_query, complexity)
                response = agent_manager.process_query(enriched_query, agent_type, agent_id)
                agent_results = [{
                    "agent": agent_type,
                    "model": agent_id,
                    "status": "success",
                    "response": response
                }]

            if not response:
                response = agent_manager.process_query(enriched_query, 'assistant', 'asst_RfRNo5Ij76ieg7mV11CqYV9v')
                agent_results.append({
                    "agent": "assistant",
                    "model": "asst_RfRNo5Ij76ieg7mV11CqYV9v",
                    "status": "success",
                    "response": response
                })

            processing_time = time.time() - start_time
            
            details = {
                "selected_agent": "multiple" if needs_moa else agent_type,
                "processing_time": f"{processing_time:.2f} segundos",
                "complexity": complexity,
                "needs_web_search": needs_web_search,
                "needs_moa": needs_moa,
                "web_context": web_context,
                "agent_processing": agent_results,
                "performance_metrics": {
                    "total_agents_called": len(agent_results),
                    "successful_responses": sum(1 for r in agent_results if r["status"] == "success"),
                    "failed_responses": sum(1 for r in agent_results if r["status"] == "error"),
                    "average_response_time": f"{processing_time / len(agent_results):.2f} segundos"
                }
            }

            cache_response(user_input, (response, details))

            return response, details

    except Exception as e:
        log_error(f"Se ha producido un error inesperado: {str(e)}")
        return "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, intenta de nuevo.", None

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
        log_error(f"Se ha producido un error inesperado: {str(e)}")
        st.error("Se ha producido un error inesperado. Por favor, recarga la página o contacta al soporte técnico.")

if __name__ == "__main__":
    main()