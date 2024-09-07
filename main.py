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
    initialize_system, cache_response, get_cached_response, summarize_text,
    perform_web_search, log_error, log_warning, log_info, evaluate_query_complexity
)
import time
import asyncio
import polars as pl
import random
import logging
from typing import Tuple, Dict, Any

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
        conversation_context = st.session_state.get('context', '')
        enriched_query = f"{conversation_context}\n\nNueva consulta: {user_input}"

        cached_response = get_cached_response(enriched_query)
        if cached_response:
            st.success("Respuesta obtenida de la caché")
            return cached_response

        with st.spinner("Procesando tu consulta..."):
            start_time = time.time()
            
            initial_evaluation = evaluate_response(agent_manager, config, 'initial', enriched_query)
            
            complexity, needs_web_search, needs_moa = evaluate_query_complexity(initial_evaluation, "")
            
            if needs_web_search:
                web_context = perform_web_search(user_input)
                enriched_query = f"{enriched_query}\nContexto web: {web_context}"
            else:
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

            # Manejar el caso en que response sea un diccionario
            if isinstance(response, dict) and 'error' in response:
                fallback_agent, fallback_model = agent_manager.get_fallback_agent()
                response = agent_manager.process_query(enriched_query, fallback_agent, fallback_model)
                agent_results.append({
                    "agent": fallback_agent,
                    "model": fallback_model,
                    "status": "fallback",
                    "response": response
                })

            best_response = max(agent_results, key=lambda x: len(x['response']))['response']

            final_evaluation = evaluate_response(agent_manager, config, 'final', user_input, best_response)

            if "no es apropiada" in final_evaluation.lower() or "no es precisa" in final_evaluation.lower():
                response = final_evaluation
            else:
                response = best_response

            processing_time = time.time() - start_time
            
            # Realizar meta-análisis
            meta_analysis_result = agent_manager.meta_analysis(user_input, best_response, initial_evaluation, final_evaluation)

            details = {
                "selected_agent": "multiple" if needs_moa else agent_type,
                "processing_time": f"{processing_time:.2f} segundos",
                "complexity": complexity,
                "needs_web_search": needs_web_search,
                "needs_moa": needs_moa,
                "web_context": web_context,
                "initial_evaluation": initial_evaluation,
                "agent_processing": agent_results,
                "final_evaluation": final_evaluation,
                "performance_metrics": {
                    "total_agents_called": len(agent_results),
                    "successful_responses": sum(1 for r in agent_results if r["status"] == "success"),
                    "failed_responses": sum(1 for r in agent_results if r["status"] == "error"),
                    "average_response_time": f"{processing_time / len(agent_results):.2f} segundos"                    
                },
                "meta_analysis": meta_analysis_result
            }

            # Usar el resultado del meta-análisis como respuesta final
            final_response = meta_analysis_result

            new_context = summarize_conversation(conversation_context, user_input, response, agent_manager, config)
            st.session_state['context'] = new_context

            cache_response(enriched_query, (final_response, details))

            return final_response, details

    except Exception as e:
        logging.error(f"Se ha producido un error inesperado: {str(e)}")
        return "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, intenta de nuevo.", None

def evaluate_response(agent_manager, config, evaluation_type, query, response=None):
    eval_config = config['evaluation_models'][evaluation_type]
    
    if evaluation_type == 'initial':
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
                evaluation = agent_manager.process_query(evaluation_prompt, eval_config['api'], eval_config['model'])
            elif attempt == 1:
                evaluation = agent_manager.process_query(evaluation_prompt, eval_config['backup_api'], eval_config['backup_model'])
            else:
                evaluation = agent_manager.process_query(evaluation_prompt, eval_config['backup_api2'], eval_config['backup_model2'])
            
            if "Error al procesar" not in evaluation:
                return evaluation
        except Exception as e:
            log_error(f"Error en evaluación {evaluation_type} con {'modelo principal' if attempt == 0 else 'modelo de respaldo'}: {str(e)}")
    
    return "No se pudo realizar la evaluación debido a múltiples errores en los modelos de evaluación."

# Resumen de la conversación con un límite de longitud máximo (500 caracteres)
# Si el contexto supera el límite, se utiliza OpenRouter o OpenAI para resumirlo
def summarize_conversation(previous_context, user_input, response, agent_manager, config, max_length=500):
    new_content = f"Usuario: {user_input}\nAsistente: {response}"
    updated_context = f"{previous_context}\n\n{new_content}".strip()
    
    if len(updated_context) > max_length:
        summary_prompt = f"Resume la siguiente conversación manteniendo los puntos clave:\n\n{updated_context}"
        
        try:
            # Intenta usar OpenRouter primero
            summary = agent_manager.process_with_openrouter(summary_prompt, config['openrouter']['default_model'])
        except Exception as e:
            log_error(f"Error al usar OpenRouter para resumen: {str(e)}. Usando OpenAI como respaldo.")
            try:
                # Si OpenRouter falla, usa OpenAI como respaldo
                summary = agent_manager.process_query(summary_prompt, 'api', config['openai']['default_model'])
            except Exception as e:
                log_error(f"Error al usar OpenAI para resumen: {str(e)}. Devolviendo contexto sin resumir.")
                return updated_context
        
        return summary
    
    return updated_context
    
def main():
    try:
        config = load_config()
        system_status = initialize_system(config)
        agent_manager = AgentManager(config)

        st.title("MALLO: MultiAgent LLM Orchestrator")

        st.write("""
        [![ver código fuente](https://img.shields.io/badge/Repositorio%20GitHub-gris?logo=github)](https://github.com/bladealex9848/MALLO)
        ![Visitantes](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fmallollm.streamlit.app&label=Visitantes&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
        """)

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

                # Mostrar el contexto actual (opcional, para depuración)
                with st.expander("Contexto de la conversación"):
                    st.text(st.session_state['context'])

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
            "[Website](https://alexanderoviedofadul.dev/) | "
            "[LinkedIn](https://www.linkedin.com/in/alexander-oviedo-fadul/) | "
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