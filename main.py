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
import traceback
from typing import Tuple, Dict, Any

from load_secrets import load_secrets, get_secret, secrets

# Carga todos los secretos al inicio de la aplicación
load_secrets()


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
            details_placeholder = st.empty()
            start_time = time.time()
            
            initial_evaluation = evaluate_response(agent_manager, config, 'initial', enriched_query)
            
            complexity, needs_web_search, needs_moa, prompt_type = evaluate_query_complexity(initial_evaluation, "")
            prompt_type = agent_manager.validate_prompt_type(user_input, prompt_type)
            
            if needs_web_search:
                web_context = perform_web_search(user_input)
                enriched_query = f"{enriched_query}\nContexto web: {web_context}"
            else:
                web_context = ""
            
            specialized_agent = agent_manager.select_specialized_agent(enriched_query)
            general_agents = agent_manager.get_prioritized_agents(enriched_query, complexity, prompt_type)

            prioritized_agents = []
            if specialized_agent:
                prioritized_agents.append((specialized_agent['type'], specialized_agent['id'], specialized_agent['name']))

            # Añadir agentes generales hasta tener un máximo de 3 agentes en total
            for agent in general_agents:
                if len(prioritized_agents) >= 3:
                    break
                if agent not in prioritized_agents:  # Evitar duplicados
                    prioritized_agents.append(agent)

            # Asegurarse de que no hay más de 3 agentes
            prioritized_agents = prioritized_agents[:3]
            
            agent_results = []
            for agent_type, agent_id, agent_name in prioritized_agents:
                details_placeholder.write(f"Procesando con {agent_name}...")
                try:
                    enriched_query_with_prompt = agent_manager.apply_specialized_prompt(enriched_query, prompt_type)
                    result = agent_manager.process_query(enriched_query_with_prompt, agent_type, agent_id, prompt_type)
                    agent_results.append({
                        "agent": agent_type,
                        "model": agent_id,
                        "name": agent_name,
                        "status": "success",
                        "response": result
                    })
                except Exception as e:
                    logging.error(f"Error processing with {agent_name}: {str(e)}")
                    agent_results.append({
                        "agent": agent_type,
                        "model": agent_id,
                        "name": agent_name,
                        "status": "error",
                        "response": str(e)
                    })

            successful_responses = [r for r in agent_results if r["status"] == "success"]
            
            if not successful_responses:
                raise ValueError("No se pudo obtener una respuesta válida de ningún agente")

            if needs_moa and len(successful_responses) > 1:
                meta_analysis_result = agent_manager.meta_analysis(
                    user_input, 
                    [r["response"] for r in successful_responses], 
                    initial_evaluation, 
                    ""
                )
                final_response = agent_manager.process_query(
                    f"Basándote en este meta-análisis, proporciona una respuesta conversacional y directa a la pregunta '{user_input}'. La respuesta debe ser natural, como si estuvieras charlando con un amigo, sin usar frases como 'Basándome en el análisis' o 'La respuesta es'. Simplemente responde de manera clara y concisa: {meta_analysis_result}",
                    agent_manager.meta_analysis_api,
                    agent_manager.meta_analysis_model
                )
            else:
                final_response = successful_responses[0]["response"]

            final_evaluation = evaluate_response(agent_manager, config, 'final', user_input, final_response)

            processing_time = time.time() - start_time

            details = {
                "selected_agents": [{"agent": r["agent"], "model": r["model"], "name": r["name"]} for r in agent_results],
                "processing_time": f"{processing_time:.2f} segundos",
                "complexity": complexity,
                "needs_web_search": needs_web_search,
                "needs_moa": needs_moa,
                "web_context": web_context,
                "prompt_type": prompt_type,
                "initial_evaluation": initial_evaluation,
                "agent_processing": agent_results,
                "final_evaluation": final_evaluation,
                "performance_metrics": {
                    "total_agents_called": len(agent_results),
                    "successful_responses": len(successful_responses),
                    "failed_responses": len(agent_results) - len(successful_responses),
                    "average_response_time": f"{processing_time:.2f} segundos"                    
                },
                "meta_analysis": meta_analysis_result if needs_moa and len(successful_responses) > 1 else None,
                "final_response": final_response
            }

            new_context = summarize_conversation(conversation_context, user_input, final_response, agent_manager, config)
            st.session_state['context'] = new_context

            cache_response(enriched_query, (final_response, details))

            return final_response, details

    except Exception as e:
        logging.error(f"Se ha producido un error inesperado: {str(e)}")
        logging.error(traceback.format_exc())
        return "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, intenta de nuevo.", {"error": str(e)}

# Evaluar la respuesta del modelo de lenguaje y proporcionar retroalimentación
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
# Si el contexto supera el límite, se utiliza cohere o OpenAI para resumirlo
def summarize_conversation(previous_context, user_input, response, agent_manager, config, max_length=500):
    new_content = f"Usuario: {user_input}\nAsistente: {response}"
    updated_context = f"{previous_context}\n\n{new_content}".strip()
    
    if len(updated_context) > max_length:
        summary_prompt = f"Resume la siguiente conversación manteniendo los puntos clave:\n\n{updated_context}"
        
        try:
            # Intenta usar openrouter primero
            # summary = agent_manager.process_with_openrouter(summary_prompt, config['openrouter']['fast_models'])
            # Intenta usar cohere primero
            summary = agent_manager.process_query(summary_prompt, 'cohere', config['cohere']['default_model'])
        except Exception as e:
            log_error(f"Error al usar cohere para resumen: {str(e)}. Usando OpenAI como respaldo.")
            try:
                # Si cohere falla, usa OpenAI como respaldo
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
        MALLO es un sistema avanzado de orquestación de múltiples agentes basados en Modelos de Lenguaje de Gran Escala (LLMs). 
        Diseñado para proporcionar respuestas precisas, contextuales y adaptativas, MALLO integra:

        - **Múltiples Modelos de IA**: Desde modelos locales hasta APIs de última generación como OpenAI, Anthropic, Groq, DeepInfra, DeepSeek, Mistral, Cohere, OpenRouter y más.
        - **Asistentes Especializados**: Expertos en diversos campos como derecho, tecnología, ciencias, matemáticas, lingüística, análisis documental y otros dominios específicos.
        - **Análisis Contextual Avanzado**: Evalúa la complejidad y tipo de cada consulta para seleccionar la mejor estrategia de respuesta y los agentes más adecuados.
        - **Prompts Especializados**: Aplica prompts adaptados al tipo de consulta (matemática, legal, científica, coding, etc.) para mejorar la precisión de las respuestas.
        - **Selección Inteligente de Agentes**: Prioriza la selección de agentes basándose en sus especialidades, capacidades y el tipo de prompt requerido.
        - **Búsqueda Web Inteligente**: Enriquece las respuestas con información actualizada cuando es necesario.
        - **Meta-análisis Avanzado**: Sintetiza y evalúa respuestas de múltiples fuentes para una mayor precisión y coherencia.
        - **Adaptabilidad Dinámica**: Ajusta su enfoque basándose en el rendimiento, la complejidad de la consulta y la retroalimentación.
        - **Prevención de Redundancia**: Implementa estrategias para evitar la selección de agentes duplicados, asegurando diversidad en las respuestas.
        - **Evaluación Continua**: Incorpora sistemas de evaluación inicial y final para garantizar la calidad y relevancia de las respuestas.
        - **Sistema de Fallback Robusto**: Garantiza respuestas incluso cuando los agentes primarios no están disponibles.
        - **Capacidades Multimodales**: Maneja consultas que involucran texto, audio, imágenes y análisis de herramientas.
        - **Optimización de Recursos**: Utiliza un sistema de caché eficiente para respuestas frecuentes, reduciendo el uso de recursos.

        MALLO no solo responde preguntas, sino que orquesta una sinergia de conocimientos y capacidades para ofrecer
        la mejor solución posible a cada consulta. Se adapta continuamente para mejorar su precisión, relevancia y eficiencia,
        optimizando el uso de recursos según la complejidad de cada tarea.
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