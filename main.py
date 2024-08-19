# Importa y ejecuta la configuración de Streamlit antes que cualquier otra cosa
from config_streamlit import set_streamlit_page_config
set_streamlit_page_config()

import streamlit as st
import yaml
import os
from agents import AgentManager
from utilities import (
    initialize_system, evaluate_query, process_query, 
    cache_response, get_cached_response, summarize_text,
    test_agent_speed, select_fastest_model, select_most_capable_model,
    display_speed_test_results, perform_web_search, get_cached_speed_results,
    evaluate_query_complexity
)
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


# Cargar el modelo de selección de agentes
try:
    agent_selector = joblib.load('agent_selector.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
except:
    agent_selector = MultinomialNB()
    vectorizer = TfidfVectorizer()

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

async def process_with_multiple_agents(user_input, agent_manager, num_agents=3):
    tasks = []
    for agent_type, agent_id in agent_manager.get_agent_priority()[:num_agents]:
        tasks.append(asyncio.create_task(
            asyncio.to_thread(agent_manager.process_query, user_input, agent_type, agent_id)
        ))
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    valid_responses = [r for r in responses if not isinstance(r, Exception)]
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
            
            # Usar el modelo de selección de agentes
            agent_features = vectorizer.transform([user_input])
            predicted_agent = agent_selector.predict(agent_features)[0]
            
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
                "selected_agent": predicted_agent,
                'processing_time': f"{processing_time:.2f} segundos",
                'initial_evaluation': initial_evaluation,
                'final_evaluation': final_evaluation,
                'query_analysis': query_analysis,
                'web_search_performed': query_analysis['requires_web_search'],
                'web_context': web_context
            }

            # Actualizar el modelo de selección de agentes
            agent_selector.partial_fit(agent_features, [predicted_agent])
            joblib.dump(agent_selector, 'agent_selector.joblib')
            joblib.dump(vectorizer, 'vectorizer.joblib')

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

        speed_test_results = get_cached_speed_results()
        if not speed_test_results:
            speed_test_results = test_agent_speed(agent_manager)

        st.title("MALLO: MultiAgent LLM Orchestrator")
        st.write("""
        MALLO es un sistema avanzado de orquestación de múltiples agentes de Modelos de Lenguaje de Gran Escala (LLMs).
        Proporciona respuestas precisas y contextuales utilizando una variedad de agentes y fuentes de información.
        """)

        st.sidebar.title("Estado del Sistema")
        for key, value in system_status.items():
            st.sidebar.text(f"{key}: {'✅' if value else '❌'}")
            
        display_speed_test_results(speed_test_results)

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