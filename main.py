import streamlit as st
import yaml
from agents import AgentManager, process_query, evaluate_query_complexity
from utilities import buscar_en_duckduckgo, charla_con_openai, extract_text_from_pdf, extract_text_from_docx, summarize_text
import time
import asyncio
from together import Together

# Cargar configuración
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configuración de la página de Streamlit
st.set_page_config(
    page_title=config['ui']['page_title'],
    page_icon=config['ui']['page_icon'],
    layout=config['ui']['layout'],
    initial_sidebar_state="expanded",
)

# Inicialización
agent_manager = AgentManager(config)
together_client = Together(api_key=st.secrets.get("TOGETHER_API_KEY"))

# Título y descripción
st.title("MALLO: MultiAgent LLM Orchestrator")
st.write("""
MALLO es un sistema de orquestación de múltiples agentes de Modelos de Lenguaje de Gran Escala (LLMs).
Puede manejar consultas utilizando una variedad de agentes, búsqueda web y asistentes especializados.
""")

# Inicialización de variables de estado de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = ""

# Carga de archivos
#uploaded_file = st.sidebar.file_uploader("Cargar archivo de contexto", type=config['utilities']['file_handling']['allowed_extensions'])
#if uploaded_file is not None:
#    file_contents = ""
#    if uploaded_file.type == "application/pdf":
#        file_contents = extract_text_from_pdf(uploaded_file)
#    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#        file_contents = extract_text_from_docx(uploaded_file)
#    else:
#        file_contents = uploaded_file.getvalue().decode("utf-8")
#    st.session_state.context += f"\nContexto del archivo: {file_contents}\n"

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    # Resumen del contexto y la conversación
    if st.session_state.context or st.session_state.messages:
        context_summary = summarize_text(st.session_state.context, max_length=200)
        conversation_summary = summarize_text("\n".join([m["content"] for m in st.session_state.messages[-5:]]), max_length=200)
        full_prompt = f"Nueva consulta: {prompt}\n\nContexto previo: {context_summary}\nConversación previa: {conversation_summary}"
    else:
        full_prompt = prompt

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Procesar la consulta
    start_time = time.time()
    with st.spinner('Procesando tu consulta...'):
        progress_placeholder = st.empty()
        process_steps = []
        
        # Evaluación inicial de la consulta
        complexity = evaluate_query_complexity(full_prompt)
        
        # Búsqueda web (si está habilitada y es necesaria)
        web_context = ""
        if config['utilities']['web_search']['enabled'] and complexity > config['thresholds']['web_search_complexity']:
            progress_placeholder.text("Realizando búsqueda web...")
            web_results = buscar_en_duckduckgo(prompt, config['utilities']['web_search']['max_results'])
            web_context = "\n".join([f"Título: {r['title']}\nResumen: {r['body']}\nURL: {r['href']}" for r in web_results])
            process_steps.append("1. Búsqueda web realizada con DuckDuckGo")

        # Selección de agente o método de procesamiento
        if complexity > config['thresholds']['moa_complexity'] and agent_manager.internet_available():
            process_type = "moa"
        elif any(keyword in full_prompt.lower() for assistant in config['specialized_assistants'] for keyword in assistant['keywords']):
            process_type = "specialized_assistant"
        elif complexity > config['thresholds']['api_complexity'] and agent_manager.internet_available():
            process_type = "api"
        else:
            process_type = "local"

        # Procesamiento de la consulta
        if process_type == "moa":
            process_steps.append("2. Iniciando proceso MOA (Mixture of Agents)")
            final_response = agent_manager.run_moa(full_prompt, web_context)
        elif process_type == "specialized_assistant":
            assistant = next((a for a in config['specialized_assistants'] if any(keyword in full_prompt.lower() for keyword in a['keywords'])), None)
            process_steps.append(f"2. Utilizando asistente especializado: {assistant['name']}")
            final_response = agent_manager.process_with_assistant(assistant['id'], full_prompt)
        elif process_type == "api":
            process_steps.append("2. Utilizando API de OpenAI")
            final_response, _ = agent_manager.process_with_api(full_prompt)
        else:
            process_steps.append("2. Utilizando modelo local Ollama")
            final_response = agent_manager.process_with_local_model(full_prompt)

        # Síntesis final (si es necesario y está configurado)
        if config['openai']['master_model'] and complexity > config['thresholds']['synthesis_complexity']:
            progress_placeholder.text("Sintetizando respuesta final...")
            full_context = f"{st.session_state.context}\n{web_context}\nRespuesta del agente: {final_response}"
            final_response = charla_con_openai(prompt, full_context, st.session_state.messages, config['openai']['master_model'])
            process_steps.append("3. Síntesis final realizada por OpenAI")

        progress_placeholder.empty()

    processing_time = time.time() - start_time

    with st.chat_message("assistant"):
        st.markdown(final_response)

    # Actualizar el contexto y el historial de la conversación
    st.session_state.context += f"\n{prompt}\n{final_response}"
    st.session_state.messages.append({"role": "assistant", "content": final_response})

    # Mostrar detalles adicionales en una pestaña
    with st.expander("Detalles de la consulta"):
        st.write(f"Consulta original: {prompt}")
        st.write(f"Consulta enriquecida: {full_prompt}")
        st.write(f"Complejidad estimada: {complexity:.2f}")
        st.write("Proceso detallado:")
        for step in process_steps:
            st.write(step)
        if web_context:
            st.write("Resultados de la búsqueda web:")
            for result in web_results:
                st.write(f"- Título: {result['title']}")
                st.write(f"  URL: {result['href']}")
        st.write(f"Respuesta final:\n{final_response}")
        st.write(f"Tiempo total de procesamiento: {processing_time:.2f} segundos")

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
    "[GitHub](https://github.com/bladealex9848) | [Website](https://alexander.oviedo.isabellaea.com/) | [Instagram](https://www.instagram.com/alexander.oviedo.fadul) | [Twitter](https://twitter.com/alexanderofadul) | [Facebook](https://www.facebook.com/alexanderof/) | [WhatsApp](https://api.whatsapp.com/send?phone=573015930519&text=Hola%20!Quiero%20conversar%20contigo!%20)"
)