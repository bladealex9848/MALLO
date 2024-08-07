import streamlit as st
import yaml
from agents import AgentManager, process_query

# Cargar configuración
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="MALLO: MultiAgent LLM Orchestrator",
    page_icon="🤖",
    layout="wide",
)

# Inicialización
agent_manager = AgentManager(config)

# Título y descripción
st.title("MALLO: MultiAgent LLM Orchestrator")
st.write("""
MALLO es un sistema de orquestación de múltiples agentes de Modelos de Lenguaje de Gran Escala (LLMs).
Puede manejar consultas utilizando una variedad de agentes, desde modelos locales hasta asistentes especializados.
""")

# Entrada del usuario
query = st.text_input("Ingrese su consulta aquí:")

if st.button("Procesar"):
    if query:
        with st.spinner('Procesando su consulta...'):
            response = process_query(agent_manager, query)
            st.write("Respuesta:")
            st.write(response)
    else:
        st.warning("Por favor, ingrese una consulta.")

# Información adicional en la barra lateral
st.sidebar.title("Acerca de MALLO")
st.sidebar.info("""
MALLO utiliza una variedad de agentes y modelos para procesar consultas.
La selección del agente se basa en la naturaleza de la consulta y la disponibilidad de recursos.
""")

# Footer
st.sidebar.markdown('---')
st.sidebar.subheader('Creado por:')
st.sidebar.markdown('Alexander Oviedo Fadul')
st.sidebar.markdown(
    "[GitHub](https://github.com/bladealex9848) | [Website](https://alexander.oviedo.isabellaea.com/) | [Instagram](https://www.instagram.com/alexander.oviedo.fadul) | [Twitter](https://twitter.com/alexanderofadul) | [Facebook](https://www.facebook.com/alexanderof/) | [WhatsApp](https://api.whatsapp.com/send?phone=573015930519&text=Hola%20!Quiero%20conversar%20contigo!%20)"
)