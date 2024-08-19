import spacy
from typing import Dict, Any, Tuple, List
from duckduckgo_search import DDGS
import streamlit as st
from agents import AgentManager, evaluate_query_complexity
import hashlib
from datetime import datetime, timedelta
import requests
import logging

# Configurar logging
logging.basicConfig(filename='mallo.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar el modelo de lenguaje de spaCy
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

def initialize_system(config: Dict[str, Any]) -> Dict[str, bool]:
    """Inicializa y verifica los componentes del sistema."""
    status = {
        "OpenAI API": check_openai_api(),
        "Groq API": check_groq_api(),
        "Local Models": check_local_models(config),
        "Web Search": check_web_search(),
        "Specialized Agents": check_specialized_agents(config)
    }
    logging.info(f"System status: {status}")
    return status

def check_openai_api() -> bool:
    """Verifica la conexión con la API de OpenAI."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        client.models.list()
        return True
    except Exception as e:
        logging.error(f"Error checking OpenAI API: {str(e)}")
        return False

def check_groq_api() -> bool:
    """Verifica la conexión con la API de Groq."""
    try:
        from groq import Groq
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        client.chat.completions.create(model="llama-3.1-70b-versatile", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking Groq API: {str(e)}")
        return False

def check_local_models(config: Dict[str, Any]) -> bool:
    """Verifica la disponibilidad de modelos locales."""
    try:
        response = requests.get(f"{config['ollama']['base_url']}/api/tags")
        return len(response.json()['models']) > 0
    except Exception as e:
        logging.error(f"Error checking local models: {str(e)}")
        return False

def check_web_search() -> bool:
    """Verifica la funcionalidad de búsqueda web."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text("test", max_results=1))
            return len(results) > 0
    except Exception as e:
        logging.error(f"Error checking web search: {str(e)}")
        return False

def check_specialized_agents(config: Dict[str, Any]) -> bool:
    """Verifica la disponibilidad de agentes especializados."""
    specialized_assistants = config.get('specialized_assistants', [])
    return len(specialized_assistants) > 0

def evaluate_query(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evalúa la consulta del usuario."""
    doc = nlp(query)
    
    analysis = {
        "complexity": evaluate_query_complexity(query),
        "domain": identify_domain(doc, config['specialized_assistants']),
        "urgency": estimate_urgency(doc),
        "requires_web_search": needs_web_search(doc)
    }
    
    return analysis

def identify_domain(doc, specialized_assistants: List[Dict[str, Any]]) -> str:
    """Identifica el dominio de la consulta."""
    for assistant in specialized_assistants:
        if any(keyword.lower() in doc.text.lower() for keyword in assistant['keywords']):
            return assistant['name']
    return "general"

def estimate_urgency(doc) -> float:
    """Estima la urgencia de la consulta."""
    urgency_words = ["urgente", "inmediatamente", "pronto", "rápido"]
    return min(sum(word.text.lower() in urgency_words for word in doc) / len(doc), 1.0)

def needs_web_search(doc) -> bool:
    """Determina si la consulta requiere búsqueda web."""
    web_search_indicators = ["actual", "reciente", "último", "noticias"]
    return any(word.text.lower() in web_search_indicators for word in doc)

def process_query(query: str, analysis: Dict[str, Any], config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Procesa la consulta del usuario."""
    agent_manager = AgentManager(config)
    agent_type, agent_id = agent_manager.get_appropriate_agent(query, analysis['complexity'])
    
    if analysis['requires_web_search']:
        web_context = perform_web_search(query)
        enriched_query = f"{query}\nContext: {web_context}"
    else:
        enriched_query = query
        web_context = ""
    
    try:
        response = agent_manager.process_query(enriched_query, agent_type, agent_id)
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        response = "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, inténtalo de nuevo."
    
    details = {
        "query_analysis": analysis,
        "selected_agent": (agent_type, agent_id),
        "web_search_performed": analysis['requires_web_search'],
        "web_context": web_context
    }
    
    return response, details

def perform_web_search(query: str) -> str:
    """Realiza una búsqueda web y retorna los resultados."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if results:
            return "\n".join([result['body'] for result in results])
        else:
            return "No se encontraron resultados en la búsqueda web."
    except Exception as e:
        logging.error(f"Error in web search: {str(e)}")
        return "Error al realizar la búsqueda web."

# Sistema de caché
cache = {}

def cache_response(query: str, response: Tuple[str, Dict[str, Any]]):
    """Almacena la respuesta en caché."""
    key = hashlib.md5(query.encode()).hexdigest()
    cache[key] = {
        'response': response,
        'timestamp': datetime.now()
    }

def get_cached_response(query: str) -> Tuple[str, Dict[str, Any]] or None:
    """Obtiene la respuesta de la caché si está disponible y es reciente."""
    key = hashlib.md5(query.encode()).hexdigest()
    if key in cache:
        cached_item = cache[key]
        if datetime.now() - cached_item['timestamp'] < timedelta(hours=1):
            return cached_item['response']
    return None

def summarize_text(text: str, max_length: int = 200) -> str:
    """Genera un resumen simple del texto."""
    if len(text) <= max_length:
        return text
    
    sentences = text.split('.')
    summary = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            summary.append(sentence)
            current_length += len(sentence)
        else:
            break
    
    return '. '.join(summary) + '.'

def log_error(error_message: str):
    """Registra un error en el archivo de log."""
    logging.error(error_message)
    st.error(error_message)

def log_warning(warning_message: str):
    """Registra una advertencia en el archivo de log."""
    logging.warning(warning_message)
    st.warning(warning_message)

def log_info(info_message: str):
    """Registra información en el archivo de log."""
    logging.info(info_message)
    st.info(info_message)