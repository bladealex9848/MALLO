import spacy
from typing import Dict, Any, Tuple, List
from duckduckgo_search import DDGS
import streamlit as st
from agents import AgentManager, evaluate_query_complexity
import hashlib
from datetime import datetime, timedelta
import requests
import logging
import time
import json
import polars as pl

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
        "DeepInfra API": check_deepinfra_api(),
        "Anthropic API": check_anthropic_api(),
        "DeepSeek API": check_deepseek_api(),
        "Mistral API": check_mistral_api(),
        "Cohere API": check_cohere_api()
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
        client.chat.completions.create(model="mixtral-8x7b-32768", messages=[{"role": "user", "content": "Test"}])
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

def check_deepinfra_api() -> bool:
    """Verifica la conexión con la API de DeepInfra."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")
        client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking DeepInfra API: {str(e)}")
        return False

def check_anthropic_api() -> bool:
    """Verifica la conexión con la API de Anthropic con reintentos."""
    max_retries = 3
    retry_delay = 5  # segundos

    for attempt in range(max_retries):
        try:
            from anthropic import Anthropic, AuthenticationError, APIConnectionError, APIStatusError
            client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except (APIConnectionError, APIStatusError) as e:
            if attempt < max_retries - 1:
                logging.warning(f"Anthropic API check failed (attempt {attempt + 1}): {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Anthropic API check failed after {max_retries} attempts: {str(e)}")
                return False
        except AuthenticationError as e:
            logging.error(f"Anthropic API authentication error: {str(e)}")
            print("Verifica que tu clave API de Anthropic sea correcta y esté actualizada.")
            return False
        except Exception as e:
            logging.error(f"Unexpected error checking Anthropic API: {str(e)}")
            print(f"Error inesperado al verificar la API de Anthropic: {str(e)}")
            return False

    return False

def check_deepseek_api() -> bool:
    """Verifica la conexión con la API de DeepSeek."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
        client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking DeepSeek API: {str(e)}")
        return False

def check_mistral_api() -> bool:
    """Verifica la conexión con la API de Mistral."""
    try:
        from mistralai import Mistral
        client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
        client.chat.complete(model="open-mistral-nemo", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking Mistral API: {str(e)}")
        return False

def check_cohere_api() -> bool:
    """Verifica la conexión con la API de Cohere."""
    try:
        import cohere
        client = cohere.Client(api_key=st.secrets["COHERE_API_KEY"])
        client.chat(model="command-r-plus", message="Test")
        return True
    except Exception as e:
        logging.error(f"Error checking Cohere API: {str(e)}")
        return False

def evaluate_query(query: str, config: Dict[str, Any], initial_evaluation: str) -> Dict[str, Any]:
    """Evalúa la consulta del usuario."""
    doc = nlp(query)
    
    domains = {
        'philosophy': ['sentido de la vida', 'existencia', 'filosofía', 'ética', 'moral'],
        'science': ['ciencia', 'física', 'química', 'biología', 'tecnología'],
        'history': ['historia', 'eventos históricos', 'personajes históricos'],
        'politics': ['política', 'gobierno', 'leyes', 'elecciones'],
        'arts': ['arte', 'música', 'literatura', 'cine'],
        'general': []
    }
    
    query_lower = query.lower()
    domain = 'general'
    for key, keywords in domains.items():
        if any(keyword in query_lower for keyword in keywords):
            domain = key
            break
    
    analysis = {
        "complexity": evaluate_query_complexity(query),
        "domain": domain,
        "urgency": estimate_urgency(doc),
        "requires_web_search": needs_web_search(doc),
        "initial_evaluation": initial_evaluation
    }
    
    return analysis

def identify_domain(doc, specialized_assistants: List[Dict[str, Any]]) -> str:
    """Identifica el dominio de la consulta."""
    for assistant in specialized_assistants:
        if any(keyword.lower() in doc.text.lower() for keyword in assistant.get('keywords', [])):
            return assistant.get('name', 'general')
    return "general"

def estimate_urgency(doc) -> float:
    """Estima la urgencia de la consulta."""
    urgency_words = ["urgente", "inmediatamente", "pronto", "rápido"]
    return min(sum(word.text.lower() in urgency_words for word in doc) / len(doc), 1.0)

def needs_web_search(doc) -> bool:
    """Determina si la consulta requiere búsqueda web."""
    web_search_indicators = ["actual", "reciente", "último", "noticias"]
    return any(word.text.lower() in web_search_indicators for word in doc)

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

SPEED_TEST_CACHE_FILE = 'speed_test_cache.json'
SPEED_TEST_CACHE_DURATION = timedelta(hours=24)

def cache_speed_results(results):
    with open(SPEED_TEST_CACHE_FILE, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f)

def get_cached_speed_results():
    try:
        with open(SPEED_TEST_CACHE_FILE, 'r') as f:
            data = json.load(f)
        timestamp = datetime.fromisoformat(data['timestamp'])
        if datetime.now() - timestamp < SPEED_TEST_CACHE_DURATION:
            return data['results']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None

def test_agent_speed(agent_manager):
    results = {}
    for agent_type in ['api', 'groq', 'together', 'deepinfra', 'anthropic', 'deepseek', 'mistral', 'cohere']:
        for agent_id in agent_manager.get_available_agents(agent_type):
            try:
                start_time = time.time()
                agent_manager.process_query("Test query", agent_type, agent_id)
                end_time = time.time()
                results[f"{agent_type}:{agent_id}"] = end_time - start_time
            except Exception as e:
                logging.error(f"Error testing agent {agent_type}:{agent_id}: {str(e)}")
                results[f"{agent_type}:{agent_id}"] = float('inf')
    
    cache_speed_results(results)
    return results

def select_fastest_model(agent_manager: AgentManager) -> str:
    """Selecciona el modelo más rápido basado en los resultados del test de velocidad."""
    speed_test_results = get_cached_speed_results()
    if not speed_test_results:
        speed_test_results = test_agent_speed(agent_manager)
    fastest_model = min(speed_test_results, key=speed_test_results.get)
    return fastest_model.split(':')[1]  # Retorna solo el ID del modelo

def select_most_capable_model(agent_manager: AgentManager) -> str:
    """Selecciona el modelo más capaz (asumiendo que es el más lento de los modelos API)."""
    speed_test_results = get_cached_speed_results()
    if not speed_test_results:
        speed_test_results = test_agent_speed(agent_manager)
    api_models = {k: v for k, v in speed_test_results.items() if k.startswith('api:')}
    if api_models:
        most_capable_model = max(api_models, key=api_models.get)
        return most_capable_model.split(':')[1]  # Retorna solo el ID del modelo
    else:
        return agent_manager.config['openai']['default_model']

def display_speed_test_results(speed_test_results: Dict[str, float]):
    """Muestra los resultados del test de velocidad en un elemento desplegable."""
    st.sidebar.title("Resultados del Test de Velocidad")
    with st.sidebar.expander("Ver resultados"):
        # Ordenar los resultados de menor a mayor tiempo (más rápido a más lento)
        sorted_results = sorted(speed_test_results.items(), key=lambda x: x[1])
        
        # Crear un DataFrame de Polars con los resultados
        df_results = pl.DataFrame({
            "Agente": [f"{agent.split(':')[0]}: {agent.split(':')[1]}" for agent, _ in sorted_results],
            "Tiempo de Respuesta": [f"{speed:.2f} segundos" for _, speed in sorted_results]
        })
        
        # Mostrar la tabla usando Streamlit
        st.dataframe(df_results, use_container_width=True)

def process_query(self, query: str, agent_type: str, agent_id: str) -> str:
        start_time = time.time()
        try:
            if agent_type == 'assistant':
                response = self.process_with_assistant(agent_id, query)
            elif agent_type == 'api':
                response, _ = self.process_with_api(query, agent_id)
            elif agent_type == 'groq':
                response = self.process_with_groq(query, agent_id)
            elif agent_type == 'local':
                response = self.process_with_local_model(query, agent_id)
            elif agent_type == 'together':
                response = self.process_with_together(query, agent_id)
            elif agent_type == 'deepinfra':
                response = self.process_with_deepinfra(query, agent_id)
            elif agent_type == 'anthropic':
                response = self.process_with_anthropic(query, agent_id)
            elif agent_type == 'deepseek':
                response = self.process_with_deepseek(query, agent_id)
            elif agent_type == 'mistral':
                response = self.process_with_mistral(query, agent_id)
            elif agent_type == 'cohere':
                response = self.process_with_cohere(query, agent_id)
            else:
                response = f"No se pudo procesar la consulta con el agente seleccionado: {agent_type}"
        except Exception as e:
            logging.error(f"Error processing query with {agent_type}:{agent_id}: {str(e)}")
            raise

        processing_time = time.time() - start_time
        self.agent_speeds[f"{agent_type}:{agent_id}"] = processing_time
        return response