import spacy
from typing import Dict, Any, Tuple, List
from duckduckgo_search import DDGS
import streamlit as st
import hashlib
from datetime import datetime, timedelta
import requests
import logging
import time
import json
import polars as pl
import re
from openai import OpenAI
from groq import Groq
from anthropic import Anthropic
import cohere

from load_secrets import load_secrets, get_secret, secrets

# Carga todos los secretos al inicio de la aplicación
load_secrets()

try:
    from mistralai import Mistral
except TypeError:
    print("Error al importar Mistral. Usando una implementación alternativa.")
    class Mistral:
        def __init__(self, *args, **kwargs):
            pass
        def chat(self, *args, **kwargs):
            return "Mistral no está disponible en este momento."

# Configurar logging
logging.basicConfig(filename='mallo.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar el modelo de lenguaje de spaCy
@st.cache_resource
def load_nlp_model():
    try:
        # Intentar cargar el modelo en español
        return spacy.load("es_core_news_sm")
    except OSError:
        logging.warning("No se pudo cargar el modelo en español. Intentando descargar...")
        try:
            # Intentar descargar el modelo en español
            spacy.cli.download("es_core_news_sm")
            return spacy.load("es_core_news_sm")
        except Exception as e:
            logging.error(f"No se pudo descargar o cargar el modelo en español: {str(e)}")
            logging.info("Intentando cargar el modelo en inglés como respaldo...")
            try:
                # Intentar cargar el modelo en inglés como respaldo
                return spacy.load("en_core_web_sm")
            except OSError:
                logging.warning("No se pudo cargar el modelo en inglés. Intentando descargar...")
                try:
                    # Intentar descargar el modelo en inglés
                    spacy.cli.download("en_core_web_sm")
                    return spacy.load("en_core_web_sm")
                except Exception as e:
                    logging.error(f"No se pudo descargar o cargar ningún modelo: {str(e)}")
                    raise RuntimeError("No se pudo cargar ningún modelo de lenguaje. Por favor, revisa la instalación de spaCy y los modelos.")

# Uso de la función
try:
    nlp = load_nlp_model()
    logging.info(f"Modelo cargado: {nlp.meta['lang']}_{nlp.meta['name']}")
except Exception as e:
    logging.error(f"Error al cargar el modelo de lenguaje: {str(e)}")
    # Aquí puedes decidir cómo manejar este error, por ejemplo:
    st.error("No se pudo cargar el modelo de lenguaje. Algunas funcionalidades pueden no estar disponibles.")
    nlp = None

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
        "Cohere API": check_cohere_api(),
        "OpenRouter API": check_openrouter_api()
    }
    logging.info(f"System status: {status}")
    return status

def check_openai_api() -> bool:
    try:
        client = OpenAI(api_key=secrets["OPENAI_API_KEY"])
        client.models.list()
        return True
    except Exception as e:
        logging.error(f"Error checking OpenAI API: {str(e)}")
        return False

def check_groq_api() -> bool:
    try:
        client = Groq(api_key=secrets["GROQ_API_KEY"])
        client.chat.completions.create(model="mixtral-8x7b-32768", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking Groq API: {str(e)}")
        return False

def check_local_models(config: Dict[str, Any]) -> bool:
    try:
        response = requests.get(f"{config['ollama']['base_url']}/api/tags")
        return len(response.json()['models']) > 0
    except Exception as e:
        logging.error(f"Error checking local models: {str(e)}")
        return False

def check_web_search() -> bool:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text("test", max_results=1))
            return len(results) > 0
    except Exception as e:
        logging.error(f"Error checking web search: {str(e)}")
        return False

def check_deepinfra_api() -> bool:
    try:
        client = OpenAI(api_key=secrets["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")
        client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking DeepInfra API: {str(e)}")
        return False

def check_anthropic_api() -> bool:
    try:
        client = Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])
        client.messages.create(model="claude-3-haiku-20240307", max_tokens=10, messages=[{"role": "user", "content": "Hello"}])
        return True
    except Exception as e:
        logging.error(f"Error checking Anthropic API: {str(e)}")
        return False

def check_deepseek_api() -> bool:
    try:
        client = OpenAI(api_key=secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
        client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking DeepSeek API: {str(e)}")
        return False

def check_mistral_api() -> bool:
    try:
        client = Mistral(api_key=secrets["MISTRAL_API_KEY"])
        client.chat.complete(model="open-mistral-nemo", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking Mistral API: {str(e)}")
        return False

def check_cohere_api() -> bool:
    try:
        client = cohere.Client(api_key=secrets["COHERE_API_KEY"])
        client.chat(model="command-r-plus", message="Test")
        return True
    except Exception as e:
        logging.error(f"Error checking Cohere API: {str(e)}")
        return False

def check_openrouter_api() -> bool:
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {secrets['OPENROUTER_API_KEY']}",
                "HTTP-Referer": "https://marduk.pro",
                "X-Title": "MALLO",
            },
            data=json.dumps({
                "model": "mattshumer/reflection-70b:free",
                "messages": [{"role": "user", "content": "Test"}]
            })
        )
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Error checking OpenRouter API: {str(e)}")
        return False

def evaluate_query_complexity(query: str, context: str) -> Tuple[float, bool, bool, str]:
    full_text = f"{context}\n\n{query}"
    word_count = len(full_text.split())
    unique_words = len(set(full_text.split()))
    avg_word_length = sum(len(word) for word in full_text.split()) / word_count if word_count > 0 else 0
    
    doc = nlp(full_text)
    sentence_count = len(list(doc.sents))
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    named_entities = len(doc.ents)
    tech_terms = len(re.findall(r'\b(?:API|función|código|programa|algoritmo|base de datos|red|servidor|nube|aprendizaje automático|IA)\b', full_text, re.IGNORECASE))
    
    features = [
        min(word_count / 100, 1),
        min(unique_words / 50, 1),
        min(avg_word_length / 10, 1),
        min(avg_sentence_length / 20, 1),
        min(named_entities / 10, 1),
        min(tech_terms / 5, 1)
    ]
    
    complexity_match = re.search(r"COMPLEJIDAD:\s*([\d.]+)", full_text)
    if complexity_match:
        try:
            complexity = float(complexity_match.group(1).rstrip('.'))
        except ValueError:
            logging.warning(f"Valor de complejidad no válido: {complexity_match.group(1)}. Usando valor calculado.")
            complexity = sum(features) / len(features)
    else:
        complexity = sum(features) / len(features)
    
    needs_web_search = "actualidad" in full_text.lower() or "reciente" in full_text.lower()
    needs_moa = complexity > 0.7 or word_count > 200 or "MOA: SI" in full_text
    
    prompt_type = determine_prompt_type(full_text)

    return complexity, needs_web_search, needs_moa, prompt_type

def determine_prompt_type(text: str) -> str:
    # Convertimos el texto a minúsculas para facilitar la búsqueda
    text = text.lower()

    if re.search(r'\b(matemáticas|cálculo|ecuación|número|álgebra|geometría|trigonometría)\b', text):
        return 'math'
    elif re.search(r'\b(código|programación|función|algoritmo|software|desarrollo|lenguaje de programación|python|java)\b', text):
        return 'coding'
    elif re.search(r'\b(ley|legal|legislación|corte|derechos|demanda|abogado|juez|constitución|código civil|código penal)\b', text):
        return 'legal'
    elif re.search(r'\b(ciencia|experimento|hipótesis|teoría|investigación|laboratorio|método científico|biología|física|química)\b', text):
        return 'scientific'
    elif re.search(r'\b(historia|histórico|época|siglo|período|civilización|antiguo|colonial|independencia|república)\b', text):
        return 'historical'
    elif re.search(r'\b(filosofía|filosófico|ética|moralidad|metafísica|epistemología|lógica|existencialismo)\b', text):
        return 'philosophical'
    elif re.search(r'\b(ética|moral|correcto|incorrecto|deber|valor|principio|dilema ético)\b', text):
        return 'ethical'
    elif re.search(r'\b(colombia|colombiano|bogotá|medellín|cali|barranquilla|cartagena|andes|caribe|pacífico|amazonas)\b', text):
        return 'colombian_context'
    elif re.search(r'\b(cultura|tradición|costumbre|folclor|gastronomía|música|arte|literatura|deporte)\b', text):
        return 'cultural'
    elif re.search(r'\b(política|gobierno|congreso|presidente|elecciones|partidos|constitución|democracia)\b', text):
        return 'political'
    elif re.search(r'\b(economía|finanzas|mercado|empleo|impuestos|inflación|pib|comercio|industria)\b', text):
        return 'economic'
    else:
        return 'default'

def select_best_agent(query: str, agents: List[Dict[str, Any]]) -> str:
    complexity, _, _ = evaluate_query_complexity(query, "")
    
    scores = {}
    for agent in agents:
        suitability_score = calculate_suitability(query, agent)
        scores[agent['name']] = suitability_score
    
    best_agent = max(scores, key=scores.get)
    return best_agent

def calculate_suitability(query: str, agent: Dict[str, Any]) -> float:
    keywords = agent.get('keywords', [])
    return sum(keyword.lower() in query.lower() for keyword in keywords) / len(keywords) if keywords else 0.5

def perform_web_search(query: str) -> str:
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

cache = {}

def cache_response(query: str, response: Tuple[str, Dict[str, Any]]):
    key = hashlib.md5(query.encode()).hexdigest()
    cache[key] = {
        'response': response,
        'timestamp': datetime.now()
    }

def get_cached_response(query: str) -> Tuple[str, Dict[str, Any]] or None:
    key = hashlib.md5(query.encode()).hexdigest()
    if key in cache:
        cached_item = cache[key]
        if datetime.now() - cached_item['timestamp'] < timedelta(hours=1):
            return cached_item['response']
    return None

def summarize_text(text: str, max_length: int = 200) -> str:
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
    if isinstance(error_message, dict):
        st.error(json.dumps(error_message, indent=2))
    else:
        st.error(str(error_message))

def log_warning(warning_message: str):
    logging.warning(warning_message)
    st.warning(warning_message)

def log_info(info_message: str):
    logging.info(info_message)
    st.info(info_message)

def update_agent_performance(agent_name: str, success: bool):
    # Esta función se deja como un placeholder para futuras implementaciones
    pass