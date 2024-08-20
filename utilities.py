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
from mistralai import Mistral
import cohere

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
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        client.models.list()
        return True
    except Exception as e:
        logging.error(f"Error checking OpenAI API: {str(e)}")
        return False

def check_groq_api() -> bool:
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
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
        client = OpenAI(api_key=st.secrets["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")
        client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking DeepInfra API: {str(e)}")
        return False

def check_anthropic_api() -> bool:
    try:
        client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        client.messages.create(model="claude-3-haiku-20240307", max_tokens=10, messages=[{"role": "user", "content": "Hello"}])
        return True
    except Exception as e:
        logging.error(f"Error checking Anthropic API: {str(e)}")
        return False

def check_deepseek_api() -> bool:
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
        client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking DeepSeek API: {str(e)}")
        return False

def check_mistral_api() -> bool:
    try:
        client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
        client.chat.complete(model="open-mistral-nemo", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking Mistral API: {str(e)}")
        return False

def check_cohere_api() -> bool:
    try:
        client = cohere.Client(api_key=st.secrets["COHERE_API_KEY"])
        client.chat(model="command-r-plus", message="Test")
        return True
    except Exception as e:
        logging.error(f"Error checking Cohere API: {str(e)}")
        return False

def evaluate_query_complexity(query: str) -> Tuple[float, bool, bool]:
    word_count = len(query.split())
    unique_words = len(set(query.split()))
    avg_word_length = sum(len(word) for word in query.split()) / word_count if word_count > 0 else 0
    
    doc = nlp(query)
    sentence_count = len(list(doc.sents))
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    named_entities = len(doc.ents)
    tech_terms = len(re.findall(r'\b(?:API|function|code|program|algorithm|database|network|server|cloud|machine learning|AI)\b', query, re.IGNORECASE))
    
    features = [
        min(word_count / 50, 1),
        min(unique_words / 30, 1),
        min(avg_word_length / 10, 1),
        min(avg_sentence_length / 20, 1),
        min(named_entities / 5, 1),
        min(tech_terms / 3, 1)
    ]
    
    complexity = sum(features) / len(features)
    needs_web_search = "actualidad" in query.lower() or "reciente" in query.lower()
    needs_moa = complexity > 0.7
    
    return complexity, needs_web_search, needs_moa

def select_best_agent(query: str, agents: List[Dict[str, Any]]) -> str:
    complexity, _, _ = evaluate_query_complexity(query)
    
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