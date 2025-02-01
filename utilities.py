import spacy
from typing import Dict, Any, Tuple, List, Optional
from duckduckgo_search import DDGS
import streamlit as st
import hashlib
from datetime import datetime, timedelta
import random
import requests
import logging
import traceback
import time
import json
import polars as pl
import re
from openai import OpenAI
from groq import Groq
from anthropic import Anthropic
import cohere
from collections import Counter
import yaml
from together import Together
from tavily import TavilyClient

from load_secrets import load_secrets, get_secret, secrets

# Carga todos los secretos al inicio de la aplicación
load_secrets()

# Cargar la configuración desde config.yaml
def load_config() -> Dict[str, Any]:
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error("No se pudo encontrar el archivo config.yaml")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error al leer el archivo config.yaml: {e}")
        return {}

# Cargar la configuración
config = load_config()

# Extraer los tipos de prompt de la configuración
critical_analysis_config = config.get('critical_analysis', {})
prompt_types = critical_analysis_config.get('prompts', {})

# Si prompt_types está vacío, usa un diccionario predeterminado
if not prompt_types:
    prompt_types = {
        'default': 'Análisis general',
        'math': 'Análisis matemático',
        'coding': 'Análisis de código',
        'legal': 'Análisis legal',
        'scientific': 'Análisis científico',
        'historical': 'Análisis histórico',
        'philosophical': 'Análisis filosófico',
        'ethical': 'Análisis ético',
        'colombian_context': 'Análisis del contexto colombiano',
        'cultural': 'Análisis cultural',
        'political': 'Análisis político',
        'economic': 'Análisis económico',
        'general': 'Análisis general',
        'audio_transcription': 'Análisis de transcripción de audio',
        'multimodal': 'Análisis multimodal',
        'tool_use': 'Análisis de uso de herramientas',
        'content_moderation': 'Análisis de moderación de contenido',
        'creative': 'Análisis creativo',
        'analytical': 'Análisis analítico'
    }

# Extraer otras configuraciones relevantes
general_config = config.get('general', {})
thresholds = config.get('thresholds', {})

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
        "OpenRouter API": check_openrouter_api(),
        "Together API": check_together_api()
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

# Conexión con Together API
'''
import os
from together import Together

client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

response = client.chat.completions.create(
    model="meta-llama/Llama-Vision-Free",
    messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
)
print(response.choices[0].message.content)
'''
def check_together_api() -> bool:
    try:
        client = Together(api_key=secrets["TOGETHER_API_KEY"])
        client.chat.completions.create(model="Qwen/Qwen2.5-72B-Instruct-Turbo", messages=[{"role": "user", "content": "Test"}])
        return True
    except Exception as e:
        logging.error(f"Error checking Together API: {str(e)}")
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

def perform_web_search(query: str, max_retries: int = 3) -> str:
    search_methods = ['you', 'tavily', 'duckduckgo']

    for method in search_methods:
        result = None
        if method == 'you':
            result = you_search(query)
        elif method == 'tavily':
            result = tavily_search(query)
        elif method == 'duckduckgo':
            result = duckduckgo_search(query, max_retries)

        if result:
            return result

    return "No se pudo completar la búsqueda web después de varios intentos."

def you_search(query: str) -> str:
    try:
        headers = {"X-API-Key": get_secret("YOU_API_KEY")}
        params = {"query": query}
        response = requests.get(
            "https://api.ydc-index.io/search",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
        if 'snippets' in data:
            return "\n".join([snippet['content'] for snippet in data['snippets'][:3]])
        return None
    except Exception as e:
        logging.error(f"Error in YOU search: {str(e)}")
        return None

def tavily_search(query: str) -> str:
    try:
        client = TavilyClient(api_key=get_secret("TAVILY_API_KEY"))
        response = client.search(query)
        if response and 'results' in response:
            return "\n".join([result['content'] for result in response['results'][:3]])
        return None
    except Exception as e:
        logging.error(f"Error in Tavily search: {str(e)}")
        return None

def duckduckgo_search(query: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([result['body'] for result in results])
            return None
        except Exception as e:
            if "Ratelimit" in str(e) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                logging.error(f"Error in DuckDuckGo search: {str(e)}")
                return None
    return None

def check_web_search() -> bool:
    try:
        result = perform_web_search("test query")
        return result is not None
    except Exception as e:
        logging.error(f"Error checking web search: {str(e)}")
        return False

def check_deepinfra_api() -> bool:
    try:
        client = OpenAI(api_key=secrets["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")
        client.chat.completions.create(model="Qwen/Qwen2.5-72B-Instruct", messages=[{"role": "user", "content": "Test"}])
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
        client.chat(model="command-r7b-12-2024", message="Test")
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
                "model": "qwen/qwen-2-7b-instruct:free",
                "messages": [{"role": "user", "content": "Test"}]
            })
        )
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Error checking OpenRouter API: {str(e)}")
        return False

# Función para evaluar la complejidad de una consulta
def evaluate_query_complexity(query: str, context: str) -> Tuple[float, bool, bool, str]:
    full_text = f"{context}\n\n{query}"
    
    # Análisis básico del texto
    word_count = len(full_text.split())
    unique_words = len(set(full_text.split()))
    avg_word_length = sum(len(word) for word in full_text.split()) / word_count if word_count > 0 else 0
    
    # Análisis con spaCy
    doc = nlp(full_text)
    sentence_count = len(list(doc.sents))
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    named_entities = len(doc.ents)
    
    # Búsqueda de términos técnicos (ampliada para incluir términos en español e inglés)
    tech_terms = len(re.findall(r'\b(?:API|función|function|código|code|programa|program|algoritmo|algorithm|base de datos|database|red|network|servidor|server|nube|cloud|aprendizaje automático|machine learning|IA|AI)\b', full_text, re.IGNORECASE))
    
    # Cálculo de características
    features = [
        min(word_count / 100, 1),
        min(unique_words / 50, 1),
        min(avg_word_length / 10, 1),
        min(avg_sentence_length / 20, 1),
        min(named_entities / 10, 1),
        min(tech_terms / 5, 1)
    ]
    
    # Cálculo de complejidad inicial
    complexity = sum(features) / len(features)
    
    # Determinar si se necesita búsqueda web
    needs_web_search = any(term in full_text.lower() for term in ["actualidad", "reciente", "último", "nueva", "actual", "recent", "latest", "new", "current"])
    
    # Determinar si se necesita MOA (Mixture of Agents)
    needs_moa = complexity > 0.7 or word_count > 200 or "MOA: SI" in full_text
    
    # Determinar el tipo de prompt y las capacidades requeridas
    prompt_type = determine_prompt_type(full_text)
    
    # Ajustar la complejidad basada en el tipo de prompt
    prompt_type_complexity_modifiers = {
        'legal': 0.2,
        'scientific': 0.15,
        'philosophical': 0.2,
        'economic': 0.1,
        'political': 0.1,
        'math': 0.2,
        'coding': 0.2,
        'creative': 0.1,
        'analytical': 0.15,
        'multimodal': 0.1,
        'audio_transcription': 0.05,
        'tool_use': 0.15,
        'content_moderation': 0.05
    }
    complexity += prompt_type_complexity_modifiers.get(prompt_type, 0)
    
    # Normalizar la complejidad final
    complexity = min(max(complexity, 0), 1)
    
    return complexity, needs_web_search, needs_moa, prompt_type

def _determine_prompt_type(text: str) -> str:
    text = text.lower()
    for p_type, prompt_info in prompt_types.items():
        # Asegúrate de que prompt_info sea un diccionario y tenga una clave 'keywords'
        if isinstance(prompt_info, dict) and 'keywords' in prompt_info:
            keywords = prompt_info['keywords']
            if isinstance(keywords, list):
                if any(keyword.lower() in text for keyword in keywords):
                    return p_type
            elif isinstance(keywords, str):
                if keywords.lower() in text:
                    return p_type
    return 'default'

def determine_prompt_type_and_capabilities(text: str) -> Tuple[str, List[str]]:
    text = text.lower()
    prompt_type = 'general'
    capabilities = []
    
    # Usar los tipos de prompt definidos en config.yaml o los predeterminados
    for p_type, prompt in prompt_types.items():
        if any(keyword.lower() in text for keyword in prompt.lower().split()):
            prompt_type = p_type
            break
    
    # Determinar capacidades requeridas basadas en el contenido
    if re.search(r'\b(código|programación|function|algorithm)\b', text):
        capabilities.append("code_generation")
    if re.search(r'\b(matemáticas|cálculo|ecuación|algebra)\b', text):
        capabilities.append("advanced_reasoning")
    if re.search(r'\b(imagen|visual|gráfico)\b', text):
        capabilities.append("vision_language")
    if re.search(r'\b(audio|voz|sonido)\b', text):
        capabilities.append("speech_to_text")
    if len(text.split()) > 500:
        capabilities.append("long_context")
    if re.search(r'\b(análisis|evalúa|compara)\b', text):
        capabilities.append("critical_thinking")
    
    return prompt_type, capabilities

# Función para determinar el tipo de prompt
def determine_prompt_type(text: str) -> str:
    # Convertimos el texto a minúsculas para facilitar la búsqueda
    text = text.lower()

    # Diccionario de tipos de prompt y sus palabras clave asociadas
    prompt_types: Dict[str, List[str]] = {
        'math': ["matemáticas", "matematicas", "cálculo", "calculo", "ecuación", "ecuacion", "número", "numero",
                 "álgebra", "algebra", "geometría", "geometria", "trigonometría", "trigonometria", "estadística",
                 "probabilidad", "análisis", "calculo diferencial", "calculo integral", "lógica matemática"],
        'coding': ["código", "programación", "programacion", "función", "funcion", "algoritmo", "software",
                   "desarrollo", "lenguaje de programación", "lenguaje de programacion", "python", "java", "c++",
                   "javascript", "html", "css", "base de datos", "base de datos", "sql", "git", "github", "depuración",
                   "depuracion", "algoritmos", "estructuras de datos", "estructuras de datos"],
        'legal': ["ley", "legal", "legislación", "legislacion", "corte", "derechos", "demanda", "abogado",
                  "juez", "constitución", "constitucion", "código civil", "codigo civil", "código penal",
                  "codigo penal", "jurisprudencia", "tribunal", "sentencia", "acusación", "acusacion", "defensa",
                  "contrato", "delito", "pena", "proceso judicial", "proceso judicial"],
        'scientific': ["ciencia", "experimento", "hipótesis", "hipotesis", "teoría", "teoria", "investigación",
                       "investigacion", "laboratorio", "método científico", "metodo cientifico", "biología", "biologia",
                       "física", "fisica", "química", "quimica", "astronomía", "astronomia", "geología", "geologia",
                       "ecología", "ecologia", "medio ambiente", "medio ambiente", "cambio climático", "cambio climatico"],
        'historical': ["historia", "histórico", "historico", "época", "epoca", "siglo", "período", "periodo",
                       "civilización", "civilizacion", "antiguo", "colonial", "independencia", "república",
                       "republica", "revolución", "revolucion", "guerra", "paz", "dictadura", "democracia",
                       "imperio", "colonia", "prehistoria", "edad media", "edad moderna", "edad contemporánea",
                       "edad contemporanea"],
        'philosophical': ["filosofía", "filosofia", "filosófico", "filosofico", "ética", "etica", "moralidad",
                          "metafísica", "metafisica", "epistemología", "epistemologia", "lógica", "logica",
                          "existencialismo", "razón", "razon", "conocimiento", "ser", "nada", "libertad",
                          "determinismo", "alma", "cuerpo", "realidad", "apariencia", "verdad", "mentira", "belleza",
                          "fealdad", "bien", "mal", "virtud", "vicio"],
        'ethical': ["ética", "etica", "moral", "correcto", "incorrecto", "deber", "valor", "principio",
                    "dilema ético", "dilema etico", "responsabilidad", "justicia", "honestidad", "integridad",
                    "respeto", "compasión", "solidaridad", "empatía", "empatia", "altruismo", "egoísmo", "egoismo"],
        'colombian_context': ["colombia", "colombiano", "bogotá", "medellín", "cali", "barranquilla", "cartagena",
                              "andes", "caribe", "pacífico", "pacifico", "amazonas", "orinoquía", "orinoquia",
                              "café", "cafe", "vallenato", "cumbia", "gabriel garcía márquez", "gabriel garcia marquez",
                              "feria de las flores", "feria de las flores", "carnaval de barranquilla",
                              "carnaval de barranquilla"],
        'cultural': ["cultura", "tradición", "tradicion", "costumbre", "folclor", "folclore", "gastronomía",
                     "gastronomia", "música", "musica", "arte", "literatura", "deporte", "cine", "teatro", "danza",
                     "pintura", "escultura", "arquitectura", "fotografía", "fotografia", "moda", "diseño", "diseño",
                     "videojuegos", "videojuegos", "festival", "celebración", "celebracion", "ritual", "mito",
                     "leyenda"],
        'political': ["política", "politica", "gobierno", "congreso", "presidente", "elecciones", "partidos",
                      "constitución", "constitucion", "democracia", "izquierda", "derecha", "centro", "liberal",
                      "conservador", "socialista", "comunista", "capitalista", "anarquista", "feminista",
                      "ecologista", "nacionalista", "globalista", "poder", "autoridad", "estado", "nación",
                      "nacion", "ciudadanía", "ciudadania", "derechos humanos", "derechos humanos", "libertad de expresión",
                      "libertad de expresion", "igualdad", "justicia social", "justicia social"],
        'economic': ["economía", "economia", "finanzas", "mercado", "empleo", "impuestos", "inflación", "inflacion",
                     "pib", "comercio", "industria", "producción", "produccion", "consumo", "ahorro", "inversión",
                     "inversion", "oferta", "demanda", "precio", "competencia", "monopolio", "oligopolio",
                     "globalización", "globalizacion", "desempleo", "desempleo", "pobreza", "riqueza", "desigualdad",
                     "crecimiento económico", "crecimiento economico", "desarrollo sostenible", "desarrollo sostenible"],

        'general': ["general", "básico", "basico", "común", "comun", "ordinario", "típico", "tipico", 
                    "estándar", "estandar", "normal", "regular", "convencional", "usual", "habitual", 
                    "frecuente", "cotidiano", "diario", "rutinario"],
        
        'audio_transcription': ["transcripción", "transcripcion", "audio", "voz", "sonido", "grabación", 
                                "grabacion", "speech to text", "reconocimiento de voz", "reconocimiento del habla", 
                                "subtítulos", "subtitulos", "conversión de audio a texto", "conversion de audio a texto"],
        
        'multimodal': ["multimodal", "texto e imagen", "imagen y texto", "visual y textual", "multimedia", 
                       "contenido mixto", "análisis de imagen", "analisis de imagen", "descripción de imagen", 
                       "descripcion de imagen", "interpretación visual", "interpretacion visual"],
        
        'tool_use': ["uso de herramientas", "herramientas", "APIs", "integración", "integracion", "funciones externas", 
                     "llamadas a API", "automatización", "automatizacion", "scripts", "plugins", "extensiones", 
                     "interacción con sistemas", "interaccion con sistemas"],
        
        'content_moderation': ["moderación", "moderacion", "filtrado de contenido", "seguridad de contenido", 
                               "detección de spam", "deteccion de spam", "control de calidad", "revisión de contenido", 
                               "revision de contenido", "políticas de contenido", "politicas de contenido", 
                               "contenido inapropiado", "contenido ofensivo"],
        
        'creative': ["creativo", "imaginativo", "innovador", "original", "artístico", "artistico", "inventivo", 
                     "inspirador", "novedoso", "único", "unico", "fuera de lo común", "fuera de lo comun", 
                     "pensamiento lateral", "lluvia de ideas", "brainstorming", "diseño creativo", "diseño creativo"],
        
        'analytical': ["analítico", "analitico", "análisis", "analisis", "evaluación", "evaluacion", "crítico", 
                       "critico", "examen detallado", "investigación", "investigacion", "estudio", "revisión", 
                       "revision", "interpretación", "interpretacion", "diagnóstico", "diagnostico", 
                       "resolución de problemas", "resolucion de problemas"]
    }

    # Contador para las coincidencias de cada tipo
    type_counts = Counter()
    
    # Recorremos cada tipo de prompt y sus palabras clave asociadas
    for prompt_type, keywords in prompt_types.items():
        # Contamos las coincidencias de cada palabra clave en el texto
        keyword_matches = sum(1 for keyword in keywords if re.search(r'\b' + re.escape(keyword) + r'\b', text))
        # Almacenamos el número de coincidencias para este tipo de prompt
        type_counts[prompt_type] = keyword_matches
    
    # Determinamos el tipo de prompt con más coincidencias
    most_common_type = type_counts.most_common(1)
    
    # Si hay al menos una coincidencia, devolvemos el tipo de prompt más común
    if most_common_type:
        return most_common_type[0][0]
    else:
        # Si no hay coincidencias, devolvemos un tipo de prompt por defecto
        return 'general'

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

cache = {}

def cache_response(query: str, response: Tuple[str, Dict[str, Any]]):
    key = hashlib.md5(query.encode()).hexdigest()
    cache[key] = {
        'response': response,
        'timestamp': datetime.now()
    }

def get_cached_response(query: str) -> Tuple[str, Dict[str, Any]]:
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

# Funciones de main.py

def load_config():
    """Carga la configuración desde el archivo YAML."""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("No se pudo encontrar el archivo de configuración 'config.yaml'.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error al leer el archivo de configuración: {str(e)}")
        st.stop()

def load_speed_test_results():
    """Carga los resultados de las pruebas de velocidad."""
    try:
        with open('model_speeds.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def display_speed_test_results(results):
    """Muestra los resultados de las pruebas de velocidad en la barra lateral."""
    data = [
        {"API": api, "Modelo": model['model'], "Velocidad": f"{model['speed']:.4f}"}
        for api, models in results.items()
        for model in models
    ]
    df = pl.DataFrame(data).sort("Velocidad")
    with st.sidebar.expander("📊 Resultados de Velocidad", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

def render_sidebar_content(system_status: Dict[str, bool], speed_test_results: Optional[Dict]):
    """Renderiza el contenido de la barra lateral con diseño mejorado."""
    with st.sidebar:
        # Cabecera
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("assets/logo.jpg", width=50)
        with col2:
            st.markdown("### MALLO")
            st.caption("MultiAgent LLM Orchestrator")
        
        # Métricas principales
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Agentes", "10+")
        with col2:
            st.metric("Modelos", "20+")
        
        # Enlaces principales
        st.markdown("""
        [![ver código fuente](https://img.shields.io/badge/Repositorio%20GitHub-gris?logo=github)](https://github.com/bladealex9848/MALLO)
        ![Visitantes](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fmallollm.streamlit.app&label=Visitantes&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
        """)
        
        # Estado del Sistema
        with st.expander("🔧 Sistema", expanded=False):
            st.caption("Estado de los componentes")
            for key, value in system_status.items():
                if value:
                    st.success(key, icon="✅")
                else:
                    st.error(key, icon="❌")
            
            if speed_test_results:
                st.markdown("##### Rendimiento")
                display_speed_test_results(speed_test_results)
        
        # Capacidades
        with st.expander("💡 Capacidades", expanded=False):
            features = {
                "🤖 Múltiples Modelos": "Integración con principales proveedores de IA",
                "🔍 Análisis Contextual": "Comprensión profunda de consultas",
                "🌐 Búsqueda Web": "Información actualizada en tiempo real",
                "⚖️ Evaluación Ética": "Respuestas alineadas con principios éticos",
                "🔄 Meta-análisis": "Síntesis de múltiples fuentes",
                "🎯 Prompts Adaptados": "Especialización por tipo de consulta"
            }
            
            for title, description in features.items():
                st.markdown(f"**{title}**")
                st.caption(description)
        
        # Acerca de
        with st.expander("ℹ️ Acerca de", expanded=False):
            st.markdown("""
            MALLO es un orquestador avanzado de IAs que selecciona y coordina 
            múltiples modelos de lenguaje para proporcionar respuestas óptimas 
            basadas en el contexto y complejidad de cada consulta.
            """)
        
        # Desarrollador
        st.markdown("---")
        st.markdown("### 👨‍💻 Desarrollador")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("assets/profile.jpg", width=60)
        with col2:
            st.markdown("#### Alexander Oviedo Fadul")
            st.caption("Developer & Legal Tech")
        
        # Enlaces sociales
        st.markdown("##### Contacto")
        social_links = {
            "🌐 Website": "https://alexanderoviedofadul.dev",
            "💼 LinkedIn": "https://linkedin.com/in/alexander-oviedo-fadul",
            "📱 WhatsApp": "https://wa.me/573015930519",
            "📧 Email": "mailto:alexander.oviedo.fadul@gmail.com",
            "🐙 GitHub": "https://github.com/bladealex9848"
        }
        
        for platform, link in social_links.items():
            st.markdown(f"[{platform}]({link})")

# Resumen de la conversación con un límite de longitud máximo (500 caracteres)
# Si el contexto supera el límite, se utiliza cohere o OpenAI para resumirlo
def summarize_conversation(
    previous_context, user_input, response, agent_manager, config, max_length=500
):
    new_content = f"Usuario: {user_input}\nAsistente: {response}"
    updated_context = f"{previous_context}\n\n{new_content}".strip()

    if len(updated_context) > max_length:
        summary_prompt = (
            f"Resume la siguiente conversación manteniendo los puntos clave:\n\n{updated_context}"
        )

        try:
            # Intenta usar openrouter primero
            # summary = agent_manager.process_with_openrouter(summary_prompt, config['openrouter']['fast_models'])
            # Intenta usar cohere primero
            summary = agent_manager.process_query(
                summary_prompt, "cohere", config["cohere"]["default_model"]
            )
        except Exception as e:
            log_error(
                f"Error al usar cohere para resumen: {str(e)}. Usando OpenAI como respaldo."
            )
            try:
                # Si cohere falla, usa OpenAI como respaldo
                summary = agent_manager.process_query(
                    summary_prompt, "api", config["openai"]["default_model"]
                )
            except Exception as e:
                log_error(
                    f"Error al usar OpenAI para resumen: {str(e)}. Devolviendo contexto sin resumir."
                )
                return updated_context

        return summary

    return updated_context

def evaluate_ethical_compliance(response: str, prompt_type: str) -> Dict[str, Any]:
    """
    Evalúa el cumplimiento ético y legal de la respuesta generada.

    Args:
    response (str): La respuesta generada por el sistema.
    prompt_type (str): El tipo de prompt utilizado para generar la respuesta.

    Returns:
    Dict[str, Any]: Un diccionario con los resultados de la evaluación.
    """
    evaluation = {
        "sesgo_detectado": False,
        "privacidad_respetada": True,
        "transparencia": True,
        "alineacion_derechos_humanos": True,
        "responsabilidad": True,
        "explicabilidad": True,
    }

    # Verificar sesgos (esto requeriría un modelo más sofisticado en la práctica)
    if any(
        palabra in response.lower()
        for palabra in ["todos los hombres", "todas las mujeres"]
    ):
        evaluation["sesgo_detectado"] = True

    # Verificar privacidad (ejemplo simplificado)
    if any(
        dato in response
        for dato in ["número de identificación", "dirección", "teléfono"]
    ):
        evaluation["privacidad_respetada"] = False

    # Verificar transparencia
    if "Esta respuesta fue generada por IA" not in response:
        evaluation["transparencia"] = False

    # La alineación con derechos humanos, responsabilidad y explicabilidad
    # requerirían análisis más complejos en un sistema real

    return evaluation

# Evaluar la respuesta del modelo de lenguaje y proporcionar retroalimentación
def evaluate_response(agent_manager, config, evaluation_type, query, response=None):
    eval_config = config["evaluation_models"][evaluation_type]

    if evaluation_type == "initial":
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
                evaluation = agent_manager.process_query(
                    evaluation_prompt, eval_config["api"], eval_config["model"]
                )
            elif attempt == 1:
                evaluation = agent_manager.process_query(
                    evaluation_prompt,
                    eval_config["backup_api"],
                    eval_config["backup_model"],
                )
            else:
                evaluation = agent_manager.process_query(
                    evaluation_prompt,
                    eval_config["backup_api2"],
                    eval_config["backup_model2"],
                )

            if "Error al procesar" not in evaluation:
                return evaluation
        except Exception as e:
            log_error(
                f"Error en evaluación {evaluation_type} con {'modelo principal' if attempt == 0 else 'modelo de respaldo'}: {str(e)}"
            )

    return "No se pudo realizar la evaluación debido a múltiples errores en los modelos de evaluación."

def export_conversation_to_md(messages, details):
    """Exporta la conversación completa a Markdown."""
    md_content = "# Conversación con MALLO\n\n"
    md_content += f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for message in messages:
        if message["role"] == "user":
            md_content += f"## 👤 Usuario\n\n{message['content']}\n\n"
        elif message["role"] == "assistant":
            md_content += f"## 🤖 Asistente\n\n{message['content']}\n\n"
            
            if details:
                md_content += "### 🔍 Detalles del Proceso\n\n"
                md_content += f"#### Razonamiento\n\n{details['initial_evaluation']}\n\n"
                md_content += f"#### Evaluación Ética\n\n```json\n{json.dumps(details['ethical_evaluation'], indent=2)}\n```\n\n"
                
                if details.get("improved_response"):
                    md_content += f"#### ✨ Respuesta Mejorada\n\n{details['improved_response']}\n\n"
                
                if details.get("meta_analysis"):
                    md_content += f"#### 🔄 Meta-análisis\n\n{details['meta_analysis']}\n\n"
                
                md_content += f"#### 📝 Métricas de Rendimiento\n\n```json\n{json.dumps(details['performance_metrics'], indent=2)}\n```\n\n"
                md_content += f"#### 🌐 Contexto de la Conversación\n\n{st.session_state.get('context', 'No disponible')}\n\n"
                md_content += "---\n\n"

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"mallo_conversation_{timestamp}.md"
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return filepath, filename

def render_response_details(details: Dict):
    """Renderiza los detalles de la respuesta en expansores organizados."""
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("💭 Proceso de Razonamiento", expanded=False):
            st.markdown(details["initial_evaluation"])
            st.download_button(
                "📥 Exportar Razonamiento",
                details["initial_evaluation"],
                "razonamiento.md",
                mime="text/markdown"
            )
    
    with col2:
        with st.expander("⚖️ Evaluación Ética", expanded=False):
            st.json(details["ethical_evaluation"])
            if details.get("improved_response"):
                st.info("Respuesta mejorada éticamente:")
                st.write(details["improved_response"])
                
def process_user_input(user_input: str, config: Dict[str, Any], agent_manager: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Procesa la entrada del usuario y genera una respuesta usando los agentes apropiados.
    
    Args:
        user_input (str): La consulta del usuario
        config (Dict[str, Any]): Configuración del sistema
        agent_manager (Any): Instancia del gestor de agentes
    
    Returns:
        Tuple[str, Dict[str, Any]]: (respuesta, detalles del procesamiento)
    """
    try:
        conversation_context = st.session_state.get('context', '')
        enriched_query = f"{conversation_context}\n\nNueva consulta: {user_input}"

        # Verificar caché
        cached_response = get_cached_response(enriched_query)
        if cached_response:
            st.success("💾 Respuesta recuperada de caché")
            return cached_response

        progress_placeholder = st.empty()
        
        # Iniciar procesamiento
        start_time = time.time()
        progress_placeholder.write("🔍 Evaluando consulta...")
        
        # Evaluación inicial
        initial_evaluation = evaluate_response(
            agent_manager, config, 'initial', enriched_query
        )
        
        # Análisis de complejidad y necesidades
        complexity, needs_web_search, needs_moa, prompt_type = evaluate_query_complexity(
            initial_evaluation, ""
        )
        prompt_type = agent_manager.validate_prompt_type(user_input, prompt_type)
        
        # Búsqueda web si es necesaria
        web_context = ""
        if needs_web_search:
            progress_placeholder.write("🌐 Realizando búsqueda web...")
            web_context = perform_web_search(user_input)
            enriched_query = f"{enriched_query}\nContexto web: {web_context}"
        
        # Selección de agentes
        progress_placeholder.write("🤖 Seleccionando agentes...")
        specialized_agent = agent_manager.select_specialized_agent(enriched_query)
        general_agents = agent_manager.get_prioritized_agents(
            enriched_query, complexity, prompt_type
        )

        # Priorización de agentes
        prioritized_agents = []
        if specialized_agent:
            prioritized_agents.append((
                specialized_agent['type'],
                specialized_agent['id'],
                specialized_agent['name']
            ))

        for agent in general_agents:
            if len(prioritized_agents) >= 2:
                break
            if agent not in prioritized_agents:
                prioritized_agents.append(agent)

        prioritized_agents = prioritized_agents[:2]
        
        # Procesamiento con agentes
        agent_results = []
        for agent_type, agent_id, agent_name in prioritized_agents:
            progress_placeholder.write(f"⚙️ Procesando con {agent_name}...")
            try:
                enriched_query_with_prompt = agent_manager.apply_specialized_prompt(
                    enriched_query, prompt_type
                )
                result = agent_manager.process_query(
                    enriched_query_with_prompt,
                    agent_type,
                    agent_id,
                    prompt_type
                )
                agent_results.append({
                    "agent": agent_type,
                    "model": agent_id,
                    "name": agent_name,
                    "status": "success",
                    "response": result
                })
            except Exception as e:
                logger.error(f"Error en el procesamiento con {agent_name}: {str(e)}")
                agent_results.append({
                    "agent": agent_type,
                    "model": agent_id,
                    "name": agent_name,
                    "status": "error",
                    "response": str(e)
                })

        # Verificar respuestas exitosas
        successful_responses = [
            r for r in agent_results if r["status"] == "success"
        ]
        
        if not successful_responses:
            raise ValueError("No se pudo obtener una respuesta válida de ningún agente")

        # Meta-análisis si es necesario
        if needs_moa and len(successful_responses) > 1:
            progress_placeholder.write("🔄 Realizando meta-análisis...")
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

        # Evaluación ética
        progress_placeholder.write("⚖️ Evaluando cumplimiento ético...")
        ethical_evaluation = evaluate_ethical_compliance(final_response, prompt_type)
        
        # Mejora ética si es necesaria
        if any(not value for value in ethical_evaluation.values()):
            progress_placeholder.write("✨ Mejorando respuesta...")
            specialized_assistant = agent_manager.get_specialized_assistant(
                'asst_F33bnQzBVqQLcjveUTC14GaM'
            )
            enhancement_prompt = f"""
            Analiza la siguiente respuesta y su evaluación ética:

            Respuesta: {final_response}

            Evaluación ética: {json.dumps(ethical_evaluation, indent=2)}

            Por favor, modifica la respuesta para mejorar su alineación con principios éticos y legales,
            abordando cualquier preocupación identificada en la evaluación. Asegúrate de que la respuesta sea
            transparente sobre el uso de IA, libre de sesgos, y respetuosa de los derechos humanos y la privacidad.
            """
            improved_response = agent_manager.process_query(
                enhancement_prompt,
                'assistant',
                specialized_assistant['id']
            )
            improved_ethical_evaluation = evaluate_ethical_compliance(
                improved_response,
                prompt_type
            )
        else:
            improved_response = None
            improved_ethical_evaluation = None

        # Evaluación final
        progress_placeholder.write("📝 Evaluación final...")
        final_evaluation = evaluate_response(
            agent_manager,
            config,
            'final',
            user_input,
            final_response
        )

        # Cálculo del tiempo de procesamiento
        processing_time = time.time() - start_time

        # Preparar detalles de la respuesta
        details = {
            "selected_agents": [
                {
                    "agent": r["agent"],
                    "model": r["model"],
                    "name": r["name"]
                } for r in agent_results
            ],
            "processing_time": f"{processing_time:.2f} segundos",
            "complexity": complexity,
            "needs_web_search": needs_web_search,
            "needs_moa": needs_moa,
            "web_context": web_context,
            "prompt_type": prompt_type,
            "initial_evaluation": initial_evaluation,
            "agent_processing": agent_results,
            "final_evaluation": final_evaluation,
            "ethical_evaluation": ethical_evaluation,
            "improved_response": improved_response,
            "improved_ethical_evaluation": improved_ethical_evaluation,
            "performance_metrics": {
                "total_agents_called": len(agent_results),
                "successful_responses": len(successful_responses),
                "failed_responses": len(agent_results) - len(successful_responses),
                "average_response_time": f"{processing_time:.2f} segundos"
            },
            "meta_analysis": meta_analysis_result if needs_moa and len(successful_responses) > 1 else None,
            "final_response": final_response
        }

        # Actualizar contexto
        new_context = summarize_conversation(
            conversation_context,
            user_input,
            final_response,
            agent_manager,
            config
        )
        st.session_state["context"] = new_context

        # Guardar en caché
        cache_response(enriched_query, (final_response, details))

        # Limpiar placeholder de progreso
        progress_placeholder.empty()
        
        return final_response, details

    except Exception as e:
        logger.error(f"Error en process_user_input: {str(e)}")
        logger.error(traceback.format_exc())
        return (
            "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, intenta de nuevo.",
            {"error": str(e)}
        )