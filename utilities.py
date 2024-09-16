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
from collections import Counter
import yaml

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