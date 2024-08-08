import re
from typing import List, Dict, Any
from PIL import Image
import pytesseract
import PyPDF2
import docx
from duckduckgo_search import DDGS
import streamlit as st
from openai import OpenAI

# Obtener las claves API de los secrets de Streamlit
API_KEY = st.secrets.get("OPENAI_API_KEY")
if not API_KEY:
    st.error("No se encontró la clave API de OpenAI en los secrets.")
    st.stop()

client = OpenAI(api_key=API_KEY)

def clean_text(text: str) -> str:
    """
    Limpia el texto de caracteres especiales y espacios extra.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """
    Divide el texto en tokens (palabras).
    """
    return text.lower().split()

def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """
    Extrae las palabras clave más frecuentes del texto.
    """
    words = tokenize(text)
    word_freq = {}
    for word in words:
        if len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    return sorted(word_freq, key=word_freq.get, reverse=True)[:num_keywords]

def summarize_text(text: str, max_length: int = 100) -> str:
    """
    Genera un resumen simple del texto.
    """
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

def buscar_en_duckduckgo(consulta: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Realiza una búsqueda web utilizando DuckDuckGo.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(consulta, max_results=num_results))
        return results
    except Exception as e:
        st.error(f"Error al buscar en DuckDuckGo: {str(e)}")
        return []

def charla_con_openai(consulta: str, contexto: str, historial: list, modelo: str = "gpt-4o-mini") -> str:
    try:
        messages = [
            {"role": "system", "content": "Eres un asistente experto que interpreta y proporciona respuestas basadas en la información proporcionada y el historial de la conversación. Ofrece respuestas concisas, relevantes y bien estructuradas."},
        ]
        # Agregar el historial de la conversación
        messages.extend(historial)
        # Agregar el nuevo contexto y la consulta actual
        messages.append({"role": "user", "content": f"Nuevo contexto:\n{contexto}\n\nConsulta actual: {consulta}"})
        
        response = client.chat.completions.create(model=modelo, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al comunicarse con OpenAI: {str(e)}")
        return "Lo siento, ocurrió un error al procesar tu consulta."


def analyze_image(image_path: str) -> str:
    """
    Analiza una imagen y extrae el texto utilizando OCR.
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return clean_text(text)

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extrae texto de un archivo PDF cargado por Streamlit.
    """
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return clean_text(text)

def extract_text_from_docx(docx_file) -> str:
    """
    Extrae texto de un archivo Word (.docx) cargado por Streamlit.
    """
    doc = docx.Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return clean_text(text)

def detect_language(text: str) -> str:
    """
    Detecta el idioma del texto (implementación simple).
    """
    common_words = {
        'en': ['the', 'be', 'to', 'of', 'and', 'in', 'that', 'have'],
        'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'ser'],
        'fr': ['le', 'la', 'de', 'et', 'un', 'une', 'que', 'en']
    }
    
    words = set(tokenize(text))
    scores = {lang: sum(1 for word in lang_words if word in words) 
              for lang, lang_words in common_words.items()}
    return max(scores, key=scores.get)