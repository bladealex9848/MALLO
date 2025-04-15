import openai
import numpy as np
from typing import List, Dict, Any
import yaml
from sentence_transformers import SentenceTransformer, util
import requests
import logging
from openai import OpenAI
from anthropic import Anthropic
from mistralai import Mistral
import cohere
import streamlit as st

class MALLOEnhancer:
    def __init__(self, student_model, config_path: str, max_iterations: int = 3):
        self.student_model = student_model
        self.config = self.load_config(config_path)
        self.professor_models = self.get_professor_models()
        self.max_iterations = max_iterations
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.clients = self.initialize_clients()

    def load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def get_professor_models(self) -> List[str]:
        models = []
        for api in ['openai', 'anthropic', 'together', 'cohere', 'groq', 'deepinfra', 'deepseek', 'mistral', 'openrouter']:
            if api in self.config and 'models' in self.config[api]:
                models.extend(self.config[api]['models'])
        return models

    def initialize_clients(self):
        return {
            'openai': OpenAI(api_key=st.secrets.get("OPENAI_API_KEY")),
            'anthropic': Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY")),
            'mistral': Mistral(api_key=st.secrets.get("MISTRAL_API_KEY")),
            'cohere': cohere.Client(api_key=st.secrets.get("COHERE_API_KEY")),
            'groq': OpenAI(api_key=st.secrets.get("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"),
            'deepinfra': OpenAI(api_key=st.secrets.get("DEEPINFRA_API_KEY"), base_url="https://api.deepinfra.com/v1/openai"),
            'deepseek': OpenAI(api_key=st.secrets.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1"),
        }

    def iterative_reflection(self, instruction: str, response: str) -> Dict[str, str]:
        mejor_instruccion = instruction
        mejor_respuesta = response
        mejor_puntuacion = self.evaluate_pair(instruction, response)

        for _ in range(self.max_iterations):
            candidatos = []
            for profesor in self.professor_models:
                nueva_instruccion = self.reflect_on_instruction(profesor, mejor_instruccion, mejor_respuesta)
                nueva_respuesta = self.reflect_on_response(profesor, nueva_instruccion, mejor_respuesta)
                candidatos.append((nueva_instruccion, nueva_respuesta))

            for candidato_instruccion, candidato_respuesta in candidatos:
                puntuacion = self.evaluate_pair(candidato_instruccion, candidato_respuesta)
                if puntuacion > mejor_puntuacion:
                    mejor_instruccion = candidato_instruccion
                    mejor_respuesta = candidato_respuesta
                    mejor_puntuacion = puntuacion

        return {"instruction": mejor_instruccion, "response": mejor_respuesta}

    def reflect_on_instruction(self, profesor: str, instruccion: str, respuesta: str) -> str:
        prompt = f"""
        Analiza y mejora el siguiente par instrucción-respuesta:

        Instrucción: {instruccion}

        Respuesta: {respuesta}

        Por favor, proporciona una versión mejorada de la instrucción que sea más clara, específica y desafiante. 
        La nueva instrucción debe ser independiente y no requerir conocimiento de la instrucción original.

        Instrucción Mejorada:
        """
        
        return self.get_model_response(profesor, prompt)

    def reflect_on_response(self, profesor: str, instruccion: str, respuesta: str) -> str:
        prompt = f"""
        Dada la siguiente instrucción y su respuesta actual, proporciona una respuesta mejorada:

        Instrucción: {instruccion}

        Respuesta Actual: {respuesta}

        Por favor, genera una respuesta más completa, precisa y detallada para la instrucción.

        Respuesta Mejorada:
        """
        
        return self.get_model_response(profesor, prompt)

    def get_model_response(self, model: str, prompt: str) -> str:
        if 'gpt' in model:
            return self.get_openai_response(model, prompt)
        elif 'claude' in model:
            return self.get_anthropic_response(model, prompt)
        elif model in self.config['together']['models']:
            return self.get_together_response(model, prompt)
        elif model in self.config['cohere']['models']:
            return self.get_cohere_response(model, prompt)
        elif model in self.config['groq']['models']:
            return self.get_groq_response(model, prompt)
        elif model in self.config['deepinfra']['models']:
            return self.get_deepinfra_response(model, prompt)
        elif model in self.config['deepseek']['models']:
            return self.get_deepseek_response(model, prompt)
        elif model in self.config['mistral']['models']:
            return self.get_mistral_response(model, prompt)
        elif model in self.config['openrouter']['models']:
            return self.get_openrouter_response(model, prompt)
        else:
            raise ValueError(f"Modelo no soportado: {model}")

    def get_openai_response(self, model: str, prompt: str) -> str:
        try:
            response = self.clients['openai'].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error al obtener respuesta de OpenAI: {str(e)}")
            return ""

    def get_anthropic_response(self, model: str, prompt: str) -> str:
        try:
            response = self.clients['anthropic'].messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Error al obtener respuesta de Anthropic: {str(e)}")
            return ""

    def get_together_response(self, model: str, prompt: str) -> str:
        try:
            response = requests.post(
                "https://api.together.xyz/inference",
                headers={
                    "Authorization": f"Bearer {self.config['together']['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.7,
                }
            )
            response.raise_for_status()
            return response.json()['output']['choices'][0]['text'].strip()
        except Exception as e:
            logging.error(f"Error al obtener respuesta de Together AI: {str(e)}")
            return ""

    def get_cohere_response(self, model: str, prompt: str) -> str:
        try:
            response = self.clients['cohere'].generate(
                model=model,
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
            )
            return response.generations[0].text.strip()
        except Exception as e:
            logging.error(f"Error al obtener respuesta de Cohere: {str(e)}")
            return ""

    def get_groq_response(self, model: str, prompt: str) -> str:
        try:
            response = self.clients['groq'].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error al obtener respuesta de Groq: {str(e)}")
            return ""

    def get_deepinfra_response(self, model: str, prompt: str) -> str:
        try:
            response = self.clients['deepinfra'].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error al obtener respuesta de DeepInfra: {str(e)}")
            return ""

    def get_deepseek_response(self, model: str, prompt: str) -> str:
        try:
            response = self.clients['deepseek'].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error al obtener respuesta de DeepSeek: {str(e)}")
            return ""

    def get_mistral_response(self, model: str, prompt: str) -> str:
        try:
            response = self.clients['mistral'].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error al obtener respuesta de Mistral: {str(e)}")
            return ""

    def get_openrouter_response(self, model: str, prompt: str) -> str:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config['openrouter']['api_key']}",
                    "HTTP-Referer": "https://marduk.pro",
                    "X-Title": "MALLO",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.7,
                }
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logging.error(f"Error al obtener respuesta de OpenRouter: {str(e)}")
            return ""

    def evaluate_pair(self, instruction: str, response: str) -> float:
        ifd_score = self.calculate_ifd(instruction, response)
        r_ifd_score = self.calculate_r_ifd(instruction, response)
        coherence_score = self.calculate_coherence(instruction, response)
        return np.mean([ifd_score, 1 / r_ifd_score, coherence_score])

    def calculate_ifd(self, instruction: str, response: str) -> float:
        # Implementar cálculo de IFD usando student_model
        # Esta es una implementación de marcador de posición
        return np.random.random()

    def calculate_r_ifd(self, instruction: str, response: str) -> float:
        # Implementar cálculo de r-IFD usando student_model
        # Esta es una implementación de marcador de posición
        return np.random.random()

    def calculate_coherence(self, instruction: str, response: str) -> float:
        instruction_embedding = self.sentence_model.encode(instruction, convert_to_tensor=True)
        response_embedding = self.sentence_model.encode(response, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(instruction_embedding, response_embedding)[0][0])

def adapt_criteria(performance_history: List[float]) -> Dict[str, Any]:
    # Ajustar criterios basados en el rendimiento reciente del modelo
    # Esta es una implementación de marcador de posición
    return {
        "umbral_complejidad": np.mean(performance_history),
        "umbral_coherencia": np.median(performance_history),
    }