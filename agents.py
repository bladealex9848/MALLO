import openai
import requests
import streamlit as st
from typing import Tuple, Dict, Any, Optional, List
from together import Together
import asyncio
import socket
import re

class AgentManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_models = self.get_ollama_models()
        self.openai_models = config.get('openai', {}).get('models', [])
        self.specialized_assistants = config.get('specialized_assistants', [])
        self.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.together_client = Together(api_key=st.secrets["TOGETHER_API_KEY"])
        self.together_models = config.get('together', {}).get('models', [])

    def get_ollama_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.config['ollama']['base_url']}/api/tags", timeout=5)
            if response.status_code == 200:
                return [model['name'] for model in response.json()['models']]
            return []
        except requests.RequestException:
            return []

    def get_local_model_name(self) -> Optional[str]:
        if self.ollama_models:
            return self.config['ollama']['default_model']
        return None

    def get_appropriate_agent(self, query: str, complexity: float) -> Tuple[str, str]:
        for assistant in self.specialized_assistants:
            if any(keyword.lower() in query.lower() for keyword in assistant['keywords']):
                return 'assistant', assistant['id']
        
        if complexity > self.config['thresholds']['moa_complexity'] and self.internet_available():
            return 'moa', 'multiple'
        
        if self.ollama_models and (complexity <= self.config['thresholds']['local_complexity'] or not self.internet_available()):
            return 'local', self.get_local_model_name()
        
        if self.internet_available():
            return 'api', self.config['openai']['default_model']
        
        return ('local', self.get_local_model_name()) if self.ollama_models else ('together', self.together_models[0])

    def process_with_local_model(self, query: str, model: Optional[str] = None) -> str:
        if model is None:
            model = self.get_local_model_name()
        
        url = f"{self.config['ollama']['base_url']}/api/generate"
        payload = {
            "model": model,
            "prompt": query,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["response"]
            return f"Error en la generación con modelo local: {response.status_code}"
        except requests.RequestException as e:
            return f"Error de conexión con el modelo local: {str(e)}"

    def process_with_api(self, query: str, model: Optional[str] = None) -> Tuple[str, int]:
        if model is None:
            model = self.config['openai']['default_model']
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                temperature=self.config['general']['temperature'],
                max_tokens=self.config['general']['max_tokens'],
            )
            return response.choices[0].message.content, response.usage.total_tokens
        except Exception as e:
            return f"Error al procesar con API: {str(e)}", 0

    def process_with_assistant(self, assistant_id: str, query: str) -> str:
        try:
            thread = self.openai_client.beta.threads.create()
            self.openai_client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            run = self.openai_client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            while run.status != 'completed':
                run = self.openai_client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            messages = self.openai_client.beta.threads.messages.list(thread_id=thread.id)
            return messages.data[0].content[0].text.value
        except Exception as e:
            return f"Error al procesar con el asistente especializado: {str(e)}"

    def process_with_together(self, query: str, model: str) -> str:
        try:
            response = self.together_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                temperature=self.config['together']['temperature'],
                max_tokens=self.config['together']['max_tokens'],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error al procesar con Together API: {str(e)}"

    async def run_multiple_models(self, query: str) -> List[str]:
        async def run_model(model):
            return self.process_with_together(query, model)

        tasks = [run_model(model) for model in self.together_models]
        results = await asyncio.gather(*tasks)
        return results

    def run_moa(self, query: str, web_context: str = "") -> str:
        results = asyncio.run(self.run_multiple_models(query))
        aggregated_results = "\n".join([f"{i+1}. {result}" for i, result in enumerate(results)])
        
        full_context = f"Web Context:\n{web_context}\n\nModel Responses:\n{aggregated_results}"
        
        aggregator_response = self.together_client.chat.completions.create(
            model=self.config['moa']['aggregator_model'],
            messages=[
                {"role": "system", "content": self.config['moa']['aggregator_system_prompt']},
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{full_context}"}
            ],
            temperature=self.config['general']['temperature'],
            max_tokens=self.config['general']['max_tokens'],
        )
        return aggregator_response.choices[0].message.content

    def internet_available(self, host="8.8.8.8", port=53, timeout=3):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False

def evaluate_query_complexity(query: str) -> float:
    try:
        # Limpia la consulta de puntuación y la convierte a minúsculas
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        
        # Divide la consulta en palabras
        words = clean_query.split()
        
        # Cuenta las palabras únicas
        unique_words = set(words)
        
        # Calcula la complejidad basada en el número de palabras únicas y la longitud total
        word_complexity = min(len(unique_words) / 10, 1.0)
        length_complexity = min(len(words) / 20, 1.0)
        
        # Combina las dos métricas
        complexity = (word_complexity + length_complexity) / 2
        
        return complexity
    except Exception as e:
        st.error(f"Error al evaluar la complejidad de la consulta: {str(e)}")
        # Devuelve una complejidad media por defecto si hay un error
        return 0.5

def process_query(agent_manager: AgentManager, query: str, web_context: str = "") -> Tuple[str, str, Any]:
    complexity = evaluate_query_complexity(query)
    agent_type, agent_id = agent_manager.get_appropriate_agent(query, complexity)
    
    if agent_type == 'moa':
        response = agent_manager.run_moa(query, web_context)
        return agent_type, 'multiple', response
    elif agent_type == 'assistant':
        response = agent_manager.process_with_assistant(agent_id, query)
        return agent_type, agent_id, response
    elif agent_type == 'local':
        response = agent_manager.process_with_local_model(query, agent_id)
        return agent_type, agent_id, response
    elif agent_type == 'together':
        response = agent_manager.process_with_together(query, agent_id)
        return agent_type, agent_id, response
    elif agent_type == 'api':
        response, tokens = agent_manager.process_with_api(query, agent_id)
        return agent_type, agent_id, (response, tokens)
    else:
        return 'unknown', None, "No se pudo determinar un agente apropiado para esta consulta."