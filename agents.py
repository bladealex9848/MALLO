import openai
import requests
import streamlit as st
from typing import Tuple, Dict, Any, Optional, List
from together import Together
import asyncio
import socket
import re
from groq import Groq

class AgentManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_models = self.get_ollama_models()
        self.openai_models = config.get('openai', {}).get('models', [])
        self.specialized_assistants = config.get('specialized_assistants', [])
        self.openai_client = self.init_openai_client()
        self.together_client = self.init_together_client()
        self.groq_client = self.init_groq_client()
        self.together_models = config.get('together', {}).get('models', [])

    def init_openai_client(self):
        try:
            return openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        except Exception as e:
            st.warning(f"Error al inicializar OpenAI client: {str(e)}")
            return None

    def init_together_client(self):
        try:
            return Together(api_key=st.secrets.get("TOGETHER_API_KEY"))
        except Exception as e:
            st.warning(f"Error al inicializar Together client: {str(e)}")
            return None

    def init_groq_client(self):
        try:
            client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))
            # Cambiamos el modelo de prueba a uno que sabemos que existe
            client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": "Test"}])
            return client
        except Exception as e:
            st.warning(f"Error al inicializar Groq client: {str(e)}")
            return None

    def get_ollama_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.config['ollama']['base_url']}/api/tags", timeout=5)
            if response.status_code == 200:
                return [model['name'] for model in response.json()['models']]
            return []
        except requests.RequestException:
            st.warning("No se pudieron obtener los modelos de Ollama")
            return []

    def get_appropriate_agent(self, query: str, complexity: float) -> Tuple[str, str]:
        for assistant in self.specialized_assistants:
            if any(keyword.lower() in query.lower() for keyword in assistant['keywords']):
                return 'assistant', assistant['id']
        
        if complexity > self.config['thresholds']['moa_complexity'] and self.internet_available():
            return 'moa', 'multiple'
        
        if self.ollama_models and (complexity <= self.config['thresholds']['local_complexity'] or not self.internet_available()):
            return 'local', self.config['ollama']['default_model']
        
        if self.internet_available():
            if self.openai_client:
                return 'api', self.config['openai']['default_model']
            elif self.groq_client:
                return 'groq', self.config['groq']['default_model']
            elif self.together_client:
                return 'together', self.config['together']['default_model']
        
        return 'fallback', 'echo'

    def process_query(self, query: str, agent_type: str, agent_id: str) -> str:
        try:
            if agent_type == 'assistant':
                return self.process_with_assistant(agent_id, query)
            elif agent_type == 'api':
                response, _ = self.process_with_api(query, agent_id)
                return response
            elif agent_type == 'groq':
                return self.process_with_groq(query, agent_id)
            elif agent_type == 'local':
                return self.process_with_local_model(query, agent_id)
            elif agent_type == 'together':
                return self.process_with_together(query, agent_id)
            elif agent_type == 'moa':
                return self.run_moa(query)
            else:
                return f"No se pudo procesar la consulta con el agente seleccionado: {agent_type}"
        except Exception as e:
            st.error(f"Error al procesar la consulta: {str(e)}")
            return "Ocurrió un error al procesar la consulta. Por favor, intenta de nuevo."

    def process_with_local_model(self, query: str, model: Optional[str] = None) -> str:
        if model is None:
            model = self.config['ollama']['default_model']
        
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

    def process_with_groq(self, query: str, model: str) -> str:
        try:
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                temperature=self.config['groq']['temperature'],
                max_tokens=self.config['groq']['max_tokens'],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error al procesar con Groq API: {str(e)}"

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

    async def run_multiple_models(self, query: str) -> List[str]:
        async def run_model(model):
            return self.process_with_together(query, model)

        tasks = [run_model(model) for model in self.together_models]
        results = await asyncio.gather(*tasks)
        return results

    def internet_available(self, host="8.8.8.8", port=53, timeout=3):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False

def evaluate_query_complexity(query: str) -> float:
    try:
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        words = clean_query.split()
        unique_words = set(words)
        word_complexity = min(len(unique_words) / 10, 1.0)
        length_complexity = min(len(words) / 20, 1.0)
        complexity = (word_complexity + length_complexity) / 2
        return complexity
    except Exception as e:
        st.error(f"Error al evaluar la complejidad de la consulta: {str(e)}")
        return 0.5