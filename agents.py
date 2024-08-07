import requests
import openai
from typing import Dict, Any

class AgentManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_models = self.get_ollama_models()
        self.openai_models = config.get('openai_models', [])
        self.specialized_assistants = config.get('specialized_assistants', [])

    def get_ollama_models(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return [model['name'] for model in response.json()['models']]
            return []
        except requests.RequestException:
            return []

    def process_with_ollama(self, query: str, model: str = "phi3:latest") -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": query,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json()["response"]
            return "Error en la generación con Ollama."
        except requests.RequestException:
            return "Error de conexión con Ollama."

    def process_with_openai(self, query: str, model: str = "gpt-4o-mini") -> str:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": query}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error al procesar con OpenAI: {str(e)}"

    def process_with_assistant(self, query: str, assistant_id: str) -> str:
        try:
            client = openai.OpenAI()
            thread = client.beta.threads.create()
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            while run.status != 'completed':
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            return messages.data[0].content[0].text.value
        except Exception as e:
            return f"Error al procesar con el asistente especializado: {str(e)}"

def process_query(agent_manager: AgentManager, query: str) -> str:
    # Primero, intentar con modelos locales de Ollama
    if agent_manager.ollama_models:
        return agent_manager.process_with_ollama(query)

    # Luego, buscar un asistente especializado adecuado
    for assistant in agent_manager.specialized_assistants:
        if any(keyword in query.lower() for keyword in assistant['keywords']):
            return agent_manager.process_with_assistant(query, assistant['id'])

    # Si no hay coincidencia, usar modelo de OpenAI por defecto
    return agent_manager.process_with_openai(query)