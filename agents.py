import openai
import requests
import streamlit as st
from typing import Tuple, Dict, Any, Optional, List
from together import Together
import asyncio
import socket
import re
from groq import Groq
import time
import logging
from openai import OpenAI
from anthropic import Anthropic
from mistralai import Mistral
import cohere

class AgentManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_models = self.get_ollama_models()
        self.openai_models = config.get('openai', {}).get('models', [])
        self.together_models = config.get('together', {}).get('models', [])
        self.specialized_assistants = config.get('specialized_assistants', [])
        
        self.clients = {
            'openai': self.init_openai_client(),
            'together': self.init_together_client(),
            'groq': self.init_groq_client(),
            'deepinfra': self.init_deepinfra_client(),
            'anthropic': self.init_anthropic_client(),
            'deepseek': self.init_deepseek_client(),
            'mistral': self.init_mistral_client(),
            'cohere': self.init_cohere_client()
        }

        self.agent_speeds = {}
        self.available_models = self.verify_models()
        self.reliable_models = []
        self.default_local_model = config['ollama']['default_model']

    def init_client(self, client_name: str, client_class, api_key: str, **kwargs):
        try:
            return client_class(api_key=api_key, **kwargs)
        except Exception as e:
            logging.warning(f"Error al inicializar {client_name} client: {str(e)}")
            return None

    def init_openai_client(self):
        return self.init_client('OpenAI', OpenAI, st.secrets.get("OPENAI_API_KEY"))

    def init_together_client(self):
        return self.init_client('Together', Together, st.secrets.get("TOGETHER_API_KEY"))

    def init_groq_client(self):
        client = self.init_client('Groq', Groq, st.secrets.get("GROQ_API_KEY"))
        if client:
            try:
                client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": "Test"}])
                return client
            except Exception as e:
                logging.warning(f"Error al probar Groq client: {str(e)}")
        return None

    def init_deepinfra_client(self):
        return self.init_client('DeepInfra', OpenAI, st.secrets.get("DEEPINFRA_API_KEY"), base_url="https://api.deepinfra.com/v1/openai")

    def init_anthropic_client(self):
        return self.init_client('Anthropic', Anthropic, st.secrets.get("ANTHROPIC_API_KEY"))

    def init_deepseek_client(self):
        return self.init_client('DeepSeek', OpenAI, st.secrets.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")

    def init_mistral_client(self):
        return self.init_client('Mistral', Mistral, st.secrets.get("MISTRAL_API_KEY"))

    def init_cohere_client(self):
        return self.init_client('Cohere', cohere.Client, st.secrets.get("COHERE_API_KEY"))

    def get_ollama_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.config['ollama']['base_url']}/api/tags", timeout=5)
            if response.status_code == 200:
                return [model['name'] for model in response.json()['models']]
            return []
        except requests.RequestException:
            logging.warning("No se pudieron obtener los modelos de Ollama")
            return []

    def verify_models(self) -> Dict[str, List[str]]:
        available_models = {'local': self.ollama_models}
        for agent_type in ['api', 'groq', 'together']:
            available_models[agent_type] = [
                model for model in self.get_available_agents(agent_type)
                if self.test_model_availability(agent_type, model)
            ]
        return available_models

    def test_model_availability(self, agent_type: str, model: str) -> bool:
        if agent_type == 'local':
            return True
        try:
            self.process_query("Test query", agent_type, model)
            return True
        except Exception as e:
            logging.warning(f"Model {agent_type}:{model} is not available: {str(e)}")
            return False

    def update_reliable_models(self, speed_test_results: Dict[str, float]):
        self.reliable_models = sorted(
            [model for model in speed_test_results.keys() if model.split(':')[1] in self.available_models[model.split(':')[0]]],
            key=lambda x: speed_test_results[x]
        )
        self.reliable_models.insert(0, f"local:{self.default_local_model}")

    def get_appropriate_agent(self, query: str, complexity: float) -> Tuple[str, str]:
        scores = self.calculate_agent_scores(query, complexity)
        
        for assistant in self.specialized_assistants:
            if any(keyword.lower() in query.lower() for keyword in assistant['keywords']):
                return 'assistant', assistant['id']
        
        if complexity < self.config['thresholds']['local_complexity']:
            return 'local', self.default_local_model
        
        for model in self.reliable_models:
            agent_type, model_name = model.split(':')
            if self.is_suitable(agent_type, model_name, complexity):
                return agent_type, model_name
        
        return 'local', self.default_local_model

    def calculate_agent_scores(self, query: str, complexity: float) -> Dict[str, float]:
        scores = {}
        
        for assistant in self.specialized_assistants:
            relevance = sum(keyword.lower() in query.lower() for keyword in assistant['keywords'])
            scores[f'assistant:{assistant["id"]}'] = relevance * 2

        scores[f'local:{self.default_local_model}'] = 1 - complexity

        if self.internet_available():
            for model in self.openai_models:
                scores[f'api:{model}'] = complexity
            
            for model in self.together_models:
                scores[f'together:{model}'] = complexity * 0.9
            
            for model in self.config['groq']['models']:
                scores[f'groq:{model}'] = complexity * 0.95

        for agent, speed in self.agent_speeds.items():
            if agent in scores:
                scores[agent] *= (1 / speed)

        return scores

    def is_suitable(self, agent_type: str, model: str, complexity: float) -> bool:
        thresholds = self.config['thresholds']
        if agent_type == 'local' and complexity < thresholds['local_complexity']:
            return True
        if agent_type == 'api' and complexity > thresholds['api_complexity']:
            return True
        if agent_type == 'groq' and thresholds['local_complexity'] <= complexity <= thresholds['api_complexity']:
            return True
        if agent_type == 'together' and complexity < thresholds['local_complexity']:
            return True
        return False

    def process_query(self, query: str, agent_type: str, agent_id: str) -> str:
        start_time = time.time()
        try:
            if agent_type == 'assistant':
                response = self.process_with_assistant(agent_id, query)
            else:
                process_method = getattr(self, f"process_with_{agent_type}", None)
                if process_method:
                    response = process_method(query, agent_id)
                else:
                    response = f"No se pudo procesar la consulta con el agente seleccionado: {agent_type}"
            
            if not response or response.startswith("Error") or response.startswith("No se pudo procesar"):
                fallback_agent, fallback_model = self.get_fallback_agent()
                response = self.process_query(query, fallback_agent, fallback_model)
                
        except Exception as e:
            logging.error(f"Error processing query with {agent_type}:{agent_id}: {str(e)}")
            fallback_agent, fallback_model = self.get_fallback_agent()
            response = self.process_query(query, fallback_agent, fallback_model)

        processing_time = time.time() - start_time
        self.agent_speeds[f"{agent_type}:{agent_id}"] = processing_time
        return response

    def get_fallback_agent(self) -> Tuple[str, str]:
        if self.clients['openai']:
            return 'api', self.config['openai']['default_model']
        elif self.clients['groq']:
            return 'groq', self.config['groq']['default_model']
        elif self.clients['together']:
            return 'together', self.config['together']['default_model']
        else:
            return 'local', self.default_local_model

    def get_next_reliable_model(self, current_model: str) -> Optional[str]:
        try:
            current_index = self.reliable_models.index(current_model)
            if current_index < len(self.reliable_models) - 1:
                return self.reliable_models[current_index + 1]
        except ValueError:
            pass
        return None

    def process_with_deepinfra(self, query: str, model: str) -> str:
        return self._process_with_openai_like_client(self.clients['deepinfra'], query, model)

    def process_with_anthropic(self, query: str, model: str) -> str:
        message = self.clients['anthropic'].messages.create(
            model=model,
            max_tokens=self.config['general']['max_tokens'],
            temperature=self.config['general']['temperature'],
            messages=[{"role": "user", "content": query}]
        )
        return message.content

    def process_with_deepseek(self, query: str, model: str) -> str:
        return self._process_with_openai_like_client(self.clients['deepseek'], query, model)

    def process_with_mistral(self, query: str, model: str) -> str:
        chat_response = self.clients['mistral'].chat.complete(
            model=model,
            messages=[{"role": "user", "content": query}]
        )
        return chat_response.choices[0].message.content

    def process_with_cohere(self, query: str, model: str) -> str:
        response = self.clients['cohere'].chat(
            model=model,
            message=query,
            temperature=self.config['general']['temperature'],
        )
        return response.text

    def process_with_local_model(self, query: str, model: Optional[str] = None) -> str:
        model = model or self.default_local_model
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

    def process_with_api(self, query: str, model: Optional[str] = None) -> str:
        model = model or self.config['openai']['default_model']
        return self._process_with_openai_like_client(self.clients['openai'], query, model)

    def process_with_assistant(self, assistant_id: str, query: str) -> str:
        try:
            thread = self.clients['openai'].beta.threads.create()
            self.clients['openai'].beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            run = self.clients['openai'].beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            while run.status != 'completed':
                time.sleep(1)
                run = self.clients['openai'].beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            messages = self.clients['openai'].beta.threads.messages.list(thread_id=thread.id)
            return messages.data[0].content[0].text.value
        except Exception as e:
            logging.error(f"Error al procesar con el asistente especializado: {str(e)}")
            return None

    def process_with_together(self, query: str, model: str) -> str:
        try:
            response = self.clients['together'].chat.completions.create(
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
            max_tokens = min(self.config['groq']['max_tokens'], 8000)
            response = self.clients['groq'].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                temperature=self.config['groq']['temperature'],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error al procesar con Groq API: {str(e)}"

    def _process_with_openai_like_client(self, client, query: str, model: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                temperature=self.config['general']['temperature'],
                max_tokens=self.config['general']['max_tokens'],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error al procesar con API: {str(e)}"

    def run_moa(self, query: str, web_context: str = "") -> str:
        results = asyncio.run(self.run_multiple_models(query))
        aggregated_results = "\n".join([f"{i+1}. {result}" for i, result in enumerate(results)])

        full_context = f"Web Context:\n{web_context}\n\nModel Responses:\n{aggregated_results}"

        aggregator_response = self.clients['together'].chat.completions.create(
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

    def get_available_agents(self, agent_type: str = None) -> List[str]:
        if agent_type:
            return {
                'api': self.openai_models,
                'groq': self.config['groq']['models'],
                'together': self.together_models,
                'local': self.ollama_models
            }.get(agent_type, [])
        else:
            all_agents = []
            for agent_type in ['api', 'groq', 'together', 'local']:
                all_agents.extend(self.get_available_agents(agent_type))
            return all_agents

    def get_agent_priority(self):
        priority = self.config.get('processing_priority', [])
        agents = []
        for agent_type in priority:
            if agent_type == 'specialized_assistants':
                agents.extend([(agent_type, assistant['id']) for assistant in self.specialized_assistants])
            elif agent_type in ['api', 'groq', 'together', 'local']:
                agents.extend([(agent_type, model) for model in self.get_available_agents(agent_type)])
        return agents

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
        logging.error(f"Error al evaluar la complejidad de la consulta: {str(e)}")
        return 0.5