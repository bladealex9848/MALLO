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
import requests
import json

from utilities import (
    log_error
)

class AgentManager:
    def __init__(self, config: Dict[str, Any]):   
        self.config = config
        self.clients = {}  # Inicializa esto con tus clientes de API
        self.agent_speeds = {}
        self.ollama_models = self.get_ollama_models()
        self.openai_models = config.get('openai', {}).get('models', [])
        self.together_models = config.get('together', {}).get('models', [])
        self.specialized_assistants = config.get('specialized_assistants', [])
        self.moa_threshold = config.get('thresholds', {}).get('moa_complexity', 0.7)
        self.agent_speeds = {}
        self.available_models = self.verify_models()
        self.reliable_models = []
        self.default_local_model = config['ollama'].get('default_model', 'phi3.5:latest')
        self.processing_priority = config.get('processing_priority', [])
        self.default_agent = ('openrouter', config['deepinfra']['default_model'])
        self.backup_default_agent = ('deepinfra', config['openrouter']['default_model'])
        self.meta_analysis_model = config['evaluation_models']['meta_analysis']['model']
        self.meta_analysis_api = config['evaluation_models']['meta_analysis']['api']
                        
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

    def get_available_models(self, agent_type: str = None) -> List[str]:
        if agent_type:
            return {
                'api': self.openai_models,
                'groq': self.config['groq']['models'],
                'together': self.together_models,
                'local': self.ollama_models,
                'openrouter': self.config['openrouter']['models'],
                'deepinfra': self.config['deepinfra']['models'],
                'anthropic': self.config['anthropic']['models'],
                'deepseek': self.config['deepseek']['models'],
                'mistral': self.config['mistral']['models'],
                'cohere': self.config['cohere']['models']                  
            }.get(agent_type, [])
        else:
            all_agents = []
            for agent_type in ['api', 'groq', 'together', 'local', 'openrouter', 'deepinfra', 'anthropic', 'deepseek', 'mistral', 'cohere']:
                all_agents.extend(self.get_available_models(agent_type))
            return all_agents

    def get_prioritized_agents(self, query: str, complexity: float) -> List[Tuple[str, str]]:
        prioritized_agents = []
        for agent_type in self.processing_priority:
            if agent_type == 'specialized_assistants':
                assistant_id = self.find_suitable_assistant(query)
                if assistant_id:
                    prioritized_agents.append(('assistant', assistant_id))
            elif agent_type == 'moa':
                if complexity > self.config['thresholds']['moa_complexity']:
                    prioritized_agents.append(('moa', 'moa'))
            elif agent_type == 'openrouter':
                models = self.get_available_models('openrouter')
                if models:
                    prioritized_agents.append(('openrouter', models[0]))
            elif agent_type == 'deepinfra':
                models = self.get_available_models('deepinfra')
                if models:
                    prioritized_agents.append(('deepinfra', models[0]))
            elif agent_type in ['groq', 'together', 'openai', 'anthropic', 'deepseek', 'cohere', 'ollama', 'mistral']:
                models = self.get_available_models(agent_type)
                if models:
                    prioritized_agents.append((agent_type, models[0]))
        
        # Añadir agentes por defecto al final de la lista si no están ya incluidos
        default_agents = [
            ('openrouter', self.config['openrouter']['default_model']),
            ('deepinfra', self.config['deepinfra']['default_model'])
        ]
        for agent, model in default_agents:
            if not any(a[0] == agent for a in prioritized_agents):
                prioritized_agents.append((agent, model))
        
        return prioritized_agents

    def process_query_with_fallback(self, query: str, prioritized_agents: List[Tuple[str, str]]) -> Tuple[str, Dict[str, Any]]:
        for agent_type, agent_id in prioritized_agents:
            try:
                response = self.process_query(query, agent_type, agent_id)
                return response, {"agent": agent_type, "model": agent_id}
            except Exception as e:
                logging.error(f"Error processing query with {agent_type}:{agent_id}: {str(e)}")
        
        # Si todos los agentes fallan, usar el modelo local por defecto
        try:
            response = self.process_with_local_model(query, self.default_local_model)
            return response, {"agent": "local", "model": self.default_local_model}
        except Exception as e:
            logging.error(f"Error processing query with local model: {str(e)}")
            raise ValueError("No se pudo procesar la consulta con ningún agente disponible")    

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
    
    def init_openrouter_client(self):
        return st.secrets.get("OPENROUTER_API_KEY")    
    
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
        if agent_type == 'deepinfra' and complexity < thresholds['api_complexity']:
            return True
        if agent_type == 'openrouter' and complexity < thresholds['local_complexity']:
            return True
        if agent_type == 'deepseek' and complexity < thresholds['api_complexity']:
            return True
        if agent_type == 'local' and complexity < thresholds['local_complexity']:
            return True
        if agent_type == 'api' and complexity > thresholds['api_complexity']:
            return True
        if agent_type == 'groq' and thresholds['local_complexity'] <= complexity <= thresholds['api_complexity']:
            return True
        if agent_type == 'together' and complexity < thresholds['local_complexity']:
            return True 
        if agent_type == 'anthropic' and complexity < thresholds['api_complexity']:
            return True        
        if agent_type == 'cohere' and complexity < thresholds['api_complexity']:
            return True
        if agent_type == 'mistral' and complexity < thresholds['api_complexity']:
            return True
        return False

    def process_query(self, query: str, agent_type: str, agent_id: str, fallback_attempts: int = 0) -> str:
        start_time = time.time()
        max_fallback_attempts = 3  # Límite de intentos de fallback para evitar bucles infinitos

        try:
            # Procesar la consulta con el agente seleccionado
            if agent_type == 'assistant':
                response = self.process_with_assistant(agent_id, query)
            else:
                process_method = getattr(self, f"process_with_{agent_type}", None)
                if process_method:
                    response = process_method(query, agent_id)
                else:
                    raise ValueError(f"No se pudo procesar la consulta con el agente seleccionado: {agent_type}")

            # Verificar y manejar la respuesta
            if isinstance(response, dict):
                if 'error' in response:
                    raise ValueError(f"Error en la respuesta: {response['error']}")
                elif 'content' in response:
                    response = response['content']
                else:
                    raise ValueError(f"Respuesta inesperada del agente {agent_type}:{agent_id}")

            # Validar la respuesta
            if not response or (isinstance(response, str) and (response.strip() == "" or response.startswith("Error") or response.startswith("No se pudo procesar"))):
                raise ValueError(f"Respuesta inválida del agente {agent_type}:{agent_id}")

            # Registrar el tiempo de procesamiento
            processing_time = time.time() - start_time
            self.agent_speeds[f"{agent_type}:{agent_id}"] = processing_time
            logging.info(f"Query processed by {agent_type}:{agent_id} in {processing_time:.2f} seconds")

            return response

        except Exception as e:
            logging.error(f"Error processing query with {agent_type}:{agent_id}: {str(e)}")
            
            # Intentar con agente de respaldo si no se ha alcanzado el límite de intentos
            if fallback_attempts < max_fallback_attempts:
                fallback_agent, fallback_model = self.get_fallback_agent()
                if fallback_agent != agent_type or fallback_model != agent_id:
                    logging.info(f"Attempting fallback with {fallback_agent}:{fallback_model}")
                    return self.process_query(query, fallback_agent, fallback_model, fallback_attempts + 1)
                else:
                    logging.error("Fallback agent is the same as the failed agent. Aborting fallback.")
            else:
                logging.error(f"Reached maximum fallback attempts ({max_fallback_attempts}). Unable to process query.")

            # Si todos los intentos fallan, lanzar una excepción
            raise ValueError(f"Failed to process query after {fallback_attempts} fallback attempts")

    def should_use_moa(self, query: str, complexity: float) -> bool:
        return complexity > self.moa_threshold

    def find_suitable_assistant(self, query: str) -> Optional[str]:
        for assistant in self.specialized_assistants:
            if any(keyword.lower() in query.lower() for keyword in assistant['keywords']):
                return assistant['id']
        return None

    def get_fallback_agent(self) -> Tuple[str, str]:
        if self.clients.get('openrouter'):
            return 'openrouter', self.config['openrouter']['default_model']
        elif self.clients.get('openai'):
            return 'api', self.config['openai']['default_model']
        elif self.clients.get('groq'):
            return 'groq', self.config['groq']['default_model']
        elif self.clients.get('together'):
            return 'together', self.config['together']['default_model']
        else:
            return 'local', self.config['ollama']['default_model']

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

    def process_with_local_model(self, query: str, model: str) -> str:
        try:
            url = f"{self.config['ollama']['base_url']}/api/generate"
            payload = {
                "model": model,
                "prompt": query,
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            log_error(f"Error al procesar con modelo local: {str(e)}")
            return f"Error al procesar con modelo local: {str(e)}"

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

    def process_with_openrouter(self, query: str, model: str) -> str:
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                "HTTP-Referer": st.secrets.get("YOUR_SITE_URL", "http://localhost:8501"),
                "X-Title": "MALLO",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": query}]
            }
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                return {"error": "Respuesta inesperada de OpenRouter"}
        
        except Exception as e:
            logging.error(f"Error al procesar con OpenRouter API: {str(e)}")
            return {"error": f"Error al procesar con OpenRouter API: {str(e)}"}

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
    
    def meta_analysis(self, query: str, responses: List[Dict[str, Any]], initial_evaluation: str, final_evaluation: str) -> str:
        prompt = f"""
        Analiza las siguientes respuestas a la pregunta: "{query}"

        Respuestas de los agentes:
        {json.dumps(responses, indent=2)}

        Evaluación inicial:
        {initial_evaluation}

        Evaluación final:
        {final_evaluation}

        Tu tarea es:
        1. Sintetizar una respuesta final que incorpore los mejores elementos de todas las respuestas.
        2. Corregir cualquier error identificado en la evaluación final.
        3. Asegurarte de que la respuesta aborde todos los puntos clave mencionados en la evaluación inicial.
        4. Proporcionar una explicación clara y concisa del razonamiento detrás de la respuesta.
        5. Mantener la respuesta enfocada y relevante a la pregunta original.

        Por favor, proporciona la respuesta sintetizada en un formato claro y estructurado.
        """

        meta_config = self.config['evaluation_models']['meta_analysis']

        for attempt in range(3):  # Intentar hasta 3 veces
            try:
                if attempt == 0:
                    response = self.process_query(prompt, meta_config['api'], meta_config['model'])
                elif attempt == 1:
                    response = self.process_query(prompt, meta_config['backup_api'], meta_config['backup_model'])
                else:
                    response = self.process_query(prompt, meta_config['backup_api2'], meta_config['backup_model2'])
                
                if response and not response.startswith("Error"):
                    return response
            except Exception as e:
                logging.error(f"Error en meta-análisis con {'modelo principal' if attempt == 0 else 'modelo de respaldo'}: {str(e)}")
        
        return "No se pudo realizar el meta-análisis debido a múltiples errores en los modelos configurados."

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
                'local': self.ollama_models,
                'openrouter': self.config['openrouter']['models'], # Añadido OpenRouter
                'deepinfra': self.config['deepinfra']['models'],
                'anthropic': self.config['anthropic']['models'],
                'deepseek': self.config['deepseek']['models'],
                'mistral': self.config['mistral']['models'],
                'cohere': self.config['cohere']['models']                  
            }.get(agent_type, [])
        else:
            all_agents = []
            for agent_type in ['api', 'groq', 'together', 'local', 'openrouter']:  # Añadido OpenRouter
                all_agents.extend(self.get_available_agents(agent_type))
            return all_agents

    def get_agent_priority(self):
        priority = self.config.get('processing_priority', [])
        agents = []
        for agent_type in priority:
            if agent_type == 'specialized_assistants':
                agents.extend([(agent_type, assistant['id']) for assistant in self.specialized_assistants])
            elif agent_type in ['api', 'groq', 'together', 'local', 'openrouter']:  # Añadido OpenRouter
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