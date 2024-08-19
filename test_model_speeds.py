import json
import time
import asyncio
import aiohttp
import yaml
import streamlit as st
from openai import OpenAI
import importlib
import sys
import requests

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()

def safe_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        print(f"Warning: Could not import {module_name}. Error: {e}")
        return None

anthropic = safe_import('anthropic')
mistralai = safe_import('mistralai')
cohere = safe_import('cohere')
together = safe_import('together')

async def get_groq_models():
    api_key = st.secrets["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return [model['id'] for model in data['data']]
            else:
                print(f"Error fetching Groq models: {response.status}")
                return []

async def test_groq_model(model):
    api_key = st.secrets["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                await response.json()
        end_time = time.time()
        return model, end_time - start_time, "success"
    except Exception as e:
        return model, None, str(e)

async def test_openai_model(model):
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        start_time = time.time()
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        end_time = time.time()
        return model, end_time - start_time, "success"
    except Exception as e:
        return model, None, str(e)

async def test_anthropic_model(model):
    if anthropic is None:
        return model, None, "Anthropic library not available"
    try:
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        start_time = time.time()
        client.messages.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        end_time = time.time()
        return model, end_time - start_time, "success"
    except Exception as e:
        return model, None, str(e)

async def test_mistral_model(model):
    if mistralai is None:
        return model, None, "Mistral library not available"
    try:
        client = mistralai.Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
        start_time = time.time()
        client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": "Hello"}]
        )
        end_time = time.time()
        return model, end_time - start_time, "success"
    except Exception as e:
        return model, None, str(e)

async def test_cohere_model(model):
    if cohere is None:
        return model, None, "Cohere library not available"
    try:
        client = cohere.Client(api_key=st.secrets["COHERE_API_KEY"])
        start_time = time.time()
        client.chat(
            model=model,
            message="Hello",
            max_tokens=10
        )
        end_time = time.time()
        return model, end_time - start_time, "success"
    except Exception as e:
        return model, None, str(e)

async def test_together_model(model):
    if together is None:
        return model, None, "Together library not available"
    try:
        client = together.Together(api_key=st.secrets["TOGETHER_API_KEY"])
        start_time = time.time()
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        end_time = time.time()
        return model, end_time - start_time, "success"
    except Exception as e:
        return model, None, str(e)

async def test_deepseek_model(model):
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
        start_time = time.time()
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        end_time = time.time()
        return model, end_time - start_time, "success"
    except Exception as e:
        return model, None, str(e)

async def test_deepinfra_model(model):
    try:
        client = OpenAI(api_key=st.secrets["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")
        start_time = time.time()
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        end_time = time.time()
        return model, end_time - start_time, "success"
    except Exception as e:
        return model, None, str(e)

async def test_local_model(model):
    try:
        url = f"{config['ollama']['base_url']}/api/generate"
        payload = {
            "model": model,
            "prompt": "Hello",
            "stream": False
        }
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        end_time = time.time()
        if response.status_code == 200:
            return model, end_time - start_time, "success"
        else:
            return model, None, f"Error: {response.status_code}"
    except Exception as e:
        return model, None, str(e)

async def test_all_models():
    tasks = []
    groq_models = await get_groq_models()
    for model in groq_models:
        tasks.append(test_groq_model(model))

    for api, models in config.items():
        if isinstance(models, dict) and 'models' in models:
            for model in models['models']:
                if api == 'openai':
                    tasks.append(test_openai_model(model))
                elif api == 'anthropic' and anthropic:
                    tasks.append(test_anthropic_model(model))
                elif api == 'mistral' and mistralai:
                    tasks.append(test_mistral_model(model))
                elif api == 'cohere' and cohere:
                    tasks.append(test_cohere_model(model))
                elif api == 'together' and together:
                    tasks.append(test_together_model(model))
                elif api == 'deepseek':
                    tasks.append(test_deepseek_model(model))
                elif api == 'deepinfra':
                    tasks.append(test_deepinfra_model(model))
                elif api == 'ollama':
                    tasks.append(test_local_model(model))

    return await asyncio.gather(*tasks)

async def main():
    results = await test_all_models()
    
    model_speeds = {}
    for model, speed, status in results:
        api = next((api for api, models in config.items() if isinstance(models, dict) and 'models' in models and model in models['models']), 'groq')
        if api not in model_speeds:
            model_speeds[api] = []
        if status == "success":
            model_speeds[api].append({
                "model": model,
                "speed": speed
            })
        else:
            print(f"Test failed for {model}: {status}")

    for api in model_speeds:
        model_speeds[api] = sorted(model_speeds[api], key=lambda x: x["speed"])

    with open('model_speeds.json', 'w') as f:
        json.dump(model_speeds, f, indent=2)

    print("Resultados guardados en model_speeds.json")
    print("\nResumen de pruebas:")
    for api, models in model_speeds.items():
        print(f"{api}: {len(models)} modelos probados")

if __name__ == "__main__":
    print("Python version:", sys.version)
    print("Installed packages:")
    for package in ['openai', 'anthropic', 'mistralai', 'cohere', 'together']:
        try:
            module = importlib.import_module(package)
            print(f"{package}: {getattr(module, '__version__', 'Version not available')}")
        except ImportError:
            print(f"{package}: Not installed")
    
    asyncio.run(main())