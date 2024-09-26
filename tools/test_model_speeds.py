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
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Configurar las advertencias para suprimir las relacionadas con pydantic
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config = load_config()

def safe_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        console.print(f"[bold red]Failed to import {module_name}: {str(e)}[/bold red]")
        return None

anthropic = safe_import('anthropic')
mistralai = safe_import('mistralai')
cohere = safe_import('cohere')
together = safe_import('together')

console = Console()

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
                console.print(f"[bold red]Error fetching Groq models: {response.status}[/bold red]")
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

async def test_openrouter_model(model):
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
            "HTTP-Referer": "https://marduk.pro",
            "X-Title": "MALLO",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                await response.json()
        end_time = time.time()
        return model, end_time - start_time, "success"
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
                elif api == 'openrouter':
                    tasks.append(test_openrouter_model(model))

    return await asyncio.gather(*tasks)

async def main():
    console.print(Panel.fit("🚀 [bold cyan]Model Speed Test[/bold cyan]", border_style="bold"))
    
    console.print("\n[bold green]Environment Information:[/bold green]")
    console.print(f"Python version: {sys.version}")
    
    table = Table(title="Installed Packages")
    table.add_column("Package", style="cyan")
    table.add_column("Version", style="magenta")
    
    for package in ['openai', 'anthropic', 'mistralai', 'cohere', 'together', 'aiohttp']:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Version not available')
            table.add_row(package, version)
        except ImportError:
            table.add_row(package, "[red]Not installed[/red]")
    
    console.print(table)
    
    with console.status("[bold green]Running model speed tests...[/bold green]", spinner="dots"):
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
            console.print(f"[bold red]Test failed for {model}: {status}[/bold red]")

    for api in model_speeds:
        model_speeds[api] = sorted(model_speeds[api], key=lambda x: x["speed"])

    with open('model_speeds.json', 'w') as f:
        json.dump(model_speeds, f, indent=2)

    console.print("\n[bold green]Results saved in model_speeds.json[/bold green]")
    
    console.print("\n[bold cyan]Test Summary:[/bold cyan]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("API", style="cyan")
    summary_table.add_column("Models Tested", justify="right")
    
    for api, models in model_speeds.items():
        summary_table.add_row(api, str(len(models)))
    
    console.print(summary_table)

if __name__ == "__main__":
    asyncio.run(main())