import json
import time
import asyncio
import aiohttp
import streamlit as st

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

async def test_all_groq_models():
    models = await get_groq_models()
    tasks = [test_groq_model(model) for model in models]
    return await asyncio.gather(*tasks)

async def main():
    results = await test_all_groq_models()
    
    model_speeds = {
        "groq": sorted(
            [
                {
                    "model": model,
                    "speed": speed,
                    "status": status
                }
                for model, speed, status in results
                if status == "success"
            ],
            key=lambda x: x["speed"]
        )
    }

    with open('groq_model_speeds.json', 'w') as f:
        json.dump(model_speeds, f, indent=2)

    print("Resultados guardados en groq_model_speeds.json")

if __name__ == "__main__":
    asyncio.run(main())