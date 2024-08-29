import json
import time
import asyncio
import streamlit as st

try:
    import replicate
except ImportError:
    print("Error: 'replicate' module not found. Please install it using 'pip install replicate'.")
    exit(1)

# Get Replicate API token from Streamlit secrets
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]

# Initialize the Replicate client with your API token
client = replicate.Client(api_token=REPLICATE_API_TOKEN) 

async def get_replicate_models():
    """Fetches all models available on Replicate and returns their IDs."""
    models = []
    for page in replicate.paginate(client.models.list):
        models.extend(page.results)
        # You can add a limit here if you have too many models
        # if len(models) > 100:
        #     break
    return [model.id for model in models]

async def test_replicate_model(model_id):
    """Tests the speed of a Replicate model by making a prediction."""
    try:
        model = client.models.get(model_id)
        # You'll need to adapt the 'input' to match the model's requirements
        # Check the Replicate model's documentation for the correct input format
        input = {"prompt": "a photorealistic image of a cat wearing a hat"}  

        start_time = time.time()
        prediction = replicate.predictions.create(
            version=model.versions.list()[0].id,
            input=input
        )
        # Wait for the prediction to complete
        prediction.wait()
        end_time = time.time()

        return model_id, end_time - start_time, "success"
    except Exception as e:
        return model_id, None, str(e)

async def test_all_replicate_models():
    """Tests the speed of all Replicate models and returns the results."""
    models = await get_replicate_models()
    tasks = [test_replicate_model(model) for model in models]
    return await asyncio.gather(*tasks)

async def main():
    """Main function to run the tests and save the results."""
    results = await test_all_replicate_models()

    model_speeds = {
        "replicate": sorted(
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

    with open('replicate_model_speeds.json', 'w') as f:
        json.dump(model_speeds, f, indent=2)

    print("Resultados guardados en replicate_model_speeds.json")

if __name__ == "__main__":
    asyncio.run(main())