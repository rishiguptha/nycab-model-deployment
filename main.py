import os
import json
import joblib
import numpy as np
import torch  # Import torch for working with BERT
from transformers import DistilBertTokenizer, DistilBertModel  # Import BERT components
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
from google.oauth2 import service_account  # Import Google service account module
import openai  # Import OpenAI library for ChatGPT integration

# Initialize FastAPI app
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# Add this to your FastAPI app setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the models and BERT components
demand_model = None
duration_model = None
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this environment variable

# Define the structure of the request body
class PredictionRequest(BaseModel):
    model: str
    features: list

def load_model_from_gcs(bucket_name: str, model_path: str):
    """
    Load a model file from Google Cloud Storage.
    """
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in os.environ:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable is not set.")

    # Load credentials from the environment variable
    service_account_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(model_path)
    local_model_path = f"/tmp/{model_path.split('/')[-1]}"
    blob.download_to_filename(local_model_path)
    return joblib.load(local_model_path)

def preprocess_with_bert(input_text):
    """
    Preprocess input text with BERT to generate features.
    """
    # Tokenize and encode the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=32, padding="max_length", truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate BERT embeddings
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token output
    return features.numpy()

import requests
from fastapi import HTTPException

async def call_perplexity_api(prediction, model_type):
    """
    Use Perplexity API to convert the prediction into a natural language statement.
    """
    import logging
    logging.basicConfig(level=logging.DEBUG)

    url = "https://api.perplexity.ai/chat/completions"
    token = os.getenv("PERPLEXITY_API_KEY")  # Ensure this environment variable is set

    if not token:
        raise HTTPException(status_code=500, detail="Perplexity API token is not set.")

    # Define the prompt for Perplexity
    prompt = f"""
    Convert the following prediction into a natural language statement:
    - Model Type: {model_type}
    - Prediction: {prediction}

    Duration predictions should state the time in minutes or hours.
    Demand predictions should describe the approximate number of trips.
    """

    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "Be precise and concise."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "search_domain_filter": ["perplexity.ai"],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Make the API call
    response = requests.post(url, json=payload, headers=headers)

    # Debugging the API response
    logging.debug(f"Perplexity API response: {response.status_code} - {response.text}")

    if response.status_code == 200:
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        logging.debug(f"Extracted Content: {content}")
        return content
    else:
        raise HTTPException(status_code=500, detail=f"Perplexity API error: {response.text}")


@app.on_event("startup")
async def load_models():
    """
    Load the models into memory during startup.
    """
    global demand_model, duration_model
    bucket_name = "nycab-bucket"
    demand_model = load_model_from_gcs(bucket_name, "models/lgb_regressor_demand.pkl")
    duration_model = load_model_from_gcs(bucket_name, "models/lgb_regressor_duration.pkl")

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Endpoint to handle prediction requests.
    """
    import logging
    logging.basicConfig(level=logging.DEBUG)

    try:
        model_type = request.model
        raw_features = request.features

        # Generate input text for BERT preprocessing
        if model_type == "demand":
            input_text = f"Location: {raw_features[0]}, {raw_features[1]}. Hour: {raw_features[2]}. Weekday: {raw_features[3]}."
        elif model_type == "duration":
            input_text = f"Trip Distance: {raw_features[0]} miles. Hour: {raw_features[1]}. Weekday: {raw_features[2]}. Passengers: {raw_features[3]}."
        else:
            raise HTTPException(status_code=400, detail="Invalid model type. Use 'demand' or 'duration'.")

        # Preprocess input with BERT
        bert_features = preprocess_with_bert(input_text)

        # Predict using the correct model
        if model_type == "demand":
            prediction = demand_model.predict(bert_features)[0]
        elif model_type == "duration":
            prediction = duration_model.predict(bert_features)[0]

        logging.debug(f"Raw Prediction: {prediction}")

        # Convert prediction to natural language using Perplexity
        natural_language_response = await call_perplexity_api(prediction, model_type)
        logging.debug(f"Natural Language Response: {natural_language_response}")

        return {"prediction": natural_language_response}

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
