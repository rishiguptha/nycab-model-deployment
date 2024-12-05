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

async def call_chatgpt_api(prediction, model_type):
    """
    Use ChatGPT (gpt-4 or gpt-3.5-turbo) to convert the prediction into a natural language statement.
    """
    prompt = f"""
    Convert the following prediction into a natural language statement:
    - Model Type: {model_type}
    - Prediction: {prediction}

    Duration predictions should state the time in minutes or hours.
    Demand predictions should describe the approximate number of trips.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use gpt-4 or gpt-3.5-turbo
        messages=[
            {"role": "system", "content": "You are a Expert at predicting demand and duration prediction, where you process the predictions into natural language statements.."},
            {"role": "user", "content": prompt},
        ],
    )
    return response["choices"][0]["message"]["content"].strip()

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

        # Convert prediction to natural language using ChatGPT
        natural_language_response = await call_chatgpt_api(prediction, model_type)

        return {"prediction": natural_language_response}

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
