import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
from google.oauth2 import service_account  # Import the missing module

app = FastAPI()

# Global variables to hold the models
demand_model = None
duration_model = None

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
    model_type = request.model
    features = np.array(request.features).reshape(1, -1)

    if model_type == "demand":
        prediction = demand_model.predict(features)
    elif model_type == "duration":
        prediction = duration_model.predict(features)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type. Use 'demand' or 'duration'.")

    return {"prediction": prediction.tolist()}
