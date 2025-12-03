from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load models
crop_model = joblib.load("crop_model.pkl")
fert_model = joblib.load("fert_model.pkl")
le = joblib.load("label_encoder.pkl")

class InputData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])

    # Crop Prediction
    crop_pred = crop_model.predict(X)[0]
    crop_name = le.inverse_transform([crop_pred])[0]

    # Fertilizer Prediction
    fert_pred = fert_model.predict(X)[0]
    fert = {
        "Urea": float(fert_pred[0]),
        "SSP": float(fert_pred[1]),
        "MOP": float(fert_pred[2]),
        "TSP": float(fert_pred[3]),
        "DAP": float(fert_pred[4]),
    }

    return {
        "recommended_crop": crop_name,
        "fertilizer_recommendation": fert
    }
