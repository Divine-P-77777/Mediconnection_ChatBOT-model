from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

import uvicorn
from contextlib import asynccontextmanager
import os
from fastapi.middleware.cors import CORSMiddleware

ENV = os.getenv("ENV", "development")

if ENV == "production":
    ALLOWED_ORIGINS = [
        "https://mediconnection.vercel.app",
    ]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
    ]



model = None
features = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, features
    try:
        if os.path.exists("model.pkl") and os.path.exists("features.pkl"):
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)

            with open("features.pkl", "rb") as f:
                features = list(pickle.load(f))

            print("Model and features loaded successfully")
        else:
            print(" model.pkl or features.pkl not found")
    except Exception as e:
        print(f" Error loading model: {e}")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SymptomPayload(BaseModel):
    symptoms: list[str]

@app.get("/")
def read_root():
    return {"message": "Disease Prediction API is running"}

@app.post("/predict")
def predict_disease(payload: SymptomPayload):
    if model is None or features is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_data = {feature: 0 for feature in features}

    for symptom in payload.symptoms:
        symptom = symptom.strip().lower()
        if symptom in input_data:
            input_data[symptom] = 1

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    return {"prediction": prediction}

if __name__ == "__main__":

    print("Starting server on port 50001 (Host 0.0.0.0)...")
    uvicorn.run("main:app", host="0.0.0.0", port=50001, reload=False)



