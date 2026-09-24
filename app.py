
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

saved = joblib.load("diabetes_model.pkl")
model, scaler, columns = saved["model"], saved["scaler"], saved["columns"]

class Patient(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(p: Patient):
    row = pd.DataFrame([p.model_dump()], columns=columns)
    row_scaled = scaler.transform(row)
    prob = float(model.predict_proba(row_scaled)[0, 1])
    return {"diabetes_probability": prob}

# Serve the HTML frontend directly from FastAPI too
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")