# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import mlflow.sklearn
import pandas as pd

app = FastAPI()

model = mlflow.sklearn.load_model("mlruns/0/models/m-14e093278d3f4cadad25bcbe2b15a8ae/artifacts")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    df = pd.get_dummies(df)
    prediction = model.predict(df)[0]
    return {"churn": int(prediction)}
