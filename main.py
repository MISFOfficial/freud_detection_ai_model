import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Load trained model
model = joblib.load("fraud_model.pkl")

# 2. Init FastAPI
app = FastAPI(title="Fraud Detection API")

# 3. Define input schema
class Transaction(BaseModel):
    amount: float
    previous_txn_count_24h: int
    avg_amount_30d: float

# 4. Root endpoint
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running 🚀"}

# 5. Predict endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input to DataFrame
    input_data = pd.DataFrame([transaction.dict()])

    # Predict fraud (0 = genuine, 1 = fraud)
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # Fraud probability

    return {
        "fraud": bool(prediction),
        "risk_percentage": round(proba * 100, 2)
    }
