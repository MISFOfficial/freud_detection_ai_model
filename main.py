from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# =============================
# Initialize FastAPI
# =============================
app = FastAPI(title="Fraud Detection API")

# Allow frontend origins
origins = ["http://localhost:3000", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Paths
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_xgb_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
CSV_FILE_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")

# =============================
# Load model and encoders safely
# =============================
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("🚨 Model or encoders not found. Push them to the repo!")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

# Load training headers for reference
if os.path.exists(CSV_FILE_PATH):
    df_train_headers = pd.read_csv(CSV_FILE_PATH, nrows=1)
    TRAIN_COLUMNS = [c for c in df_train_headers.columns if c != "status"]
else:
    TRAIN_COLUMNS = []

DROP_COLS = ["transaction_id", "timestamp", "status"]

# =============================
# Health check endpoint
# =============================
@app.get("/ping")
def ping():
    return {"status": "alive"}

# =============================
# Prediction endpoint
# =============================
@app.post("/predict")
async def predict_transaction(request: Request):
    try:
        data = await request.json()
        txn = data[0] if isinstance(data, list) else data

        # Flatten nested device info if present
        flattened = {
            **txn,
            "device_os": txn.get("devices", {}).get("os", txn.get("device_os", "Unknown")),
            "device_browser": txn.get("devices", {}).get("browser", txn.get("device_browser", "Unknown")),
            "device_id": txn.get("devices", {}).get("deviceId", txn.get("device_id", "Unknown")),
        }
        flattened.pop("devices", None)

        # Apply label encoding safely
        for col, le in label_encoders.items():
            if col in flattened:
                try:
                    flattened[col] = le.transform([flattened[col]])[0]
                except ValueError:
                    # Handle unseen label
                    flattened[col] = -1

        # Prepare DataFrame
        features = pd.DataFrame([flattened])

        # Drop unnecessary columns
        for col in DROP_COLS:
            if col in features.columns:
                features.drop(columns=[col], inplace=True)

        # Add missing training columns with default values
        for col in TRAIN_COLUMNS:
            if col not in features.columns:
                features[col] = 0 if col in ["avg_amount_30d", "previous_txn_count_24h", "amount"] else ""

        # Ensure column order
        features = features[TRAIN_COLUMNS]

        # Predict
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else 0

        return {
            "message": "✅ Prediction successful",
            "transaction": flattened,
            "is_fraud": bool(pred),
            "fraud_probability": round(float(prob), 4),
            "fraud_message": "🚨 Fraud detected!" if pred else "✅ Transaction looks safe"
        }

    except Exception as e:
        return {"error": str(e)}