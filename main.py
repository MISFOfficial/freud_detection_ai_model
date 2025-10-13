from fastapi import FastAPI, Request
import pandas as pd
import os
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend
origins = ["http://localhost:3000", "*"]  # Allow all for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_xgb_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
CSV_FILE_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")

# Load model and encoders
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("🚨 Model or encoders not found. Push them to the repo!")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

# Get CSV headers
if os.path.exists(CSV_FILE_PATH):
    df_train_headers = pd.read_csv(CSV_FILE_PATH, nrows=1)
    TRAIN_COLUMNS = [c for c in df_train_headers.columns if c != "status"]
else:
    TRAIN_COLUMNS = []

DROP_COLS = ["transaction_id", "timestamp", "status"]

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.post("/save-transaction")
async def save_transaction(request: Request):
    try:
        data = await request.json()
        transaction = data[0] if isinstance(data, list) else data

        # Flatten devices
        flattened = {
            **transaction,
            "device_os": transaction.get("devices", {}).get("os", transaction.get("device_os", "Unknown")),
            "device_browser": transaction.get("devices", {}).get("browser", transaction.get("device_browser", "Unknown")),
            "device_id": transaction.get("devices", {}).get("deviceId", transaction.get("device_id", "Unknown")),
        }
        flattened.pop("devices", None)

        # Save to CSV
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
        file_exists = os.path.exists(CSV_FILE_PATH)
        df_existing = pd.read_csv(CSV_FILE_PATH) if file_exists else pd.DataFrame()
        df_existing = pd.concat([df_existing, pd.DataFrame([flattened])], ignore_index=True)
        df_existing.to_csv(CSV_FILE_PATH, index=False)

        # Prepare features
        features = pd.DataFrame([flattened])
        for col in DROP_COLS:
            if col in features.columns:
                features.drop(columns=[col], inplace=True)

        # Fill missing columns
        for col in TRAIN_COLUMNS:
            if col not in features.columns:
                features[col] = 0 if col in ["avg_amount_30d", "previous_txn_count_24h", "amount"] else ""

        # Ensure column order
        features = features[TRAIN_COLUMNS]

        # Encode categorical columns safely
        for col, le in label_encoders.items():
            if col in features.columns:
                # Replace unseen labels with a default
                features[col] = features[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else 0

        return {
            "message": "✅ Transaction saved successfully",
            "data": flattened,
            "is_fraud": bool(pred),
            "fraud_probability": round(float(prob), 4),
            "fraud_message": "🚨 Fraud detected!" if pred else "✅ Transaction looks safe"
        }

    except Exception as e:
        return {"error": str(e)}

# Only one uvicorn.run for Render
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
