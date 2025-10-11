from fastapi import FastAPI, Request
import pandas as pd
import os
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_FILE_PATH = "data/transactions.csv"
MODEL_PATH = "fraud_xgb_model.pkl"
SCALER_PATH = "fraud_scaler.pkl"

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load CSV headers used during training
if os.path.exists(CSV_FILE_PATH):
    df_train_headers = pd.read_csv(CSV_FILE_PATH, nrows=1)
    TRAIN_COLUMNS = [c for c in df_train_headers.columns if c != "is_fraud"]
else:
    TRAIN_COLUMNS = []

DROP_COLS = ["transaction_id", "timestamp", "is_fraud"]  # always drop these

@app.post("/save-transaction")
async def save_transaction(request: Request):
    try:
        data = await request.json()
        transaction = data[0] if isinstance(data, list) else data
        print("✅ Received Transaction:", transaction)

        # Flatten devices
        flattened = {
            **transaction,
            "device_os": transaction.get("devices", {}).get("os", transaction.get("device_os", "Unknown")),
            "device_browser": transaction.get("devices", {}).get("browser", transaction.get("device_browser", "Unknown")),
            "device_id": transaction.get("devices", {}).get("deviceId", transaction.get("device_id", "Unknown")),
        }
        flattened.pop("devices", None)

        # Save to CSV
        file_exists = os.path.exists(CSV_FILE_PATH)
        df_existing = pd.read_csv(CSV_FILE_PATH) if file_exists else pd.DataFrame()
        df_existing = pd.concat([df_existing, pd.DataFrame([flattened])], ignore_index=True)
        df_existing.to_csv(CSV_FILE_PATH, index=False)
        print("💾 Transaction saved successfully.")

        # Prepare features exactly like training
        features = pd.DataFrame([flattened])

        # Drop unnecessary columns
        for col in DROP_COLS:
            if col in features.columns:
                features.drop(columns=[col], inplace=True)

        # Fill missing columns
        for col in TRAIN_COLUMNS:
            if col not in features.columns:
                features[col] = 0 if col in ["avg_amount_30d", "previous_txn_count_24h", "amount"] else ""

        # Ensure column order matches training
        features = features[TRAIN_COLUMNS]

        # Scale and predict
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1] if hasattr(model, "predict_proba") else 0

        # ✅ Send probability immediately
        return {
            "message": "✅ Transaction saved successfully",
            "data": flattened,
            "is_fraud": bool(pred),
            "fraud_probability": round(float(prob), 4),   # This is your immediate risk %
            "fraud_message": "🚨 Fraud detected!" if pred else "✅ Transaction looks safe"
        }

    except Exception as e:
        print("❌ Error while saving transaction or predicting:", e)
        return {"error": str(e)}
