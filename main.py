from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

# =============================
# Initialize FastAPI
# =============================
app = FastAPI(title="Fraud Detection API")

# Allow frontend (adjust the origin for your frontend)
origins = ["http://localhost:3000", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Load model and encoders
# =============================
MODEL_PATH = "fraud_detection_model.pkl"
ENCODER_PATH = "label_encoders.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("🚨 Model or encoders not found!")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

# =============================
# Health check
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

        df = pd.DataFrame([txn])

        # Apply label encoding safely (unseen labels get -1)
        for column, le in label_encoders.items():
            if column in df:
                df[column] = df[column].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Predict
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else 0

        return {
            "is_fraud": bool(pred),
            "fraud_probability": round(float(prob), 4),
            "fraud_message": "🚨 Fraud detected!" if pred else "✅ Transaction looks safe"
        }

    except Exception as e:
        return {"error": str(e)}

# =============================
# Run API
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
