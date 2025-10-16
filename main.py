from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

# ===========================
# CORS CONFIGURATION
# ===========================
# Allow your frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, etc.
    allow_headers=["*"],
)

# ===========================
# LOAD MODEL AND ENCODERS
# ===========================
model = joblib.load("models/fraud_xgb_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# ===========================
# ROUTES
# ===========================
@app.get("/")
def home():
    return {"message": "🚀 Fraud Detection API is running!"}


@app.get("/ping")
async def ping():
    return {"status": "alive"}


@app.post("/predict")
async def predict(request: Request):
    transaction = await request.json()
    df = pd.DataFrame([transaction])

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    # Ensure all columns expected by XGBoost exist
    expected_cols = model.get_booster().feature_names
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    # Predict
    prediction = model.predict(df)[0]
    result = "Fraudulent" if prediction == 1 else "Legitimate"
    return {"prediction": result, "input": transaction}

# ===========================
# RUN SERVER
# ===========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
