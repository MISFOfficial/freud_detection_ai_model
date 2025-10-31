from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

# ===========================
# 🔹 FastAPI app
# ===========================
app = FastAPI(title="Fraud Detection API")

# ===========================
# 🔹 CORS CONFIGURATION
# ===========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js frontend
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# 🔹 LOAD MODEL & ENCODERS
# ===========================
model = joblib.load("models/fraud_xgb_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# ===========================
# 🔹 ROUTES
# ===========================
@app.get("/")
def home():
    return {"message": "🚀 Fraud Detection API is running!"}

@app.get("/ping")
async def ping():
    return {"status": "alive"}

@app.post("/predict")
async def predict(request: Request):
    """
    Receive a transaction JSON, preprocess, and return model prediction
    """
    try:
        # 1️⃣ Parse JSON
        transaction = await request.json()
        df = pd.DataFrame([transaction])
        print("data", df)

        # 2️⃣ Encode categorical features
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
            else:
                # Column missing in input, add default
                df[col] = 0

        # 3️⃣ Ensure column order matches training
        expected_cols = model.get_booster().feature_names
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        # 4️⃣ Make prediction
        prediction = model.predict(df)[0]
        result = "Fraud" if prediction == 1 else "Real"

        return {"prediction": result, "input": transaction}

    except Exception as e:
        return {"error": str(e), "input": transaction}

# ===========================
# 🔹 RUN SERVER
# ===========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
