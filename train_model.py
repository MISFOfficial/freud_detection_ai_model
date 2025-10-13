import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =============================
# 1. LOAD DATA
# =============================
data = pd.read_csv("data/transactions.csv")

print("✅ Data loaded successfully!")
print("Shape:", data.shape)
print(data.head())

# =============================
# 2. HANDLE MISSING VALUES
# =============================
data = data.dropna()

# =============================
# 3. FEATURE ENCODING
# =============================
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# =============================
# 4. SPLIT FEATURES & TARGET
# =============================
X = data.drop(columns=["status"])
y = data["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# 5. TRAIN MODEL
# =============================
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# =============================
# 6. EVALUATE MODEL
# =============================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n✅ Model Training Complete!")
print("Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =============================
# 7. SAVE MODEL & ENCODERS
# =============================
joblib.dump(model, "fraud_detection_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("\n✅ Model and encoders saved successfully!")

# =============================
# 8. PREDICT NEW TRANSACTION (Safe for unseen labels)
# =============================
def predict_new(data_dict):
    df = pd.DataFrame([data_dict])
    
    # Apply same label encoding, unseen values get -1
    for column, le in label_encoders.items():
        if column in df:
            df[column] = df[column].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    prediction = model.predict(df)[0]
    return "Fraud" if prediction == 1 else "Real"

# =============================
# 9. TEST EXAMPLE
# =============================
test_data = {
    "avg_amount_30d": 0,
    "merchant_id": "unknown_merchant_123",
    "payment_method": "MOBILEBANKING",
    "card_type": "BKASH-BKash",
    "city": "Dhaka",
    "device_browser": "Chromium 140",
    "country": "Bangladesh",
    "previous_txn_count_24h": 0,
    "location": "37.4056,-122.0775",
    "device_os": "Linux",
    "user_id": "SMLBbqFsRCTtwGDsrPrLTZqeSJF3",
    "amount": 4234,
    "device_id": "ed6f1dc08530136442efbb1a9837eb67"
}

print("\nExample Prediction:", predict_new(test_data))

