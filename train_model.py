# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("data/transactions.csv")

# 2️⃣ Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3️⃣ Split features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 6️⃣ Save model and encoders
joblib.dump(model, "models/fraud_xgb_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("✅ Model training complete and saved!")
