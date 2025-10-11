# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# from imblearn.over_sampling import SMOTE # Remove SMOTE import
from xgboost import XGBClassifier
import joblib

# ============================
# 1️⃣ Load dataset
# ============================
df = pd.read_csv("data/transactions.csv")
print("Initial data shape:", df.shape)
print("Label distribution before fix:\n", df["is_fraud"].value_counts())

# ============================
# 2️⃣ Fix missing or invalid labels
# ============================
df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0)
df["is_fraud"] = df["is_fraud"].apply(lambda x: 1 if x > 0 else 0)

# If only one class exists, add a few fake fraud samples
if len(df["is_fraud"].unique()) == 1:
    print("⚠️ Only one class detected. Adding fake fraud samples for training...")
    fake_frauds = df.sample(2, replace=True).copy()
    fake_frauds["is_fraud"] = 1
    df = pd.concat([df, fake_frauds], ignore_index=True)

print("Label distribution after fix:\n", df["is_fraud"].value_counts())

# ============================
# 3️⃣ Encode categorical features
# ============================
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in ["timestamp", "transaction_id"]]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ============================
# 4️⃣ Separate features and target
# ============================
X = df.drop(columns=["is_fraud", "transaction_id", "timestamp"], errors="ignore")
y = df["is_fraud"]

# ============================
# 5️⃣ Balance dataset with SMOTE # Remove SMOTE step
# ============================
# minority_class_count = y.value_counts().min()
# if minority_class_count > SMOTE().n_neighbors:
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     print("Label distribution after SMOTE:\n", pd.Series(y_resampled).value_counts())
# else:
#     X_resampled, y_resampled = X, y
#     print(f"⚠️ Not enough samples ({minority_class_count}) in the minority class for SMOTE (requires > {SMOTE().n_neighbors}). Skipping SMOTE.")

# Use original data if SMOTE is skipped
X_resampled, y_resampled = X, y
print("SMOTE skipped.")


# ============================
# 6️⃣ Train-test split
# ============================
# Ensure stratification is possible with the current label distribution
if len(y_resampled.unique()) > 1 and y_resampled.value_counts().min() > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    print("⚠️ Stratification not possible due to insufficient samples in one or more classes. Skipping stratification.")


# ============================
# 7️⃣ Feature scaling
# ============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# 8️⃣ Train XGBoost model
# ============================
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

print("🚀 Training model...")
model.fit(X_train, y_train)
print("✅ Model training complete")

# ============================
# 9️⃣ Evaluate
# ============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print("-------------------")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# ============================
# 🔟 Save model and scaler
# ============================
joblib.dump(model, "fraud_xgb_model.pkl")
joblib.dump(scaler, "fraud_scaler.pkl")
print("\n✅ Model and scaler saved successfully.")