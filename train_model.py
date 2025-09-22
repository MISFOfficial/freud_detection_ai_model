import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# 1. Load dataset
df = pd.read_csv("data/synthetic_transactions.csv")

# 2. Features (X) & Target (y)
X = df[["amount", "previous_txn_count_24h", "avg_amount_30d"]]
y = df["is_fraud"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"  # new versions require this
)

# 5. Train model
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model saved as fraud_model.pkl")
