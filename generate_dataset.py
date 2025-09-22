# generate_dataset.py
import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()
num_samples = 3000
data = []

for i in range(num_samples):
    transaction_id = f"txn_{i+1}"
    user_id = f"user_{random.randint(1,700)}"
    amount = round(np.random.exponential(scale=50) + np.random.choice([0,50,200], p=[0.85,0.1,0.05]), 2)
    timestamp = fake.date_time_between(start_date='-90d', end_date='now')
    payment_method = random.choice(["card", "wallet", "bank_transfer", "credit_card", "debit_card"])
    merchant_id = f"m_{random.randint(1,120)}"
    previous_txn_count_24h = int(np.random.poisson(1.5))
    avg_amount_30d = round(np.random.normal(120, 60), 2)

    # simple synthetic fraud rule + random
    is_fraud = 0
    if amount > 300 and previous_txn_count_24h < 1 and payment_method in ("card","credit_card"):
        if random.random() < 0.65:
            is_fraud = 1
    elif random.random() < 0.02:
        is_fraud = 1

    data.append([
        transaction_id, user_id, amount, timestamp.isoformat(sep=' '),
        payment_method, merchant_id,
        previous_txn_count_24h, avg_amount_30d, is_fraud
    ])

df = pd.DataFrame(data, columns=[
    "transaction_id","user_id","amount","timestamp",
    "payment_method","merchant_id",
    "previous_txn_count_24h","avg_amount_30d","is_fraud"
])

os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", "synthetic_transactions.csv")
df.to_csv(csv_path, index=False)
print("✅ Dataset generated:", csv_path)
print(df.is_fraud.value_counts().to_dict())
