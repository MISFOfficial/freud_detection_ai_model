import pandas as pd
import numpy as np
import random

# Number of samples
n = 5000

# Possible values
card_types = ["Visa", "MasterCard", "AmEx", "Discover"]
cities = ["Dhaka", "Chittagong", "Sylhet", "Khulna", "Rajshahi", "Barishal", "Rangpur"]
countries = ["Bangladesh", "India", "USA", "UK", "Canada"]
browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
oses = ["Android", "iOS", "Windows", "macOS", "Linux"]
payment_methods = ["Credit Card", "Debit Card", "PayPal", "Crypto"]
statuses = ["success", "failed", "pending", "refunded"]

# Generate dataset
data = {
    "amount": np.round(np.random.uniform(5.0, 5000.0, n), 2),
    "card_type": np.random.choice(card_types, n),
    "city": np.random.choice(cities, n),
    "country": np.random.choice(countries, n),
    "device_browser": np.random.choice(browsers, n),
    "device_id": [f"D{random.randint(10000,99999)}" for _ in range(n)],
    "device_os": np.random.choice(oses, n),
    "is_fraud": np.random.choice([0, 1], n, p=[0.92, 0.08]),  # 8% fraud rate
    "location": [f"{round(random.uniform(23.5, 25.5), 4)}, {round(random.uniform(88.0, 90.5), 4)}" for _ in range(n)],
    "merchant_id": [f"M{random.randint(1, 100):03d}" for _ in range(n)],
    "payment_method": np.random.choice(payment_methods, n),
    "status": np.random.choice(statuses, n, p=[0.7, 0.1, 0.15, 0.05]),  # mostly successful
    "user_id": [f"U{random.randint(1, 500):03d}" for _ in range(n)],
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("synthetic_fraud_data.csv", index=False)

print("✅ synthetic_fraud_data.csv created successfully with", len(df), "rows")
