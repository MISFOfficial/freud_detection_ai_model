import pandas as pd
import numpy as np
import random
import uuid

# =============================
# 1. Define realistic options
# =============================
cities = ["Dhaka", "Chittagong", "Khulna", "Sylhet", "Barishal", "Rajshahi", "Rangpur", "Mymensingh"]
payment_methods = ["CARD", "MOBILEBANKING", "CASH", "PAYPAL"]
card_types = ["VISA", "MASTER", "AMEX", "BKASH-BKash", "NAGAD"]
browsers = ["Chrome 140", "Firefox 118", "Edge 110", "Safari 16", "Chromium 140"]
os_list = ["Android", "iOS", "Windows", "Linux"]
countries = ["Bangladesh"]

# =============================
# 2. Generate random data
# =============================
n_samples = 5000
data_list = []

for _ in range(n_samples):
    avg_amount_30d = random.randint(100, 5000)
    merchant_id = f"merchant_{random.randint(1, 1000)}"
    payment_method = random.choice(payment_methods)
    card_type = random.choice(card_types)
    city = random.choice(cities)
    device_browser = random.choice(browsers)
    country = random.choice(countries)
    previous_txn_count_24h = random.randint(0, 10)
    status = random.choices([0, 1], weights=[0.85, 0.15])[0]  # 15% fraud
    location = f"{round(random.uniform(20, 26), 6)},{round(random.uniform(88, 92), 6)}"
    device_os = random.choice(os_list)
    user_id = str(uuid.uuid4())
    amount = random.randint(50, 10000)
    device_id = str(uuid.uuid4())

    data_list.append([
        avg_amount_30d, merchant_id, payment_method, card_type, city,
        device_browser, country, previous_txn_count_24h, status,
        location, device_os, user_id, amount, device_id
    ])

# =============================
# 3. Create DataFrame
# =============================
columns = [
    "avg_amount_30d", "merchant_id", "payment_method", "card_type", "city",
    "device_browser", "country", "previous_txn_count_24h", "status",
    "location", "device_os", "user_id", "amount", "device_id"
]

df = pd.DataFrame(data_list, columns=columns)

# =============================
# 4. Save to CSV
# =============================
df.to_csv("testing_transactions.csv", index=False)
print("✅ Testing dataset with 5000 samples created successfully!")
