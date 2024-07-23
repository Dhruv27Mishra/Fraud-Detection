import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Define the number of transactions
num_transactions = 100000
fraud_ratio = 0.10  # 5% of the transactions are fraudulent

# Generate synthetic data
transactions = []
for _ in range(num_transactions):
    transaction_id = fake.uuid4()
    customer_id = fake.uuid4()
    merchant_id = fake.uuid4()
    transaction_amount = round(random.uniform(1, 1000), 2)
    transaction_type = random.choice(['online', 'in-store', 'mobile'])
    transaction_date = fake.date_time_this_year()
    is_fraud = 1 if random.random() < fraud_ratio else 0
    
    transactions.append([
        transaction_id, customer_id, merchant_id, transaction_amount, 
        transaction_type, transaction_date, is_fraud
    ])

# Create a DataFrame
columns = ['transaction_id', 'customer_id', 'merchant_id', 'transaction_amount', 
           'transaction_type', 'transaction_date', 'is_fraud']
df = pd.DataFrame(transactions, columns=columns)

# Split the data into training and testing datasets
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Save to CSV files
train_df.to_csv('train_transactions.csv', index=False)
test_df.to_csv('test_transactions.csv', index=False)

# Display the first few rows of the datasets
print("Training Data:")
print(train_df.head())
print("\nTesting Data:")
print(test_df.head())
