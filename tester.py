import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib

# Load the datasets
train_df = pd.read_csv('train_transactions.csv')
test_df = pd.read_csv('test_transactions.csv')

# Preprocess the data
def preprocess(df):
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['transaction_year'] = df['transaction_date'].dt.year
    df['transaction_month'] = df['transaction_date'].dt.month
    df['transaction_day'] = df['transaction_date'].dt.day
    df['transaction_hour'] = df['transaction_date'].dt.hour
    df = df.drop('transaction_date', axis=1)
    
    label_encoders = {}
    for column in ['transaction_type', 'customer_id', 'merchant_id']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df, label_encoders

train_df, label_encoders = preprocess(train_df)
test_df, _ = preprocess(test_df)

# Drop the transaction_id column
train_df = train_df.drop('transaction_id', axis=1)
test_df = test_df.drop('transaction_id', axis=1)

# Separate features and target
X_train = train_df.drop('is_fraud', axis=1)
y_train = train_df['is_fraud']
X_test = test_df.drop('is_fraud', axis=1)
y_test = test_df['is_fraud']

# Train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save the model to a file
joblib.dump(model, 'fraud_detection_model.pkl')

# Load the model
model = joblib.load('fraud_detection_model.pkl')

# Predict on new data
new_data = X_test.sample(1)  # Just a sample for demonstration
prediction = model.predict(new_data)
print(f"Prediction for new data: {prediction}")

