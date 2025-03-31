import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'C:\Users\Harshita Singh\Downloads\creditcard.csv'
data = pd.read_csv(file_path)

# Handle class imbalance if necessary (Oversampling/Undersampling)
X = data.drop(columns=['Class'])  # Features
y = data['Class']  # Target variable

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Real-time fraud detection simulation
def detect_fraud(transaction):
    transaction_scaled = scaler.transform([transaction])  # Scale input
    prediction = model.predict(transaction_scaled)
    return "Fraudulent Transaction Detected!" if prediction[0] == 1 else "Transaction is Safe."

# Example: Simulating a new transaction
sample_transaction = X.iloc[0].values  # Take first row as a test case
print(detect_fraud(sample_transaction))
