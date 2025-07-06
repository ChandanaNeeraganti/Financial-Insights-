import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load the new balanced synthetic data
customers = pd.read_csv('models/synthetic_customers.csv')

# Features for ML
feature_cols = [
    'monthly_income', 'monthly_expenses', 'savings_balance', 'investment_balance',
    'total_debt', 'payment_history_score', 'credit_utilization_ratio', 'credit_age_months',
    'credit_score', 'risk_score', 'financial_health_score'
]
X = customers[feature_cols]
y = customers['loan_approved']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/loan_approval_model.pkl')

print('Trained and saved loan approval model to models/loan_approval_model.pkl')
print('Train set approval rate:', y_train.mean())
print('Test set approval rate:', y_test.mean()) 