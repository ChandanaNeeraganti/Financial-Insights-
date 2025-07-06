import pandas as pd
import numpy as np
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.financial_models import CustomerFinancialAnalyzer

random.seed(42)
np.random.seed(42)

rows = []
num_approved = 0
num_rejected = 0
n_samples = 1000
analyzer = CustomerFinancialAnalyzer()

# Generate more realistic scenarios
scenarios = [
    # Good customers (should be approved)
    {'income_range': (50000, 200000), 'expense_ratio': (0.3, 0.6), 'savings_ratio': (0.1, 0.4), 'debt_ratio': (0, 0.3), 'payment_history': (0.9, 1.0), 'credit_age': (24, 120), 'target_approval': 1},
    
    # Moderate customers (mixed approval)
    {'income_range': (30000, 80000), 'expense_ratio': (0.5, 0.8), 'savings_ratio': (0.05, 0.2), 'debt_ratio': (0.2, 0.5), 'payment_history': (0.7, 0.95), 'credit_age': (12, 60), 'target_approval': 0.5},
    
    # Poor customers (should be rejected)
    {'income_range': (20000, 100000), 'expense_ratio': (0.8, 1.1), 'savings_ratio': (0, 0.05), 'debt_ratio': (0.4, 0.8), 'payment_history': (0.5, 0.8), 'credit_age': (0, 24), 'target_approval': 0},
    
    # High income but poor management (should be rejected)
    {'income_range': (200000, 500000), 'expense_ratio': (0.9, 1.2), 'savings_ratio': (0, 0.02), 'debt_ratio': (0.3, 0.7), 'payment_history': (0.6, 0.9), 'credit_age': (0, 36), 'target_approval': 0},
    
    # Low income with good management (should be approved)
    {'income_range': (15000, 40000), 'expense_ratio': (0.4, 0.7), 'savings_ratio': (0.1, 0.3), 'debt_ratio': (0, 0.2), 'payment_history': (0.9, 1.0), 'credit_age': (36, 120), 'target_approval': 1}
]

while len(rows) < n_samples:
    # Randomly select a scenario
    scenario = random.choice(scenarios)
    
    # Generate customer data based on scenario
    income = random.randint(*scenario['income_range'])
    expense_ratio = random.uniform(*scenario['expense_ratio'])
    expenses = int(income * expense_ratio)
    
    savings_ratio = random.uniform(*scenario['savings_ratio'])
    savings = int(income * savings_ratio * 12)  # Annual savings
    
    debt_ratio = random.uniform(*scenario['debt_ratio'])
    debt = int(income * debt_ratio * 12)  # Annual debt
    
    investment = random.randint(0, int(savings * 0.5))  # Some savings as investment
    payment_history = round(random.uniform(*scenario['payment_history']), 2)
    credit_util = round(random.uniform(0.01, 0.99), 2)
    credit_age = random.randint(*scenario['credit_age'])
    
    # Calculate derived metrics
    dti = debt / income if income > 0 else 1.0
    savings_rate = (income - expenses) / income if income > 0 else 0
    
    # Determine approval based on scenario and actual metrics
    should_approve = scenario['target_approval']
    if should_approve == 0.5:  # Mixed scenario
        should_approve = random.choice([0, 1])
    
    # Force balance: alternate adding approved/rejected
    if should_approve == 1 and num_approved < n_samples // 2:
        label = 1
        num_approved += 1
    elif should_approve == 0 and num_rejected < n_samples // 2:
        label = 0
        num_rejected += 1
    else:
        continue
    
    customer_data = {
        'monthly_income': income,
        'monthly_expenses': expenses,
        'savings_balance': savings,
        'investment_balance': investment,
        'total_debt': debt,
        'payment_history_score': payment_history,
        'credit_utilization_ratio': credit_util,
        'credit_age_months': credit_age
    }
    
    credit_score = analyzer.calculate_credit_score(customer_data)
    risk_score = analyzer.assess_risk_level(customer_data)['risk_score']
    financial_health_score = analyzer.calculate_financial_health_indicators(customer_data)['financial_health_score']
    
    rows.append({
        'customer_id': f'SYN{len(rows)+1:04d}',
        'customer_name': f'Synthetic Customer {len(rows)+1}',
        'monthly_income': income,
        'monthly_expenses': expenses,
        'savings_balance': savings,
        'investment_balance': investment,
        'total_debt': debt,
        'payment_history_score': payment_history,
        'credit_utilization_ratio': credit_util,
        'credit_age_months': credit_age,
        'credit_score': credit_score,
        'risk_score': risk_score,
        'financial_health_score': financial_health_score,
        'loan_approved': label
    })

# Save to CSV
synthetic_df = pd.DataFrame(rows)
print(synthetic_df['loan_approved'].value_counts())
synthetic_df.to_csv('models/synthetic_customers.csv', index=False)
print('Generated realistic synthetic_customers.csv with diverse loan approval labels.') 