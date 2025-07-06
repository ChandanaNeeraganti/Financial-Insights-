"""
Sample customer data for testing the financial analysis model
This provides realistic customer profiles for demonstration
"""

SAMPLE_CUSTOMERS = [
    {
        'customer_id': 'CUST001',
        'customer_name': 'John Smith',
        'monthly_income': 8500,
        'monthly_expenses': 5200,
        'savings_balance': 25000,
        'investment_balance': 45000,
        'total_debt': 180000,
        'payment_history_score': 0.95,
        'credit_utilization_ratio': 0.25,
        'credit_age_months': 84,
        'employment_years': 8,
        'loan_history': 2
    },
    {
        'customer_id': 'CUST002',
        'customer_name': 'Sarah Johnson',
        'monthly_income': 6200,
        'monthly_expenses': 3800,
        'savings_balance': 12000,
        'investment_balance': 18000,
        'total_debt': 95000,
        'payment_history_score': 0.88,
        'credit_utilization_ratio': 0.35,
        'credit_age_months': 60,
        'employment_years': 5,
        'loan_history': 1
    },
    {
        'customer_id': 'CUST003',
        'customer_name': 'Michael Brown',
        'monthly_income': 4500,
        'monthly_expenses': 3200,
        'savings_balance': 5000,
        'investment_balance': 8000,
        'total_debt': 65000,
        'payment_history_score': 0.75,
        'credit_utilization_ratio': 0.45,
        'credit_age_months': 36,
        'employment_years': 3,
        'loan_history': 1
    },
    {
        'customer_id': 'CUST004',
        'customer_name': 'Emily Davis',
        'monthly_income': 12000,
        'monthly_expenses': 6800,
        'savings_balance': 45000,
        'investment_balance': 120000,
        'total_debt': 220000,
        'payment_history_score': 0.98,
        'credit_utilization_ratio': 0.20,
        'credit_age_months': 120,
        'employment_years': 12,
        'loan_history': 3
    },
    {
        'customer_id': 'CUST005',
        'customer_name': 'David Wilson',
        'monthly_income': 3800,
        'monthly_expenses': 2800,
        'savings_balance': 3000,
        'investment_balance': 5000,
        'total_debt': 45000,
        'payment_history_score': 0.65,
        'credit_utilization_ratio': 0.60,
        'credit_age_months': 24,
        'employment_years': 2,
        'loan_history': 0
    },
    {
        'customer_id': 'CUST006',
        'customer_name': 'Lisa Anderson',
        'monthly_income': 7500,
        'monthly_expenses': 4200,
        'savings_balance': 18000,
        'investment_balance': 35000,
        'total_debt': 140000,
        'payment_history_score': 0.92,
        'credit_utilization_ratio': 0.30,
        'credit_age_months': 72,
        'employment_years': 7,
        'loan_history': 2
    },
    {
        'customer_id': 'CUST007',
        'customer_name': 'Robert Taylor',
        'monthly_income': 5500,
        'monthly_expenses': 4100,
        'savings_balance': 8000,
        'investment_balance': 15000,
        'total_debt': 85000,
        'payment_history_score': 0.70,
        'credit_utilization_ratio': 0.50,
        'credit_age_months': 48,
        'employment_years': 4,
        'loan_history': 1
    },
    {
        'customer_id': 'CUST008',
        'customer_name': 'Jennifer Martinez',
        'monthly_income': 6800,
        'monthly_expenses': 3900,
        'savings_balance': 15000,
        'investment_balance': 25000,
        'total_debt': 110000,
        'payment_history_score': 0.85,
        'credit_utilization_ratio': 0.40,
        'credit_age_months': 54,
        'employment_years': 6,
        'loan_history': 1
    },
    {
        'customer_id': 'CUST009',
        'customer_name': 'Christopher Garcia',
        'monthly_income': 4200,
        'monthly_expenses': 3500,
        'savings_balance': 4000,
        'investment_balance': 6000,
        'total_debt': 55000,
        'payment_history_score': 0.60,
        'credit_utilization_ratio': 0.70,
        'credit_age_months': 18,
        'employment_years': 1,
        'loan_history': 0
    },
    {
        'customer_id': 'CUST010',
        'customer_name': 'Amanda Rodriguez',
        'monthly_income': 9200,
        'monthly_expenses': 5500,
        'savings_balance': 22000,
        'investment_balance': 55000,
        'total_debt': 160000,
        'payment_history_score': 0.94,
        'credit_utilization_ratio': 0.28,
        'credit_age_months': 90,
        'employment_years': 9,
        'loan_history': 2
    }
]

def get_sample_customer(customer_id=None):
    """Get a specific customer or random customer from sample data"""
    import random
    if customer_id:
        for customer in SAMPLE_CUSTOMERS:
            if customer['customer_id'] == customer_id:
                return customer
        return None
    return random.choice(SAMPLE_CUSTOMERS)

def get_all_sample_customers():
    """Get all sample customers"""
    return SAMPLE_CUSTOMERS 