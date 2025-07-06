import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

class CSVProcessor:
    """
    Utility class for processing and validating CSV customer data
    Ensures data quality and consistency for financial analysis
    """
    
    def __init__(self):
        self.required_columns = [
            'customer_id', 'customer_name', 'monthly_income', 
            'monthly_expenses', 'savings_balance', 'investment_balance',
            'total_debt', 'payment_history_score', 'credit_utilization_ratio',
            'credit_age_months'
        ]
        
        self.optional_columns = [
            'employment_years', 'loan_history', 'bank_accounts',
            'credit_cards', 'mortgage_balance', 'auto_loan_balance',
            'student_loan_balance', 'other_debt'
        ]
        
        self.column_descriptions = {
            'customer_id': 'Unique identifier for the customer',
            'customer_name': 'Full name of the customer',
            'monthly_income': 'Gross monthly income in USD',
            'monthly_expenses': 'Total monthly expenses in USD',
            'savings_balance': 'Current savings account balance in USD',
            'investment_balance': 'Total investment portfolio value in USD',
            'total_debt': 'Total outstanding debt in USD',
            'payment_history_score': 'Payment history score (0.0 to 1.0)',
            'credit_utilization_ratio': 'Credit utilization ratio (0.0 to 1.0)',
            'credit_age_months': 'Length of credit history in months',
            'employment_years': 'Years of employment (optional)',
            'loan_history': 'Number of previous loans (optional)',
            'bank_accounts': 'Number of bank accounts (optional)',
            'credit_cards': 'Number of credit cards (optional)',
            'mortgage_balance': 'Outstanding mortgage balance (optional)',
            'auto_loan_balance': 'Outstanding auto loan balance (optional)',
            'student_loan_balance': 'Outstanding student loan balance (optional)',
            'other_debt': 'Other outstanding debt (optional)'
        }
    
    def validate_csv(self, file_path: str) -> Dict:
        """
        Validate CSV file structure and data quality
        Returns validation results with errors and warnings
        """
        validation_result = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'data_quality_score': 0.0,
            'missing_columns': [],
            'invalid_rows': [],
            'data_summary': {}
        }
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check for required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                validation_result['missing_columns'] = missing_columns
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Check data types and ranges
            if 'monthly_income' in df.columns:
                invalid_income = df[df['monthly_income'] <= 0]
                if not invalid_income.empty:
                    validation_result['warnings'].append(f"Found {len(invalid_income)} rows with invalid income (≤ 0)")
            
            if 'payment_history_score' in df.columns:
                invalid_payment = df[(df['payment_history_score'] < 0) | (df['payment_history_score'] > 1)]
                if not invalid_payment.empty:
                    validation_result['warnings'].append(f"Found {len(invalid_payment)} rows with invalid payment history scores")
            
            if 'credit_utilization_ratio' in df.columns:
                invalid_utilization = df[(df['credit_utilization_ratio'] < 0) | (df['credit_utilization_ratio'] > 1)]
                if not invalid_utilization.empty:
                    validation_result['warnings'].append(f"Found {len(invalid_utilization)} rows with invalid credit utilization ratios")
            
            # Check for missing values
            missing_values = df[self.required_columns].isnull().sum()
            if missing_values.sum() > 0:
                validation_result['warnings'].append(f"Found missing values in required columns: {missing_values[missing_values > 0].to_dict()}")
            
            # Calculate data quality score
            total_cells = len(df) * len(self.required_columns)
            valid_cells = total_cells - missing_values.sum()
            validation_result['data_quality_score'] = valid_cells / total_cells
            
            # Data summary
            validation_result['data_summary'] = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'required_columns_present': len(self.required_columns) - len(missing_columns),
                'optional_columns_present': len([col for col in self.optional_columns if col in df.columns])
            }
            
            # Determine if file is valid
            validation_result['is_valid'] = len(validation_result['errors']) == 0
            
        except Exception as e:
            validation_result['errors'].append(f"Error reading CSV file: {str(e)}")
        
        return validation_result
    
    def process_csv(self, file_path: str) -> List[Dict]:
        """
        Process CSV file and return list of customer dictionaries
        Includes data cleaning and validation
        """
        try:
            df = pd.read_csv(file_path)
            
            # Clean and validate data
            df = self._clean_data(df)
            
            # Convert to list of dictionaries
            customers = df.to_dict('records')
            
            # Add default values for missing optional columns
            for customer in customers:
                customer.setdefault('employment_years', 0)
                customer.setdefault('loan_history', 0)
                customer.setdefault('bank_accounts', 1)
                customer.setdefault('credit_cards', 0)
                customer.setdefault('mortgage_balance', 0)
                customer.setdefault('auto_loan_balance', 0)
                customer.setdefault('student_loan_balance', 0)
                customer.setdefault('other_debt', 0)
            
            return customers
            
        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data
        """
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values in required columns
        for col in self.required_columns:
            if col in df.columns:
                if col in ['monthly_income', 'monthly_expenses', 'savings_balance', 
                          'investment_balance', 'total_debt']:
                    # For financial columns, fill with 0
                    df[col] = df[col].fillna(0)
                elif col in ['payment_history_score', 'credit_utilization_ratio']:
                    # For ratio columns, fill with median
                    df[col] = df[col].fillna(df[col].median())
                elif col == 'credit_age_months':
                    # For credit age, fill with median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # For other columns, fill with appropriate defaults
                    df[col] = df[col].fillna('Unknown')
        
        # Ensure numeric columns are numeric
        numeric_columns = ['monthly_income', 'monthly_expenses', 'savings_balance', 
                          'investment_balance', 'total_debt', 'payment_history_score', 
                          'credit_utilization_ratio', 'credit_age_months']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Ensure ratios are within valid ranges
        if 'payment_history_score' in df.columns:
            df['payment_history_score'] = df['payment_history_score'].clip(0, 1)
        
        if 'credit_utilization_ratio' in df.columns:
            df['credit_utilization_ratio'] = df['credit_utilization_ratio'].clip(0, 1)
        
        return df
    
    def create_sample_csv(self, output_path: str, num_customers: int = 10):
        """
        Create a sample CSV file with the required format
        """
        import random
        
        sample_data = []
        
        for i in range(num_customers):
            customer = {
                'customer_id': f'CUST{i+1:03d}',
                'customer_name': f'Customer {i+1}',
                'monthly_income': random.randint(3000, 15000),
                'monthly_expenses': random.randint(2000, 8000),
                'savings_balance': random.randint(1000, 50000),
                'investment_balance': random.randint(5000, 100000),
                'total_debt': random.randint(20000, 300000),
                'payment_history_score': round(random.uniform(0.5, 1.0), 2),
                'credit_utilization_ratio': round(random.uniform(0.1, 0.8), 2),
                'credit_age_months': random.randint(12, 120),
                'employment_years': random.randint(1, 15),
                'loan_history': random.randint(0, 5),
                'bank_accounts': random.randint(1, 4),
                'credit_cards': random.randint(0, 5),
                'mortgage_balance': random.randint(0, 400000),
                'auto_loan_balance': random.randint(0, 50000),
                'student_loan_balance': random.randint(0, 100000),
                'other_debt': random.randint(0, 20000)
            }
            sample_data.append(customer)
        
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def get_csv_template(self) -> str:
        """
        Return CSV template with headers and example row
        """
        template_data = {
            'customer_id': ['CUST001'],
            'customer_name': ['John Smith'],
            'monthly_income': [8500],
            'monthly_expenses': [5200],
            'savings_balance': [25000],
            'investment_balance': [45000],
            'total_debt': [180000],
            'payment_history_score': [0.95],
            'credit_utilization_ratio': [0.25],
            'credit_age_months': [84],
            'employment_years': [8],
            'loan_history': [2],
            'bank_accounts': [2],
            'credit_cards': [3],
            'mortgage_balance': [150000],
            'auto_loan_balance': [25000],
            'student_loan_balance': [5000],
            'other_debt': [0]
        }
        
        df = pd.DataFrame(template_data)
        return df.to_csv(index=False) 