import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class CustomerFinancialAnalyzer:
    """
    Comprehensive financial analysis model for banking customers
    Provides insights for bankers to make informed decisions
    """
    
    def __init__(self):
        self.credit_model = None
        self.risk_model = None
        self.financial_health_model = None
        self.scaler = StandardScaler()
        
    def calculate_credit_score(self, customer_data):
        """
        Calculate credit score based on multiple financial indicators
        Returns score between 300-850
        """
        # Extract key financial metrics
        income = customer_data.get('monthly_income', 0)
        expenses = customer_data.get('monthly_expenses', 0)
        savings = customer_data.get('savings_balance', 0)
        debt = customer_data.get('total_debt', 0)
        payment_history = customer_data.get('payment_history_score', 0)
        credit_utilization = customer_data.get('credit_utilization_ratio', 0)
        credit_age = customer_data.get('credit_age_months', 0)
        
        # Calculate debt-to-income ratio
        dti_ratio = (debt / income) if income > 0 else 1.0
        
        # Calculate savings rate
        savings_rate = ((income - expenses) / income) if income > 0 else 0
        
        # Base credit score calculation
        base_score = 300
        
        # Income factor (0-150 points)
        income_score = min(150, (income / 10000) * 50)
        
        # Payment history factor (0-200 points)
        payment_score = payment_history * 2
        
        # Credit utilization factor (0-100 points)
        utilization_score = max(0, 100 - (credit_utilization * 100))
        
        # Debt-to-income factor (0-100 points)
        dti_score = max(0, 100 - (dti_ratio * 100))
        
        # Savings factor (0-100 points)
        savings_score = min(100, savings_rate * 200)
        
        # Credit age factor (0-100 points)
        age_score = min(100, credit_age / 12)
        
        total_score = base_score + income_score + payment_score + utilization_score + dti_score + savings_score + age_score
        
        return min(850, max(300, int(total_score)))
    
    def assess_risk_level(self, customer_data):
        """
        Assess customer risk level (Low, Medium, High, Very High)
        """
        credit_score = self.calculate_credit_score(customer_data)
        income = customer_data.get('monthly_income', 0)
        debt = customer_data.get('total_debt', 0)
        dti_ratio = (debt / income) if income > 0 else 1.0
        
        risk_factors = []
        risk_score = 0
        
        # Credit score risk
        if credit_score < 580:
            risk_factors.append("Poor credit score")
            risk_score += 3
        elif credit_score < 670:
            risk_factors.append("Fair credit score")
            risk_score += 2
        elif credit_score < 740:
            risk_factors.append("Good credit score")
            risk_score += 1
        
        # Debt-to-income risk
        if dti_ratio > 0.43:
            risk_factors.append("High debt-to-income ratio")
            risk_score += 3
        elif dti_ratio > 0.36:
            risk_factors.append("Moderate debt-to-income ratio")
            risk_score += 2
        elif dti_ratio > 0.28:
            risk_factors.append("Acceptable debt-to-income ratio")
            risk_score += 1
        
        # Income stability
        if income < 3000:
            risk_factors.append("Low income")
            risk_score += 2
        
        # Determine risk level
        if risk_score >= 6:
            risk_level = "Very High"
        elif risk_score >= 4:
            risk_level = "High"
        elif risk_score >= 2:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors
        }
    
    def calculate_financial_health_indicators(self, customer_data):
        """
        Calculate comprehensive financial health indicators
        """
        income = customer_data.get('monthly_income', 0)
        expenses = customer_data.get('monthly_expenses', 0)
        savings = customer_data.get('savings_balance', 0)
        debt = customer_data.get('total_debt', 0)
        investments = customer_data.get('investment_balance', 0)
        
        # Emergency fund ratio (should be 3-6 months of expenses)
        emergency_fund_ratio = savings / expenses if expenses > 0 else 0
        
        # Savings rate
        savings_rate = ((income - expenses) / income) if income > 0 else 0
        
        # Debt-to-income ratio
        dti_ratio = (debt / income) if income > 0 else 1.0
        
        # Investment ratio
        investment_ratio = investments / income if income > 0 else 0
        
        # Net worth
        net_worth = savings + investments - debt
        
        # Financial health score (0-100)
        health_score = 0
        
        # Emergency fund scoring (0-25 points)
        if emergency_fund_ratio >= 6:
            health_score += 25
        elif emergency_fund_ratio >= 3:
            health_score += 20
        elif emergency_fund_ratio >= 1:
            health_score += 10
        
        # Savings rate scoring (0-25 points)
        if savings_rate >= 0.2:
            health_score += 25
        elif savings_rate >= 0.1:
            health_score += 20
        elif savings_rate >= 0.05:
            health_score += 10
        
        # Debt management scoring (0-25 points)
        if dti_ratio <= 0.28:
            health_score += 25
        elif dti_ratio <= 0.36:
            health_score += 15
        elif dti_ratio <= 0.43:
            health_score += 5
        
        # Investment scoring (0-25 points)
        if investment_ratio >= 0.1:
            health_score += 25
        elif investment_ratio >= 0.05:
            health_score += 15
        elif investment_ratio >= 0.02:
            health_score += 5
        
        return {
            'emergency_fund_ratio': emergency_fund_ratio,
            'savings_rate': savings_rate,
            'debt_to_income_ratio': dti_ratio,
            'investment_ratio': investment_ratio,
            'net_worth': net_worth,
            'financial_health_score': health_score,
            'health_category': self._categorize_health(health_score)
        }
    
    def _categorize_health(self, score):
        """Categorize financial health based on score"""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        else:
            return "Critical"
    
    def generate_lending_recommendations(self, customer_data):
        """
        Generate lending recommendations based on customer analysis
        """
        credit_score = self.calculate_credit_score(customer_data)
        risk_assessment = self.assess_risk_level(customer_data)
        health_indicators = self.calculate_financial_health_indicators(customer_data)
        
        recommendations = {
            'loan_approval': False,
            'recommended_loan_amount': 0,
            'interest_rate_range': '',
            'loan_terms': [],
            'conditions': [],
            'risk_mitigation': []
        }
        
        # Loan approval logic
        if credit_score >= 700 and risk_assessment['risk_level'] in ['Low', 'Medium']:
            recommendations['loan_approval'] = True
            
            # Calculate recommended loan amount based on income and DTI
            income = customer_data.get('monthly_income', 0)
            max_monthly_payment = income * 0.28  # 28% of income
            recommended_loan_amount = max_monthly_payment * 12  # Annual payment capacity
            
            recommendations['recommended_loan_amount'] = min(recommended_loan_amount, 500000)
            
            # Interest rate recommendations
            if credit_score >= 750:
                recommendations['interest_rate_range'] = '4.5% - 6.5%'
            elif credit_score >= 700:
                recommendations['interest_rate_range'] = '6.0% - 8.0%'
            
            # Loan terms
            if health_indicators['financial_health_score'] >= 60:
                recommendations['loan_terms'].append('Standard terms available')
            else:
                recommendations['loan_terms'].append('Restricted terms recommended')
        
        # Risk mitigation strategies
        if risk_assessment['risk_level'] in ['High', 'Very High']:
            recommendations['risk_mitigation'].append('Require co-signer')
            recommendations['risk_mitigation'].append('Higher down payment required')
            recommendations['risk_mitigation'].append('Shorter loan term recommended')
        
        if health_indicators['emergency_fund_ratio'] < 3:
            recommendations['conditions'].append('Establish emergency fund before loan approval')
        
        return recommendations
    
    def create_customer_summary(self, customer_data):
        """
        Create comprehensive customer summary for bankers
        """
        credit_score = self.calculate_credit_score(customer_data)
        risk_assessment = self.assess_risk_level(customer_data)
        health_indicators = self.calculate_financial_health_indicators(customer_data)
        recommendations = self.generate_lending_recommendations(customer_data)
        
        summary = {
            'customer_id': customer_data.get('customer_id', 'N/A'),
            'customer_name': customer_data.get('customer_name', 'N/A'),
            'credit_score': credit_score,
            'credit_rating': self._get_credit_rating(credit_score),
            'risk_assessment': risk_assessment,
            'financial_health': health_indicators,
            'lending_recommendations': recommendations,
            'key_metrics': {
                'monthly_income': customer_data.get('monthly_income', 0),
                'monthly_expenses': customer_data.get('monthly_expenses', 0),
                'total_debt': customer_data.get('total_debt', 0),
                'savings_balance': customer_data.get('savings_balance', 0),
                'investment_balance': customer_data.get('investment_balance', 0),
                'payment_history_score': customer_data.get('payment_history_score', 0),
                'credit_utilization_ratio': customer_data.get('credit_utilization_ratio', 0)
            },
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
    
    def _get_credit_rating(self, score):
        """Convert credit score to rating"""
        if score >= 800:
            return "Exceptional"
        elif score >= 740:
            return "Very Good"
        elif score >= 670:
            return "Good"
        elif score >= 580:
            return "Fair"
        else:
            return "Poor" 