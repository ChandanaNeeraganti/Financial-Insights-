import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import io
import re
# Add project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.bank_statement_parser import BankStatementParser
import numpy as np
import joblib
import sklearn
from universal_bank_parser import UniversalBankStatementParser
from sentence_transformers import SentenceTransformer


from models.financial_models import CustomerFinancialAnalyzer
from data.sample_customers import get_all_sample_customers

# --- Merchant category mapping and extraction ---
merchant_category_map = {
    'amazon': 'Shopping',
    'swiggy': 'Food Delivery',
    'zomato': 'Food Delivery',
    'paytm': 'Wallet/Recharge',
    'irctc': 'Travel',
    'uber': 'Transport',
    'ola': 'Transport',
    'petrol': 'Fuel',
    'fuel': 'Fuel',
    'salary': 'Salary',
    'interest': 'Interest',
    'dividend': 'Investment',
    'mutual fund': 'Investment',
    'fd': 'Investment',
    'rd': 'Investment',
    'ppf': 'Investment',
    'lic': 'Insurance',
    'nps': 'Investment',
    'bonus': 'Salary',
    'reimbursement': 'Reimbursement',
    'refund': 'Refund',
    'rent': 'Rent',
    'grocery': 'Grocery',
    'shopping': 'Shopping',
    'food': 'Food',
    'cashback': 'Cashback',
    'insurance': 'Insurance',
    'loan': 'Loan',
    'emi': 'Loan',
    'savings': 'Savings',
    'closure proceeds': 'Savings',
    'maturity proceeds': 'Savings',
    'transfer': 'Transfer',
    'upi': 'UPI',
    'imps': 'IMPS',
    'rtgs': 'RTGS',
    'bank transfer': 'Transfer',
    'gift': 'Gift',
    'award': 'Award',
    'profit': 'Investment',
    'sale': 'Sale',
    'cash deposit': 'Deposit',
    'settlement': 'Settlement',
    'arrears': 'Salary',
    'advance': 'Advance',
    'medical': 'Medical',
    'consulting': 'Consulting',
    'stipend': 'Stipend',
    'scholarship': 'Scholarship',
    'royalty': 'Royalty',
    'pension': 'Pension',
    'award': 'Award',
    'commission': 'Commission',
    'reversal': 'Reversal',
    'expense claim': 'Reimbursement',
    'final settlement': 'Settlement',
    'self': 'Transfer',
    'client': 'Client',
    'employer': 'Salary',
    'medical claim': 'Medical',
    'insurance claim': 'Insurance',
    'dividend payout': 'Investment',
    'bonus received': 'Salary',
    'profit share': 'Investment',
    'sale proceeds': 'Sale',
    'online transfer': 'Transfer',
    'reimbursement credit': 'Reimbursement',
    'expense claim': 'Reimbursement',
    'medical claim': 'Medical',
    'insurance claim': 'Insurance',
    'final settlement': 'Settlement',
    'arrears': 'Salary',
    'advance payment': 'Advance',
    'advance credit': 'Advance',
}
def extract_merchant_category(description):
    desc = str(description).lower()
    for keyword, category in merchant_category_map.items():
        if keyword in desc:
            return category
    return 'Other'

# --- Expanded keyword lists for hybrid logic ---
income_keywords = [
    'salary', 'opening balance', 'refund', 'reimbursement', 'interest', 'dividend', 'bonus', 'arrears', 'stipend', 'scholarship', 'royalty', 'commission', 'pension', 'award', 'profit', 'sale', 'rent received', 'credit', 'closing balance', 'incentive', 'payment from', 'received from', 'salary credit', 'salary deposit', 'salary payment', 'salary transfer', 'employer', 'consulting', 'settlement', 'reimbursement credit', 'expense claim', 'medical claim', 'insurance claim', 'final settlement', 'advance', 'arrears', 'advance payment', 'advance credit', 'cash deposit', 'online transfer', 'gift', 'award', 'profit share', 'sale proceeds', 'sale credit', 'dividend payout', 'bonus received', 'rent received', 'credited by', 'credited to', 'credited', 'credit'
]
savings_patterns = [
    'closure', 'maturity', 'redemption', 'transfer to own account', 'fd closure', 'rd closure', 'fd maturity', 'rd maturity', 'mf redemption', 'sip redemption', 'ppf withdrawal', 'lic refund', 'policy maturity', 'proceeds credit', 'td closure', 'refund from mf', 'fd interest', 'rd interest', 'mutual fund redemption', 'fixed deposit maturity', 'recurring deposit maturity', 'investment proceeds', 'sip installment', 'invested in mf', 'transfer to rd', 'transfer to fd', 'transfer to ppf', 'transfer to demat', 'lic premium', 'td purchase', 'deposit to nps', 'transfer to savings', 'own account transfer', 'linked account transfer', 'sip debit', 'mutual fund purchase', 'fixed deposit opening', 'rd opening', 'ppf deposit', 'nps contribution', 'insurance premium', 'investment debit'
]

def hybrid_label_rule(row, model_label):
    ttype = str(row['Type']).upper()
    desc = str(row.get('Description', '')).lower()
    # 1. Force Income for strong income keywords
    if any(kw in desc for kw in income_keywords):
        return 'Income', 'Rule-Income-Keyword'
    # 2. Only allow Savings if strong savings pattern
    if model_label == 'Savings':
        if any(pat in desc for pat in savings_patterns):
            return 'Savings', 'Rule-Savings-Pattern'
        else:
            # If not a strong savings pattern, fallback to Expense or Income
            if ttype == 'CREDIT':
                return 'Income', 'Rule-Income-Override'
            else:
                return 'Expense', 'Rule-Expense-Override'
    # 3. Type-based fallback
    if ttype == 'CREDIT':
        if model_label == 'Expense':
            return 'Income', 'Rule-Income-Override'
        else:
            return model_label, 'Model'
    elif ttype == 'DEBIT':
        if model_label == 'Income':
            return 'Expense', 'Rule-Expense-Override'
        else:
            return model_label, 'Model'
    else:
        return model_label, 'Model'

class FinancialDashboard:
    """
    Interactive dashboard for financial analysis and customer insights
    Designed for bankers to make informed decisions
    """
    
    def __init__(self):
        self.analyzer = CustomerFinancialAnalyzer()
        self.customer_data = None
        self.analysis_results = []
        # Load ML model
        try:
            self.ml_model = joblib.load('models/loan_approval_model.pkl')
        except Exception:
            self.ml_model = None
        
    def load_csv_data(self, uploaded_file):
        """Load and validate CSV customer data"""
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = [
                'customer_id', 'customer_name', 'monthly_income', 
                'monthly_expenses', 'savings_balance', 'investment_balance',
                'total_debt', 'payment_history_score', 'credit_utilization_ratio',
                'credit_age_months'
            ]
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.info("Required columns: customer_id, customer_name, monthly_income, monthly_expenses, savings_balance, investment_balance, total_debt, payment_history_score, credit_utilization_ratio, credit_age_months")
                return None
            
            # Convert to list of dictionaries
            customers = df.to_dict('records')
            return customers
            
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return None
    
    def analyze_customers(self, customers):
        """Analyze all customers and store results"""
        self.analysis_results = []
        progress_bar = st.progress(0)
        
        for i, customer in enumerate(customers):
            summary = self.analyzer.create_customer_summary(customer)
            # Merge original customer fields into summary at top level
            summary.update(customer)
            self.analysis_results.append(summary)
            progress_bar.progress((i + 1) / len(customers))
        
        progress_bar.empty()
        return self.analysis_results
    
    def display_customer_overview(self):
        """Display high-level customer overview with clean, simple design"""
        if not self.analysis_results:
            return
        
        st.header("📊 Customer Portfolio Overview")
        
        # Calculate key metrics
        total_customers = len(self.analysis_results)
        avg_credit_score = sum(r['credit_score'] for r in self.analysis_results) / total_customers
        avg_risk_score = sum(r['risk_assessment']['risk_score'] for r in self.analysis_results) / total_customers
        approved_count = sum(1 for r in self.analysis_results if r['lending_recommendations']['loan_approval'])
        approval_rate = (approved_count / total_customers) * 100
        
        # Simple metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", total_customers)
        
        with col2:
            st.metric("Average Credit Score", f"{avg_credit_score:.0f}")
        
        with col3:
            st.metric("Average Risk Score", f"{avg_risk_score:.1f}")
        
        with col4:
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        # Create data for charts
        risk_counts = {}
        health_counts = {}
        credit_ranges = {'Poor (300-579)': 0, 'Fair (580-669)': 0, 'Good (670-739)': 0, 'Very Good (740-799)': 0, 'Excellent (800-850)': 0}
        
        for result in self.analysis_results:
            # Risk levels
            risk_level = result['risk_assessment']['risk_level']
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            # Health categories
            health_cat = result['financial_health']['health_category']
            health_counts[health_cat] = health_counts.get(health_cat, 0) + 1
            
            # Credit score ranges
            score = result['credit_score']
            if score < 580:
                credit_ranges['Poor (300-579)'] += 1
            elif score < 670:
                credit_ranges['Fair (580-669)'] += 1
            elif score < 740:
                credit_ranges['Good (670-739)'] += 1
            elif score < 800:
                credit_ranges['Very Good (740-799)'] += 1
            else:
                credit_ranges['Excellent (800-850)'] += 1
        
        # Charts section
        st.subheader("Portfolio Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Risk distribution chart
            risk_df = pd.DataFrame(list(risk_counts.items()), columns=['Risk Level', 'Count'])
            fig = px.pie(risk_df, values='Count', names='Risk Level', 
                        title="Risk Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Financial health distribution
            health_df = pd.DataFrame(list(health_counts.items()), columns=['Health Category', 'Count'])
            fig = px.bar(health_df, x='Health Category', y='Count',
                        title="Financial Health Distribution")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Credit score distribution
            credit_df = pd.DataFrame(list(credit_ranges.items()), columns=['Credit Range', 'Count'])
            fig = px.bar(credit_df, x='Credit Range', y='Count',
                        title="Credit Score Distribution")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Insights section
        st.subheader("Portfolio Insights")
        
        if avg_credit_score < 600:
            st.warning("Average credit score is low. Consider focusing on credit-building products or stricter lending criteria.")
        elif avg_credit_score >= 700:
            st.success("Average credit score is healthy. Portfolio is well-positioned for lending.")
        else:
            st.info("Average credit score is acceptable. Monitor credit trends.")
        
        if avg_risk_score > 4:
            st.error("Portfolio risk is high. Review risk management strategies.")
        elif avg_risk_score <= 2:
            st.success("Portfolio has low risk profile. Consider expanding lending criteria.")
        else:
            st.info("Portfolio risk is manageable. Continue monitoring.")
        
        if approval_rate < 30:
            st.warning("Approval rate is low. Review approval criteria or target segments.")
        elif approval_rate > 70:
            st.info("High approval rate. Ensure risk controls are adequate.")
        else:
            st.success("Approval rate is well-balanced. Portfolio is performing well.")
        
        # Portfolio summary
        st.subheader("Portfolio Summary")

        # Initialize totals to 0 to avoid UnboundLocalError
        total_income = 0
        total_expenses = 0
        total_savings = 0
        total_debt = 0
        total_all = 0
        total_debit = 0
        total_credit = 0
        label_sums = {}

        if hasattr(self, 'ml_df') and self.ml_df is not None:
            # Force mapping of DR/CR to DEBIT/CREDIT for summary calculations
            self.ml_df['Type'] = self.ml_df['Type'].astype(str).str.strip().str.upper().replace({'DR': 'DEBIT', 'CR': 'CREDIT'})
            # Detect label column
            label_col = None
            for col in ['Predicted_Label', 'Label']:
                if col in self.ml_df.columns:
                    label_col = col
                    break
            if label_col is None:
                st.error('No label column (Predicted_Label or Label) found in uploaded data!')
                return
            # Ensure Amount is numeric
            self.ml_df['Amount'] = pd.to_numeric(self.ml_df['Amount'], errors='coerce')
            # Normalize label values
            self.ml_df[label_col] = self.ml_df[label_col].astype(str).str.strip().str.upper()
            st.write(f"[DEBUG] Unique values in Type column: {self.ml_df['Type'].unique()}")
            st.write(f"[DEBUG] Label column: {label_col}")
            st.write(f"[DEBUG] Label value counts: {self.ml_df[label_col].value_counts().to_dict()}")
            # Debug: show sum for each unique label
            label_sums = self.ml_df.groupby(label_col)['Amount'].sum().to_dict()
            st.write(f"[DEBUG] Amount sum by label: {label_sums}")
            # Extra debug: show first 5 rows labeled as SAVINGS
            savings_rows = self.ml_df[self.ml_df[label_col] == 'SAVINGS']
            st.write(f"[DEBUG] First 5 SAVINGS rows:")
            st.write(savings_rows.head())

            total_income = self.ml_df.loc[self.ml_df[label_col] == 'INCOME', 'Amount'].sum()
            total_expenses = self.ml_df.loc[self.ml_df[label_col] == 'EXPENSE', 'Amount'].sum()
            total_savings = self.ml_df.loc[self.ml_df[label_col] == 'SAVINGS', 'Amount'].sum()
            total_debt = self.ml_df.loc[self.ml_df[label_col] == 'DEBT', 'Amount'].sum()
            total_all = self.ml_df['Amount'].sum()
            total_debit = self.ml_df.loc[self.ml_df['Type'] == 'DEBIT', 'Amount'].sum()
            total_credit = self.ml_df.loc[self.ml_df['Type'] == 'CREDIT', 'Amount'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Income (CSV)", f"₹{total_income:,.0f}")
        
        with col2:
            st.metric("Total Expenses (CSV)", f"₹{total_expenses:,.0f}")
        
        with col3:
            st.metric("Total Savings (CSV)", f"₹{total_savings:,.0f}")
        
        with col4:
            st.metric("Total Debt", f"₹{total_debt:,.0f}")
        
        # Add a second row for raw totals
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Total Amount (All Transactions)", f"₹{total_all:,.0f}")
        with col6:
            st.metric("Total Debit (CSV)", f"₹{total_debit:,.0f}")
        with col7:
            st.metric("Total Credit (CSV)", f"₹{total_credit:,.0f}")
        
        # Add a bar chart for Income, Expense, Savings
        chart_labels = []
        chart_values = []
        for label in ['INCOME', 'EXPENSE', 'SAVINGS']:
            if label in label_sums:
                chart_labels.append(label)
                chart_values.append(label_sums[label])
        if chart_labels:
            fig = go.Figure([go.Bar(x=chart_labels, y=chart_values, marker_color=['green', 'red', 'blue'])])
            fig.update_layout(title="Total by Category", xaxis_title="Category", yaxis_title="Total Amount (₹)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Show a table of all transactions and their predicted labels for review
        if hasattr(self, 'ml_df') and self.ml_df is not None:
            st.subheader("All Transactions with Predicted Category")
            st.dataframe(self.ml_df[['Date', 'Description', 'Amount', 'Type', 'Predicted_Label']])
            # Export button for manual labeling
            csv_export = self.ml_df[['Date', 'Description', 'Amount', 'Type', 'Predicted_Label']].to_csv(index=False)
            st.download_button("Export Transactions for Labeling", data=csv_export, file_name="transactions_for_labeling.csv", mime="text/csv")
    
    def display_credit_score_analysis(self):
        """Display credit score analysis"""
        if not self.analysis_results:
            return
        
        st.header("💳 Credit Score Analysis")
        
        # Create credit score distribution
        credit_scores = [r['credit_score'] for r in self.analysis_results]
        ratings = [r['credit_rating'] for r in self.analysis_results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Credit score histogram
            fig = px.histogram(x=credit_scores, nbins=20, 
                             title="Credit Score Distribution",
                             labels={'x': 'Credit Score', 'y': 'Number of Customers'})
            fig.add_vline(x=580, line_dash="dash", line_color="red", 
                         annotation_text="Poor Credit Threshold")
            fig.add_vline(x=670, line_dash="dash", line_color="orange", 
                         annotation_text="Fair Credit Threshold")
            fig.add_vline(x=740, line_dash="dash", line_color="yellow", 
                         annotation_text="Good Credit Threshold")
            fig.add_vline(x=800, line_dash="dash", line_color="green", 
                         annotation_text="Exceptional Credit Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Credit rating distribution
            rating_counts = pd.Series(ratings).value_counts()
            fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                        title="Credit Rating Distribution",
                        labels={'x': 'Credit Rating', 'y': 'Number of Customers'})
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Insights & Suggestions ---
        st.markdown("---")
        st.subheader("Insights & Suggestions")
        poor = sum(1 for s in credit_scores if s < 580)
        fair = sum(1 for s in credit_scores if 580 <= s < 670)
        good = sum(1 for s in credit_scores if 670 <= s < 740)
        very_good = sum(1 for s in credit_scores if 740 <= s < 800)
        exceptional = sum(1 for s in credit_scores if s >= 800)
        if poor / len(credit_scores) > 0.5:
            st.error("Majority of customers have poor credit. Tighten lending or offer credit improvement programs.")
        elif good + very_good + exceptional > len(credit_scores) * 0.5:
            st.success("Most customers have good or better credit. Consider pre-approved offers.")
        else:
            st.info("Credit score distribution is mixed. Segment offers accordingly.")
    
    def display_financial_health_analysis(self):
        """Display financial health indicators"""
        if not self.analysis_results:
            return
        
        st.header("🏥 Financial Health Analysis")
        
        # Extract financial health data
        health_scores = [r['financial_health']['financial_health_score'] for r in self.analysis_results]
        health_categories = [r['financial_health']['health_category'] for r in self.analysis_results]
        dti_ratios = [r['financial_health']['debt_to_income_ratio'] for r in self.analysis_results]
        savings_rates = [r['financial_health']['savings_rate'] for r in self.analysis_results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Financial health score distribution
            fig = px.histogram(x=health_scores, nbins=15,
                             title="Financial Health Score Distribution",
                             labels={'x': 'Health Score', 'y': 'Number of Customers'})
            fig.add_vline(x=20, line_dash="dash", line_color="red", 
                         annotation_text="Critical")
            fig.add_vline(x=40, line_dash="dash", line_color="orange", 
                         annotation_text="Poor")
            fig.add_vline(x=60, line_dash="dash", line_color="yellow", 
                         annotation_text="Fair")
            fig.add_vline(x=80, line_dash="dash", line_color="green", 
                         annotation_text="Excellent")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # DTI vs Savings Rate scatter plot
            fig = px.scatter(x=dti_ratios, y=savings_rates,
                           title="Debt-to-Income vs Savings Rate",
                           labels={'x': 'Debt-to-Income Ratio', 'y': 'Savings Rate'},
                           color=health_categories)
            fig.add_hline(y=0.1, line_dash="dash", line_color="green", 
                         annotation_text="Good Savings Rate (10%)")
            fig.add_vline(x=0.28, line_dash="dash", line_color="green", 
                         annotation_text="Good DTI (28%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Insights & Suggestions ---
        st.markdown("---")
        st.subheader("Insights & Suggestions")
        low_health = sum(1 for s in health_scores if s < 40)
        high_health = sum(1 for s in health_scores if s >= 60)
        if low_health / len(health_scores) > 0.5:
            st.error("Most customers have poor financial health. Consider offering financial wellness programs.")
        elif high_health / len(health_scores) > 0.5:
            st.success("Majority have good financial health. Portfolio is resilient.")
        else:
            st.info("Financial health is mixed. Target interventions where needed.")
    
    def display_lending_recommendations(self):
        """Display lending recommendations summary"""
        if not self.analysis_results:
            return
        
        st.header("💰 Lending Recommendations")
        
        # Filter approved customers
        approved_customers = [r for r in self.analysis_results if r['lending_recommendations']['loan_approval']]
        rejected_customers = [r for r in self.analysis_results if not r['lending_recommendations']['loan_approval']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Approved Customers")
            if approved_customers:
                approved_df = pd.DataFrame([
                    {
                        'Customer': r['customer_name'],
                        'Credit Score': r['credit_score'],
                        'Risk Level': r['risk_assessment']['risk_level'],
                        'Recommended Amount': f"₹{r['lending_recommendations']['recommended_loan_amount']:,.0f}",
                        'Interest Rate': r['lending_recommendations']['interest_rate_range']
                    }
                    for r in approved_customers
                ])
                st.dataframe(approved_df, use_container_width=True)
            else:
                st.info("No customers approved for loans")
        
        with col2:
            st.subheader("❌ Rejected Customers")
            if rejected_customers:
                rejection_reasons = []
                for r in rejected_customers:
                    reasons = []
                    if r['credit_score'] < 700:
                        reasons.append("Low credit score")
                    if r['risk_assessment']['risk_level'] in ['High', 'Very High']:
                        reasons.append("High risk level")
                    rejection_reasons.append(", ".join(reasons))
                
                rejected_df = pd.DataFrame([
                    {
                        'Customer': r['customer_name'],
                        'Credit Score': r['credit_score'],
                        'Risk Level': r['risk_assessment']['risk_level'],
                        'Rejection Reasons': reason
                    }
                    for r, reason in zip(rejected_customers, rejection_reasons)
                ])
                st.dataframe(rejected_df, use_container_width=True)
            else:
                st.info("All customers approved for loans")
        
        # --- Insights & Suggestions ---
        # st.markdown("---")
        # st.subheader("Insights & Suggestions")
        # approved_customers = [r for r in self.analysis_results if r['lending_recommendations']['loan_approval']]
        # rejected_customers = [r for r in self.analysis_results if not r['lending_recommendations']['loan_approval']]
        # if not approved_customers:
        #     st.warning("No customers approved for loans. Review approval criteria or customer targeting.")
        # if rejected_customers:
        #     common_reasons = []
        #     for r in rejected_customers:
        #         if r['credit_score'] < 700:
        #             common_reasons.append("Low credit score")
        #         if r['risk_assessment']['risk_level'] in ['High', 'Very High']:
        #             common_reasons.append("High risk level")
        #     if common_reasons:
        #         st.info(f"Common rejection reasons: {', '.join(set(common_reasons))}")
    
    def display_individual_customer_analysis(self):
        """Display detailed analysis for individual customers"""
        if not self.analysis_results:
            return
        
        st.header("👤 Individual Customer Analysis")
        
        # Customer selector
        customer_options = [f"{r['customer_name']} ({r['customer_id']})" for r in self.analysis_results]
        selected_customer = st.selectbox("Select Customer", customer_options)
        
        if selected_customer:
            # Find selected customer
            customer_id = selected_customer.split("(")[-1].split(")")[0]
            customer_result = next(r for r in self.analysis_results if r['customer_id'] == customer_id)
            
            # Display customer details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Customer Information")
                st.write(f"**Name:** {customer_result['customer_name']}")
                st.write(f"**Customer ID:** {customer_result['customer_id']}")
                st.write(f"**Credit Score:** {customer_result['credit_score']} ({customer_result['credit_rating']})")
                st.write(f"**Risk Level:** {customer_result['risk_assessment']['risk_level']}")
                st.write(f"**Financial Health:** {customer_result['financial_health']['health_category']}")
                
                # Key metrics
                st.subheader("Key Financial Metrics")
                metrics = customer_result['key_metrics']
                st.write(f"**Monthly Income:** ₹{metrics['monthly_income']:,.0f}")
                st.write(f"**Monthly Expenses:** ₹{metrics['monthly_expenses']:,.0f}")
                st.write(f"**Total Debt:** ₹{metrics['total_debt']:,.0f}")
                st.write(f"**Savings Balance:** ₹{metrics['savings_balance']:,.0f}")
                st.write(f"**Investment Balance:** ₹{metrics['investment_balance']:,.0f}")
            
            with col2:
                # Financial health indicators
                st.subheader("Financial Health Indicators")
                health = customer_result['financial_health']
                st.write(f"**Emergency Fund Ratio:** {health['emergency_fund_ratio']:.1f} months")
                st.write(f"**Savings Rate:** {health['savings_rate']:.1%}")
                st.write(f"**Debt-to-Income Ratio:** {health['debt_to_income_ratio']:.1%}")
                st.write(f"**Investment Ratio:** {health['investment_ratio']:.1%}")
                st.write(f"**Net Worth:** ₹{health['net_worth']:,.0f}")
                
                # Lending recommendations
                st.subheader("Lending Decision")
                lending = customer_result['lending_recommendations']
                if lending['loan_approval']:
                    st.success("✅ **APPROVED**")
                    st.write(f"**Recommended Amount:** ₹{lending['recommended_loan_amount']:,.0f}")
                    st.write(f"**Interest Rate Range:** {lending['interest_rate_range']}")
                else:
                    st.error("❌ **REJECTED**")
                
                if lending['conditions']:
                    st.write("**Conditions:**")
                    for condition in lending['conditions']:
                        st.write(f"• {condition}")
                
                if lending['risk_mitigation']:
                    st.write("**Risk Mitigation:**")
                    for mitigation in lending['risk_mitigation']:
                        st.write(f"• {mitigation}")
            
            # --- Insights & Suggestions ---
            st.markdown("---")
            st.subheader("Insights & Suggestions")
            if customer_result['credit_score'] >= 740 and customer_result['risk_assessment']['risk_level'] == 'Low':
                st.success("Excellent candidate for premium products or pre-approved loans.")
            elif customer_result['risk_assessment']['risk_level'] in ['High', 'Very High']:
                st.error("High risk customer. Recommend additional due diligence or risk mitigation.")
            elif customer_result['financial_health']['health_category'] in ['Poor', 'Critical']:
                st.warning("Customer has poor financial health. Suggest financial counseling or restricted lending.")
            else:
                st.info("Customer is in a moderate segment. Review full profile before proceeding.")
    
    def display_insights_and_segmentation(self):
        """Display advanced insights and customer segmentation"""
        if not self.analysis_results:
            return
        st.header("🔍 Insights & Customer Segmentation")
        customer = self.analysis_results[0]  # Only one customer for bank statement mode
        metrics = customer['key_metrics']
        health = customer['financial_health']
        
        # Savings rate
        savings_rate = health['savings_rate']
        net_cash_flow = metrics['monthly_income'] - metrics['monthly_expenses']
        
        # Segmentation logic
        income = metrics['monthly_income']
        expenses = metrics['monthly_expenses']
        balance = metrics['savings_balance']
        
        if income >= 50000:
            income_segment = "High Income"
        elif income >= 20000:
            income_segment = "Moderate Income"
        else:
            income_segment = "Low Income"
        
        if savings_rate >= 0.2:
            savings_segment = "Saver"
        elif savings_rate >= 0.05:
            savings_segment = "Balanced"
        else:
            savings_segment = "High Spender"
        
        if abs(net_cash_flow) < 0.05 * income:
            cashflow_segment = "Stable Cash Flow"
        elif net_cash_flow > 0:
            cashflow_segment = "Net Saver"
        else:
            cashflow_segment = "Net Spender"
        
        st.subheader("Customer Segmentation")
        st.write(f"**Income Segment:** {income_segment}")
        st.write(f"**Spending/Saving Segment:** {savings_segment}")
        st.write(f"**Cash Flow Segment:** {cashflow_segment}")
        
        st.markdown("---")
        st.subheader("Key Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Monthly Income", f"₹{income:,.0f}")
        with col2:
            st.metric("Monthly Expenses", f"₹{expenses:,.0f}")
        with col3:
            st.metric("Net Cash Flow", f"₹{net_cash_flow:,.0f}", delta=f"{savings_rate:.1%} Savings Rate")
        st.write(f"**Savings Rate:** {savings_rate:.1%}")
        st.write(f"**Current Balance:** ₹{balance:,.0f}")
        
        # Trend chart if possible
        if 'analysis_date' in customer:
            st.info("Trends are based on the uploaded statement period. For multi-month trends, upload a longer statement.")
        
        # Recommendations
        st.markdown("---")
        st.subheader("Recommendations for Banker")
        if income_segment == "High Income" and savings_segment == "Saver":
            st.success("This customer is financially strong and a low-risk candidate for most banking products.")
        elif savings_segment == "High Spender":
            st.warning("Customer spends most of their income. Consider counseling on savings or offering budgeting tools.")
        elif cashflow_segment == "Net Spender":
            st.error("Customer is spending more than they earn. High risk for lending.")
        else:
            st.info("Customer is in a moderate segment. Review full profile before proceeding.")
    
    def display_ml_prediction_tab(self):
        if not self.analysis_results:
            return
        import streamlit as st
        import plotly.graph_objects as go
        st.header("🤖 ML Model Loan Approval Prediction")
        st.markdown("""
        This page uses a machine learning model to predict loan approval for each customer based on their financial profile. 
        The model considers income, expenses, savings, debt, credit history, and rule-based scores (credit score, risk score, financial health score). 
        Use this as a data-driven second opinion alongside rule-based recommendations.
        """)
        # Check if the first analysis result has the minimum required keys for auto-generated or real profiles
        min_required_keys = [
            'monthly_income', 'monthly_expenses', 'savings_balance',
            'investment_balance', 'total_debt', 'payment_history_score',
            'credit_utilization_ratio', 'credit_age_months'
        ]
        first_result = self.analysis_results[0]
        if not all(key in first_result for key in min_required_keys):
            st.info("Loan approval prediction is only available for customer-level or auto-generated profiles. Please upload a customer data CSV file or a bank statement.")
            return
        # For auto-generated profiles, fill in missing advanced fields with defaults
        customer_options = [f"Customer {i+1}" for i in range(len(self.analysis_results))]
        selected_customer = st.selectbox("Select Customer for ML Prediction", customer_options)
        if selected_customer:
            idx = int(selected_customer.split(" ")[-1]) - 1
            customer_result = self.analysis_results[idx]
            # Prepare input for ML model
            ml_input = customer_result.copy()
            # Add dummy advanced fields if missing
            ml_input.setdefault('credit_score', 700)
            ml_input.setdefault('risk_score', 2.0)
            ml_input.setdefault('financial_health_score', 3.0)
            ml_pred, ml_prob = self.predict_loan_approval_ml(ml_input)
            st.subheader("ML Model Prediction Result")
            if ml_pred is not None:
                if ml_pred == 1:
                    st.success(f"Loan Approved ({ml_prob:.0%} confidence)")
                else:
                    st.error(f"Loan Rejected ({(1-ml_prob):.0%} confidence)")
                # Visual confidence bar
                st.markdown("**Prediction Confidence:**")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = ml_prob*100 if ml_pred == 1 else (1-ml_prob)*100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': 'green' if ml_pred == 1 else 'red'},
                        'steps': [
                            {'range': [0, 50], 'color': '#ffe6e6' if ml_pred == 0 else '#e6ffe6'},
                            {'range': [50, 100], 'color': '#e6ffe6' if ml_pred == 1 else '#ffe6e6'}
                        ],
                    },
                    number = {'suffix': '%'}
                ))
                st.plotly_chart(fig, use_container_width=True)
                # Actionable insights
                st.markdown("---")
                st.subheader("Actionable Insights")
                if ml_pred == 1 and ml_prob > 0.8:
                    st.success("This customer is a strong candidate for loan approval. Consider offering premium products or pre-approved offers.")
                elif ml_pred == 1:
                    st.info("Customer is likely to be approved, but review full profile for risk factors.")
                elif ml_pred == 0 and ml_prob > 0.8:
                    st.error("Customer is a high risk for loan rejection. Consider additional documentation or risk mitigation.")
                else:
                    st.warning("Customer is borderline for approval. Manual review recommended.")
            else:
                st.info("ML model not available or not loaded.")
            st.markdown("---")
            st.write("**Features used for prediction:**")
            st.json(ml_input)

    def export_results(self):
        """Export analysis results to Excel"""
        if not self.analysis_results:
            return
        
        # Create comprehensive Excel report
        with pd.ExcelWriter('financial_analysis_report.xlsx', engine='xlsxwriter') as writer:
            
            # Summary sheet
            summary_data = []
            for r in self.analysis_results:
                summary_data.append({
                    'Customer ID': r['customer_id'],
                    'Customer Name': r['customer_name'],
                    'Credit Score': r['credit_score'],
                    'Credit Rating': r['credit_rating'],
                    'Risk Level': r['risk_assessment']['risk_level'],
                    'Risk Score': r['risk_assessment']['risk_score'],
                    'Financial Health Score': r['financial_health']['financial_health_score'],
                    'Health Category': r['financial_health']['health_category'],
                    'Loan Approval': r['lending_recommendations']['loan_approval'],
                    'Recommended Amount': r['lending_recommendations']['recommended_loan_amount'],
                    'Interest Rate Range': r['lending_recommendations']['interest_rate_range'],
                    'Monthly Income': r['key_metrics']['monthly_income'],
                    'Monthly Expenses': r['key_metrics']['monthly_expenses'],
                    'Total Debt': r['key_metrics']['total_debt'],
                    'Savings Balance': r['key_metrics']['savings_balance'],
                    'Investment Balance': r['key_metrics']['investment_balance'],
                    'DTI Ratio': r['financial_health']['debt_to_income_ratio'],
                    'Savings Rate': r['financial_health']['savings_rate'],
                    'Emergency Fund Ratio': r['financial_health']['emergency_fund_ratio'],
                    'Net Worth': r['financial_health']['net_worth']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Risk factors sheet
            risk_data = []
            for r in self.analysis_results:
                risk_data.append({
                    'Customer ID': r['customer_id'],
                    'Customer Name': r['customer_name'],
                    'Risk Level': r['risk_assessment']['risk_level'],
                    'Risk Factors': ', '.join(r['risk_assessment']['risk_factors'])
                })
            
            risk_df = pd.DataFrame(risk_data)
            risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
            
            # Lending recommendations sheet
            lending_data = []
            for r in self.analysis_results:
                lending_data.append({
                    'Customer ID': r['customer_id'],
                    'Customer Name': r['customer_name'],
                    'Loan Approval': r['lending_recommendations']['loan_approval'],
                    'Recommended Amount': r['lending_recommendations']['recommended_loan_amount'],
                    'Interest Rate Range': r['lending_recommendations']['interest_rate_range'],
                    'Loan Terms': ', '.join(r['lending_recommendations']['loan_terms']),
                    'Conditions': ', '.join(r['lending_recommendations']['conditions']),
                    'Risk Mitigation': ', '.join(r['lending_recommendations']['risk_mitigation'])
                })
            
            lending_df = pd.DataFrame(lending_data)
            lending_df.to_excel(writer, sheet_name='Lending Recommendations', index=False)
        
        return 'financial_analysis_report.xlsx'

    def load_bank_statement(self, uploaded_file, customer_id='AUTO', customer_name='Unknown'):
        """Load and parse a raw bank statement CSV file"""
        try:
            parser = BankStatementParser(customer_id=customer_id, customer_name=customer_name)
            # Save uploaded file to a temp location
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            customer = parser.parse(tmp_path)
            self.ml_df = customer  # Store the DataFrame for use in all analysis functions
            return [customer]
        except Exception as e:
            st.error(f"Error parsing bank statement: {str(e)}")
            return None

    def predict_loan_approval_ml(self, customer):
        if self.ml_model is None:
            return None, None
        # Use the correct customer-level features for the loan approval model
        feature_cols = [
            'monthly_income', 'monthly_expenses', 'savings_balance', 'investment_balance',
            'total_debt', 'payment_history_score', 'credit_utilization_ratio', 'credit_age_months',
            'credit_score', 'risk_score', 'financial_health_score'
        ]
        X = pd.DataFrame([{col: customer.get(col, 0) for col in feature_cols}])
        print("ML input features:", X.to_dict(orient='records')[0])  # Debug print
        pred = self.ml_model.predict(X)[0]
        proba = self.ml_model.predict_proba(X)[0]
        if len(proba) == 1:
            prob = proba[0] if pred == 0 else 1 - proba[0]
        else:
            prob = proba[1]
        return pred, prob

# --- Add this helper function to aggregate transaction data and build customer profile ---
def build_customer_profile_from_transactions(df):
    import numpy as np
    from datetime import datetime
    # Aggregate sums
    monthly_income = df.loc[df['Predicted_Label'] == 'Income', 'Amount'].sum()
    monthly_expenses = df.loc[df['Predicted_Label'] == 'Expense', 'Amount'].sum()
    savings_balance = df.loc[df['Predicted_Label'] == 'Savings', 'Amount'].sum()
    # Estimate credit_age_months from earliest transaction
    if 'Date' in df.columns:
        try:
            # Try to parse dates
            dates = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
            min_date = dates.min()
            max_date = dates.max()
            if pd.notnull(min_date) and pd.notnull(max_date):
                credit_age_months = max(1, (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month))
            else:
                credit_age_months = 24  # fallback default
        except Exception:
            credit_age_months = 24
    else:
        credit_age_months = 24
    # Build profile dict with sensible defaults for other fields
    customer_profile = {
        'monthly_income': monthly_income,
        'monthly_expenses': monthly_expenses,
        'savings_balance': savings_balance,
        'investment_balance': 0,
        'total_debt': 0,
        'payment_history_score': 1.0,
        'credit_utilization_ratio': 0.3,
        'credit_age_months': credit_age_months,
    }
    return customer_profile

def rule_based_ai_analysis(customer_data):
    """
    Rule-based AI analysis that provides intelligent insights
    """
    income = customer_data.get('income', 0)
    expenses = customer_data.get('expenses', 0)
    savings = customer_data.get('savings', 0)
    transactions = customer_data.get('transactions', 0)
    
    analysis = []
    
    # Risk Assessment
    risk_score = 0
    risk_factors = []
    
    if income > 0:
        expense_ratio = expenses / income
        savings_ratio = savings / income
        
        if expense_ratio > 0.9:
            risk_score += 30
            risk_factors.append("Critical: Expenses exceed 90% of income")
        elif expense_ratio > 0.8:
            risk_score += 20
            risk_factors.append("High: Expenses exceed 80% of income")
        elif expense_ratio > 0.7:
            risk_score += 10
            risk_factors.append("Moderate: Expenses exceed 70% of income")
        
        if savings_ratio < 0.05:
            risk_score += 25
            risk_factors.append("Critical: Savings below 5% of income")
        elif savings_ratio < 0.1:
            risk_score += 15
            risk_factors.append("High: Savings below 10% of income")
        elif savings_ratio < 0.2:
            risk_score += 5
            risk_factors.append("Moderate: Savings below 20% of income")
    
    # Income Stability Analysis
    if income > 50000:
        if savings < 0.15 * income:
            analysis.append("🔍 **Income Analysis**: High income but low savings - potential lifestyle inflation")
        else:
            analysis.append("✅ **Income Analysis**: Strong income with good savings discipline")
    elif income > 25000:
        analysis.append("📊 **Income Analysis**: Moderate income level - focus on expense optimization")
    else:
        analysis.append("⚠️ **Income Analysis**: Lower income - prioritize essential expenses")
    
    # Savings Behavior Analysis
    if savings > 0.3 * income:
        analysis.append("🏆 **Savings Excellence**: Outstanding savings rate - eligible for premium products")
    elif savings > 0.2 * income:
        analysis.append("✅ **Good Savings**: Above-average savings - consider investment products")
    elif savings > 0.1 * income:
        analysis.append("📈 **Moderate Savings**: Adequate savings - room for improvement")
    else:
        analysis.append("⚠️ **Low Savings**: Below recommended savings rate - financial education needed")
    
    # Product Recommendations
    recommendations = []
    if income > 40000 and expenses/income < 0.7:
        recommendations.append("💳 **Credit Card**: Eligible for premium credit cards")
    if savings > 0.2 * income:
        recommendations.append("🏦 **Fixed Deposits**: High savings - suggest FD products")
    if income > 30000 and savings < 0.15 * income:
        recommendations.append("📱 **Recurring Deposits**: Help build savings discipline")
    if income > 50000:
        recommendations.append("🏠 **Home Loan**: High income - pre-approved home loan eligible")
    if savings > 0.25 * income:
        recommendations.append("📊 **Investment Products**: Suggest SIP, mutual funds")
    
    # Risk Level Classification
    if risk_score >= 40:
        risk_level = "🔴 HIGH RISK"
    elif risk_score >= 20:
        risk_level = "🟡 MEDIUM RISK"
    else:
        risk_level = "🟢 LOW RISK"
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "analysis": analysis,
        "recommendations": recommendations,
        "customer_profile": f"Income: ₹{income:,.2f}, Expenses: ₹{expenses:,.2f}, Savings: ₹{savings:,.2f}, Transactions: {transactions}"
    }

def ai_opportunity_risk_analyzer(df):
    st.header('🤖 AI Opportunity & Risk Analyzer')
    if df is None or df.empty:
        st.info('No customer data loaded. Please upload a statement or CSV.')
        return
    
    # Compute key metrics
    income = df[df['Predicted_Label'] == 'INCOME']['Amount'].sum()
    expenses = df[df['Predicted_Label'] == 'EXPENSE']['Amount'].sum()
    savings = df[df['Predicted_Label'] == 'SAVINGS']['Amount'].sum()
    n_txns = len(df)
    
    # Rule-based AI Analysis
    customer_data = {
        'income': income,
        'expenses': expenses,
        'savings': savings,
        'transactions': n_txns
    }
    
    ai_result = rule_based_ai_analysis(customer_data)
    
    # Display AI Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Risk Assessment")
        st.markdown(f"**Risk Level: {ai_result['risk_level']}**")
        st.markdown(f"**Risk Score: {ai_result['risk_score']}/100**")
        
        if ai_result['risk_factors']:
            st.write("**Risk Factors:**")
            for factor in ai_result['risk_factors']:
                st.write(f"• {factor}")
    
    with col2:
        st.subheader("📊 Customer Profile")
        st.write(ai_result['customer_profile'])
        if income > 0:
            st.write(f"Expense Ratio: {expenses/income:.1%}")
            st.write(f"Savings Ratio: {savings/income:.1%}")
    
    st.markdown("---")
    
    # Detailed Analysis
    st.subheader("🧠 AI Analysis")
    for analysis in ai_result['analysis']:
        st.markdown(analysis)
    
    # Product Recommendations
    st.subheader("💡 Product Recommendations")
    if ai_result['recommendations']:
        for rec in ai_result['recommendations']:
            st.markdown(rec)
    else:
        st.write("No specific product recommendations at this time.")
    
    # Action Items
    st.subheader("🎯 Action Items")
    if ai_result['risk_score'] > 30:
        st.warning("**Immediate Actions Required:**")
        st.write("• Schedule financial counseling session")
        st.write("• Review expense patterns")
        st.write("• Set up automatic savings")
    elif ai_result['risk_score'] > 15:
        st.info("**Recommended Actions:**")
        st.write("• Monitor spending habits")
        st.write("• Increase savings rate")
        st.write("• Consider budget planning tools")
    else:
        st.success("**Maintenance Actions:**")
        st.write("• Continue current financial discipline")
        st.write("• Explore investment opportunities")
        st.write("• Consider premium banking services")

def main():
    st.set_page_config(
        page_title="Banker's Financial Insights Dashboard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏦 Banker's Financial Insights Dashboard")
    st.markdown("### Comprehensive Customer Analysis for Informed Banking Decisions")
    
    st.write('scikit-learn version:', sklearn.__version__)
    
    dashboard = FinancialDashboard()
    
    # Load BERT model and classifier at the top, so they are always defined
    try:
        bert_embedder = joblib.load('bert_embedder.pkl')
        bert_clf = joblib.load('bert_classifier.pkl')
    except Exception as e:
        bert_embedder = None
        bert_clf = None
        print(f"⚠️ Could not load BERT model: {e}")
    
    # Load enhanced model for transaction classification
    try:
        enhanced_model = joblib.load('transaction_classifier_enhanced.pkl')
        enhanced_vectorizer = joblib.load('transaction_vectorizer_enhanced.pkl')
        print("✅ Loaded enhanced model")
        use_enhanced_model = True
    except Exception as e:
        print(f"⚠️ Could not load enhanced model: {e}")
        use_enhanced_model = False
    
    # Define savings detection function inline
    def detect_savings_keywords(description):
        """Strong rule-based savings detection"""
        if not description:
            return False
        
        desc_upper = description.upper()
        
        # Highly specific savings keywords with word boundaries
        savings_patterns = [
            r'\bFD\b', r'\bFIXED DEPOSIT\b', r'\bSIP\b', r'\bMUTUAL FUND\b',
            r'\bPPF\b', r'\bNPS\b', r'\bRD\b', r'\bRECURRING DEPOSIT\b',
            r'\bLIC\b', r'\bPREMIUM\b', r'\bINVESTMENT\b', r'\bMATURITY\b',
            r'\bCLOSURE\b', r'\bTRANSFER TO.*DEPOSIT\b', r'\bAUTO SWEEP\b',
            r'\bSWEEP.*FD\b', r'\bGOAL.*SAVINGS\b', r'\bEDUCATION FUND\b',
            r'\bCHILD PLAN\b', r'\bRETIREMENT\b', r'\bPENSION\b'
        ]
        
        for pattern in savings_patterns:
            if re.search(pattern, desc_upper):
                return True
        return False
    
    # Sidebar for file upload
    st.sidebar.header("📁 Data Input")
    
    upload_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Upload Bank Statement", "Use Sample Data", "Use Cleaned Transactions CSV"]
    )
    
    customers = None
    if upload_option == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload customer data CSV file",
            type=['csv'],
            help="CSV should contain: customer_id, customer_name, monthly_income, monthly_expenses, savings_balance, investment_balance, total_debt, payment_history_score, credit_utilization_ratio, credit_age_months"
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.columns = [c.strip() for c in df.columns]
            if 'Type' in df.columns:
                df['Type'] = df['Type'].astype(str).str.strip().str.upper().replace({'DR': 'DEBIT', 'CR': 'CREDIT'})
            df['Description'] = df['Description'].astype(str)
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            required_cols = ['Amount', 'Type', 'Description']
            df = df.dropna(subset=required_cols)
            # Add Merchant_Category
            df['Merchant_Category'] = df['Description'].apply(extract_merchant_category)
            
            # Use enhanced model if available, otherwise fallback to BERT
            if use_enhanced_model:
                # Prepare text features for enhanced model
                df['text_features'] = df['Description'].fillna('') + ' ' + df['Type'].fillna('')
                X_vec = enhanced_vectorizer.transform(df['text_features'])
                model_labels = enhanced_model.predict(X_vec)
                print('Enhanced model predictions (before rules):', pd.Series(model_labels).value_counts())
                
                # Apply enhanced classification with savings detector
                final_labels = []
                label_sources = []
                for i, row in df.iterrows():
                    description = str(row.get('Description', ''))
                    transaction_type = str(row.get('Type', '')).upper()
                    ml_pred = model_labels[i]
                    
                    # 1. Check for savings using rule-based detector
                    if detect_savings_keywords(description):
                        final_labels.append('Savings')
                        label_sources.append('Rule-Savings-Detector')
                    # 2. Apply type-based overrides
                    elif transaction_type == 'CREDIT':
                        if ml_pred == 'EXPENSE':
                            final_labels.append('Income')
                            label_sources.append('Rule-Income-Override')
                        else:
                            final_labels.append(ml_pred)
                            label_sources.append('Enhanced-Model')
                    elif transaction_type == 'DEBIT':
                        if ml_pred == 'INCOME':
                            final_labels.append('Expense')
                            label_sources.append('Rule-Expense-Override')
                        else:
                            final_labels.append(ml_pred)
                            label_sources.append('Enhanced-Model')
                    else:
                        final_labels.append(ml_pred)
                        label_sources.append('Enhanced-Model')
                
                df['Predicted_Label'] = final_labels
                df['Label_Source'] = label_sources
            else:
                # When using bert_embedder.encode, check if bert_embedder is not None
                if bert_embedder is None:
                    st.error("BERT embedder model is missing. Please upload 'bert_embedder.pkl' to the app directory.")
                    return
                combined_text = (df['Type'].astype(str).str.upper().str.strip() + ': ' + df['Description'].astype(str).str.strip() + ' | ' + df['Merchant_Category'].astype(str).str.strip()).tolist()
                X = bert_embedder.encode(combined_text, show_progress_bar=False)
                model_labels = bert_clf.predict(X)
                print('BERT model predictions (before rules):', pd.Series(model_labels).value_counts())
                label_and_source = [hybrid_label_rule(row, model_labels[i]) for i, row in df.iterrows()]
                df['Predicted_Label'] = [ls[0] for ls in label_and_source]
                df['Label_Source'] = [ls[1] for ls in label_and_source]
            # --- FORCE TYPE-CONSISTENT OVERRIDE ---
            credit_mask = df['Type'].str.upper() == 'CREDIT'
            debit_mask = df['Type'].str.upper() == 'DEBIT'
            mask_credit_override = credit_mask & ~df['Predicted_Label'].isin(['Savings', 'Income'])
            df.loc[mask_credit_override, 'Predicted_Label'] = 'Income'
            df.loc[mask_credit_override, 'Label_Source'] = 'Rule-Income-Force'
            mask_debit_override = debit_mask & ~df['Predicted_Label'].isin(['Savings', 'Expense'])
            df.loc[mask_debit_override, 'Predicted_Label'] = 'Expense'
            df.loc[mask_debit_override, 'Label_Source'] = 'Rule-Expense-Force'
            print('Predicted_Label value counts:', df['Predicted_Label'].value_counts())
            print('Sample CREDIT rows after override:')
            print(df[df['Type'].str.upper() == 'CREDIT'][['Type', 'Description', 'Predicted_Label', 'Label_Source']].head(10))
            customers = df.to_dict('records')
            st.sidebar.success(f"✅ Loaded {len(customers)} transactions")
            dashboard.analyze_customers(customers)
    elif upload_option == "Upload Bank Statement":
        uploaded_files = st.sidebar.file_uploader(
            "Upload raw bank statement CSV file(s)",
            type=['csv'],
            accept_multiple_files=True,
            help="Supported: PNB, SBI, ICICI, APGB statement CSVs. You can upload multiple files at once."
        )
        customer_id = st.sidebar.text_input("Customer ID (optional)", value="AUTO")
        customer_name = st.sidebar.text_input("Customer Name (optional)", value="Unknown")
        customers = []
        failed_files = []
        parser = UniversalBankStatementParser()
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    df = pd.read_csv(uploaded_file)
                    df.columns = [c.strip() for c in df.columns]
                    df.columns = [c.strip().lower() for c in df.columns]
                    # Always rename to expected Title Case
                    rename_map = {
                        'date': 'Date',
                        'description': 'Description',
                        'amount': 'Amount',
                        'type': 'Type'
                    }
                    df = df.rename(columns=rename_map)
                    print('Columns after renaming:', df.columns)
                    if 'Type' in df.columns:
                        df['Type'] = df['Type'].astype(str).str.strip().str.upper().replace({'DR': 'DEBIT', 'CR': 'CREDIT'})
                        print('Type values after DR/CR conversion:', df['Type'].unique())
                        print(df[['Type', 'Description' if 'Description' in df.columns else 'remarks', 'Amount']].head(20))
                    required_cols = ['Date', 'Description', 'Amount', 'Type']
                    missing = [col for col in required_cols if col not in df.columns]
                    if not missing:
                        df = df.rename(columns={
                            'date': 'Date',
                            'description': 'Description',
                            'amount': 'Amount',
                            'type': 'Type'
                        })
                        std_df = df
                    else:
                        # --- Universal bank statement parser ---
                        cols = [c.strip().lower() for c in df.columns]
                        std_df = None
                        # ICICI/PNB style
                        if 'date' in cols and 'amount' in cols and ('remarks' in cols or 'description' in cols) and 'type' in cols:
                            desc_col = 'description' if 'description' in cols else 'remarks'
                            std_df = df.rename(columns={desc_col: 'Description', 'date': 'Date', 'amount': 'Amount', 'type': 'Type'})
                            std_df = std_df[['Date', 'Description', 'Amount', 'Type']]
                        # SBI style (all lowercase)
                        elif 'txn date' in cols and 'debit' in cols and 'credit' in cols and 'description' in cols:
                            # Melt Debit/Credit into Amount/Type
                            df['Type'] = df.apply(lambda row: 'DEBIT' if pd.notnull(row['debit']) and str(row['debit']).strip() != '' else ('CREDIT' if pd.notnull(row['credit']) and str(row['credit']).strip() != '' else ''), axis=1)
                            df['Amount'] = df.apply(lambda row: row['debit'] if row['Type'] == 'DEBIT' else row['credit'], axis=1)
                            std_df = df.rename(columns={'txn date': 'Date', 'description': 'Description'})
                            std_df = std_df[['Date', 'Description', 'Amount', 'Type']]
                        # APGB style (all lowercase)
                        elif 'post date' in cols and 'debit' in cols and 'credit' in cols and 'narration' in cols:
                            df['Type'] = df.apply(lambda row: 'DEBIT' if pd.notnull(row['debit']) and str(row['debit']).strip() != '' else ('CREDIT' if pd.notnull(row['credit']) and str(row['credit']).strip() != '' else ''), axis=1)
                            df['Amount'] = df.apply(lambda row: row['debit'] if row['Type'] == 'DEBIT' else row['credit'], axis=1)
                            std_df = df.rename(columns={'post date': 'Date', 'narration': 'Description'})
                            std_df = std_df[['Date', 'Description', 'Amount', 'Type']]
                        # Fallback: try to use any available date, amount, type, description columns
                        elif set(['date', 'description', 'amount', 'type']).issubset(set(cols)):
                            std_df = df.rename(columns={'date': 'Date', 'description': 'Description', 'amount': 'Amount', 'type': 'Type'})
                            std_df = std_df[['Date', 'Description', 'Amount', 'Type']]
                        else:
                            st.sidebar.error(f"❌ Could not auto-detect format for file: {uploaded_file.name}. Columns found: {df.columns.tolist()}")
                            continue
                        # Standardize types
                        std_df['Description'] = std_df['Description'].astype(str)
                        std_df['Amount'] = pd.to_numeric(std_df['Amount'], errors='coerce')
                        std_df['Type'] = std_df['Type'].astype(str).str.strip().str.upper().replace({'DR': 'DEBIT', 'CR': 'CREDIT'})
                        print('Unique values in Type column after cleaning:', std_df['Type'].unique())
                        print(std_df[['Type', 'Description', 'Amount']].head(20))
                        print('Rows before dropna:', len(std_df))
                        std_df = std_df.dropna(subset=['Description', 'Amount', 'Type'])
                        print('Rows after dropna:', len(std_df))
                    # Append to customers if valid
                    if std_df is not None and not std_df.empty:
                        # --- Robust cleaning for Amount column ---
                        if 'Amount' in std_df.columns:
                            # Remove rows where Amount is missing or blank
                            std_df = std_df[std_df['Amount'].notnull()]
                            std_df = std_df[std_df['Amount'].astype(str).str.strip() != '']
                            # Convert to float, coerce errors to NaN
                            std_df['Amount'] = pd.to_numeric(std_df['Amount'], errors='coerce')
                            # Drop rows where conversion failed (Amount is NaN)
                            std_df = std_df[std_df['Amount'].notnull()]
                            # Show error if all rows dropped
                            if std_df.empty:
                                st.sidebar.error(f"❌ All rows in {uploaded_file.name} had invalid or missing Amount values. Please check your file.")
                                continue
                        # --- Check for required columns before ML prediction ---
                        required_cols = ['Date', 'Description', 'Amount', 'Type']
                        missing_cols = [col for col in required_cols if col not in std_df.columns]
                        if std_df.empty or missing_cols:
                            st.sidebar.error(f"❌ {uploaded_file.name} is missing required columns: {missing_cols} or is empty. Skipping ML prediction.")
                            std_df['Predicted_Label'] = 'Unknown'
                            customers.append(std_df)
                            continue
                        # Add Merchant_Category
                        std_df['Merchant_Category'] = std_df['Description'].apply(extract_merchant_category)
                        
                        # Use enhanced model if available, otherwise fallback to BERT
                        if use_enhanced_model:
                            # Prepare text features for enhanced model
                            std_df['text_features'] = std_df['Description'].fillna('') + ' ' + std_df['Type'].fillna('')
                            X_vec = enhanced_vectorizer.transform(std_df['text_features'])
                            model_labels = enhanced_model.predict(X_vec)
                            print('Enhanced model predictions (before rules):', pd.Series(model_labels).value_counts())
                            
                            # Apply enhanced classification with savings detector
                            final_labels = []
                            label_sources = []
                            for i, row in std_df.iterrows():
                                description = str(row.get('Description', ''))
                                transaction_type = str(row.get('Type', '')).upper()
                                ml_pred = model_labels[i]
                                
                                # 1. Check for savings using rule-based detector
                                if detect_savings_keywords(description):
                                    final_labels.append('Savings')
                                    label_sources.append('Rule-Savings-Detector')
                                # 2. Apply type-based overrides
                                elif transaction_type == 'CREDIT':
                                    if ml_pred == 'EXPENSE':
                                        final_labels.append('Income')
                                        label_sources.append('Rule-Income-Override')
                                    else:
                                        final_labels.append(ml_pred)
                                        label_sources.append('Enhanced-Model')
                                elif transaction_type == 'DEBIT':
                                    if ml_pred == 'INCOME':
                                        final_labels.append('Expense')
                                        label_sources.append('Rule-Expense-Override')
                                    else:
                                        final_labels.append(ml_pred)
                                        label_sources.append('Enhanced-Model')
                                else:
                                    final_labels.append(ml_pred)
                                    label_sources.append('Enhanced-Model')
                            
                            std_df['Predicted_Label'] = final_labels
                            std_df['Label_Source'] = label_sources
                        else:
                            # When using bert_embedder.encode, check if bert_embedder is not None
                            if bert_embedder is None:
                                st.error("BERT embedder model is missing. Please upload 'bert_embedder.pkl' to the app directory.")
                                return
                            combined_text = (std_df['Type'].astype(str).str.upper().str.strip() + ': ' + std_df['Description'].astype(str).str.strip() + ' | ' + std_df['Merchant_Category'].astype(str).str.strip()).tolist()
                            X = bert_embedder.encode(combined_text, show_progress_bar=False)
                            model_labels = bert_clf.predict(X)
                            print('BERT model predictions (before rules):', pd.Series(model_labels).value_counts())
                            label_and_source = [hybrid_label_rule(row, model_labels[i]) for i, row in std_df.iterrows()]
                            std_df['Predicted_Label'] = [ls[0] for ls in label_and_source]
                            std_df['Label_Source'] = [ls[1] for ls in label_and_source]
                        # --- FORCE TYPE-CONSISTENT OVERRIDE ---
                        credit_mask = std_df['Type'].str.upper() == 'CREDIT'
                        debit_mask = std_df['Type'].str.upper() == 'DEBIT'
                        mask_credit_override = credit_mask & ~std_df['Predicted_Label'].isin(['Savings', 'Income'])
                        std_df.loc[mask_credit_override, 'Predicted_Label'] = 'Income'
                        std_df.loc[mask_credit_override, 'Label_Source'] = 'Rule-Income-Force'
                        mask_debit_override = debit_mask & ~std_df['Predicted_Label'].isin(['Savings', 'Expense'])
                        std_df.loc[mask_debit_override, 'Predicted_Label'] = 'Expense'
                        std_df.loc[mask_debit_override, 'Label_Source'] = 'Rule-Expense-Force'
                        print('Predicted_Label value counts:', std_df['Predicted_Label'].value_counts())
                        print('Sample CREDIT rows after override:')
                        print(std_df[std_df['Type'].str.upper() == 'CREDIT'][['Type', 'Description', 'Predicted_Label', 'Label_Source']].head(10))
                        customers.append(std_df)
                    else:
                        st.sidebar.error(f"❌ Parsed DataFrame is empty for file: {uploaded_file.name}")
                except Exception as e:
                    st.sidebar.error(f"❌ Could not read file: {uploaded_file.name}. Error: {e}")
                    failed_files.append(uploaded_file.name)
                    continue
            # After processing all files, analyze if any customers loaded
            if customers:
                st.sidebar.success(f"✅ Loaded {len(customers)} bank statements")
                # Concatenate all uploaded DataFrames
                all_df = pd.concat(customers, ignore_index=True)
                dashboard.ml_df = all_df  # So display_customer_overview can use Predicted_Label

                # Create a summary customer dict for analysis
                customer_dict = build_customer_profile_from_transactions(all_df)
                dashboard.analyze_customers([customer_dict])
            else:
                st.sidebar.error("❌ No valid bank statements loaded")
    elif upload_option == "Use Cleaned Transactions CSV":
        cleaned_csv_path = "transactions_for_labeling_cleaned.csv"
        if os.path.exists(cleaned_csv_path):
            df = pd.read_csv(cleaned_csv_path)
            df.columns = [c.strip() for c in df.columns]
            if 'Type' in df.columns:
                df['Type'] = df['Type'].astype(str).str.strip().str.upper().replace({'DR': 'DEBIT', 'CR': 'CREDIT'})
            df['Description'] = df['Description'].astype(str)
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            required_cols = ['Amount', 'Type', 'Description']
            df = df.dropna(subset=required_cols)
            # Add Merchant_Category
            df['Merchant_Category'] = df['Description'].apply(extract_merchant_category)
            # Use BERT embedder with Type+Description|Merchant_Category
            combined_text = (df['Type'].astype(str).str.upper().str.strip() + ': ' + df['Description'].astype(str).str.strip() + ' | ' + df['Merchant_Category'].astype(str).str.strip()).tolist()
            X = bert_embedder.encode(combined_text, show_progress_bar=False)
            model_labels = bert_clf.predict(X)
            print('Raw model predictions (before rules):', pd.Series(model_labels).value_counts())
            label_and_source = [hybrid_label_rule(row, model_labels[i]) for i, row in df.iterrows()]
            df['Predicted_Label'] = [ls[0] for ls in label_and_source]
            df['Label_Source'] = [ls[1] for ls in label_and_source]
            # --- FORCE TYPE-CONSISTENT OVERRIDE ---
            credit_mask = df['Type'].str.upper() == 'CREDIT'
            debit_mask = df['Type'].str.upper() == 'DEBIT'
            mask_credit_override = credit_mask & ~df['Predicted_Label'].isin(['Savings', 'Income'])
            df.loc[mask_credit_override, 'Predicted_Label'] = 'Income'
            df.loc[mask_credit_override, 'Label_Source'] = 'Rule-Income-Force'
            mask_debit_override = debit_mask & ~df['Predicted_Label'].isin(['Savings', 'Expense'])
            df.loc[mask_debit_override, 'Predicted_Label'] = 'Expense'
            df.loc[mask_debit_override, 'Label_Source'] = 'Rule-Expense-Force'
            print('Predicted_Label value counts:', df['Predicted_Label'].value_counts())
            print('Sample CREDIT rows after override:')
            print(df[df['Type'].str.upper() == 'CREDIT'][['Type', 'Description', 'Predicted_Label', 'Label_Source']].head(10))
            customers = df.to_dict('records')
            st.sidebar.success(f"✅ Loaded {len(customers)} transactions from cleaned CSV")
            dashboard.analyze_customers(customers)
        else:
            st.sidebar.error(f"❌ {cleaned_csv_path} not found. Please generate it first.")
    else:
        customers = get_all_sample_customers()
        st.sidebar.success(f"✅ Loaded {len(customers)} sample transactions")
        dashboard.analyze_customers(customers)
    
    if dashboard.analysis_results:
        # Create tabs for different analysis views
        tab_names = [
            'Customer Overview',
            'Credit Score Analysis',
            'Financial Health',
            'Lending Recommendations',
            'Insights & Segmentation',
            'ML Prediction',
            'AI Opportunity & Risk Analyzer'  # Removed 'Export' tab
        ]
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            dashboard.display_customer_overview()
        
        with tabs[1]:
            dashboard.display_credit_score_analysis()
        
        with tabs[2]:
            dashboard.display_financial_health_analysis()
        
        with tabs[3]:
            dashboard.display_lending_recommendations()
        
        with tabs[4]:
            dashboard.display_insights_and_segmentation()
        
        with tabs[5]:
            dashboard.display_ml_prediction_tab()
        
        with tabs[6]:
            # Use the most recently loaded/processed DataFrame
            if hasattr(dashboard, 'ml_df') and dashboard.ml_df is not None:
                ai_opportunity_risk_analyzer(dashboard.ml_df)
            else:
                st.info('No transaction data available for AI analysis.')

if __name__ == "__main__":
    main() 
