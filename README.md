# 🏦 Banker's Financial Insights Model

A comprehensive financial analysis system designed specifically for bankers to gain clear insights into customer financial health and make informed lending decisions.

## 🎯 Overview

This model provides bankers with:

- **Credit Score Analysis**: Calculate and categorize customer credit scores
- **Risk Assessment**: Evaluate customer risk levels with detailed risk factors
- **Financial Health Indicators**: Comprehensive financial health scoring
- **Lending Recommendations**: Automated loan approval and amount recommendations
- **Interactive Dashboard**: Visual analytics and customer portfolio management
- **CSV Data Processing**: Handle customer data from CSV files with validation

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
# Install dependencies
pip install -r requirements.txt
```

### 2. Create Sample Data

```bash
# Create a sample CSV file with 10 customers
python main.py --create-sample sample_customers.csv

# Or create with custom number of customers
python main.py --create-sample my_data.csv --num-customers 50
```

### 3. Run Analysis

```bash
# Analyze customer data from CSV
python main.py --csv sample_customers.csv --output results.xlsx
```

### 4. Launch Interactive Dashboard

```bash
# Start the Streamlit dashboard
streamlit run dashboard/financial_dashboard.py
```

## 📊 CSV Input Format

### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `customer_id` | Unique identifier | CUST001 |
| `customer_name` | Full name | John Smith |
| `monthly_income` | Gross monthly income (USD) | 8500 |
| `monthly_expenses` | Total monthly expenses (USD) | 5200 |
| `savings_balance` | Current savings balance (USD) | 25000 |
| `investment_balance` | Investment portfolio value (USD) | 45000 |
| `total_debt` | Total outstanding debt (USD) | 180000 |
| `payment_history_score` | Payment history (0.0-1.0) | 0.95 |
| `credit_utilization_ratio` | Credit utilization (0.0-1.0) | 0.25 |
| `credit_age_months` | Credit history length (months) | 84 |

### Optional Columns

| Column | Description | Example |
|--------|-------------|---------|
| `employment_years` | Years of employment | 8 |
| `loan_history` | Number of previous loans | 2 |
| `bank_accounts` | Number of bank accounts | 2 |
| `credit_cards` | Number of credit cards | 3 |
| `mortgage_balance` | Outstanding mortgage (USD) | 150000 |
| `auto_loan_balance` | Outstanding auto loan (USD) | 25000 |
| `student_loan_balance` | Outstanding student loan (USD) | 5000 |
| `other_debt` | Other outstanding debt (USD) | 0 |

### CSV Template

```csv
customer_id,customer_name,monthly_income,monthly_expenses,savings_balance,investment_balance,total_debt,payment_history_score,credit_utilization_ratio,credit_age_months,employment_years,loan_history,bank_accounts,credit_cards,mortgage_balance,auto_loan_balance,student_loan_balance,other_debt
CUST001,John Smith,8500,5200,25000,45000,180000,0.95,0.25,84,8,2,2,3,150000,25000,5000,0
```

## 🔍 Analysis Features

### 1. Credit Score Calculation
- **Range**: 300-850
- **Factors**: Income, payment history, credit utilization, debt-to-income ratio, savings rate, credit age
- **Categories**: Poor (300-579), Fair (580-669), Good (670-739), Very Good (740-799), Exceptional (800-850)

### 2. Risk Assessment
- **Risk Levels**: Low, Medium, High, Very High
- **Risk Factors**: Credit score, debt-to-income ratio, income stability, payment history
- **Risk Score**: Numerical score (0-10) for detailed risk evaluation

### 3. Financial Health Indicators
- **Emergency Fund Ratio**: Months of expenses covered by savings
- **Savings Rate**: Percentage of income saved monthly
- **Debt-to-Income Ratio**: Total debt relative to income
- **Investment Ratio**: Investment balance relative to income
- **Net Worth**: Total assets minus total debt
- **Health Score**: 0-100 comprehensive financial health score

### 4. Lending Recommendations
- **Loan Approval**: Automated approval/rejection based on criteria
- **Recommended Amount**: Calculated based on income and debt-to-income ratio
- **Interest Rate Range**: Suggested rates based on credit score
- **Loan Terms**: Standard or restricted terms based on financial health
- **Conditions**: Specific requirements for loan approval
- **Risk Mitigation**: Strategies to reduce lending risk

## 📈 Dashboard Features

### Interactive Analytics
- **Portfolio Overview**: High-level customer portfolio metrics
- **Credit Score Analysis**: Distribution and rating analysis
- **Financial Health Analysis**: Health indicators and trends
- **Lending Decisions**: Approval/rejection summary with reasons
- **Individual Customer Analysis**: Detailed customer profiles

### Export Capabilities
- **Excel Reports**: Comprehensive multi-sheet reports
- **Summary Sheet**: Key metrics and recommendations
- **Risk Analysis**: Detailed risk assessment
- **Lending Recommendations**: Complete lending decisions

## 🛠️ Usage Examples

### Command Line Interface

```bash
# Show CSV template format
python main.py --template

# Create sample data
python main.py --create-sample customers.csv --num-customers 25

# Run analysis with custom output
python main.py --csv customers.csv --output analysis_results.xlsx

# Launch dashboard
python main.py --dashboard
```

### Python API

```python
from models.financial_models import CustomerFinancialAnalyzer
from utils.csv_processor import CSVProcessor

# Initialize components
analyzer = CustomerFinancialAnalyzer()
processor = CSVProcessor()

# Process CSV data
customers = processor.process_csv('customers.csv')

# Analyze customers
for customer in customers:
    summary = analyzer.create_customer_summary(customer)
    print(f"Customer: {summary['customer_name']}")
    print(f"Credit Score: {summary['credit_score']}")
    print(f"Risk Level: {summary['risk_assessment']['risk_level']}")
    print(f"Loan Approved: {summary['lending_recommendations']['loan_approval']}")
```

## 📋 Output Reports

### Excel Report Structure

1. **Summary Sheet**
   - Customer information and key metrics
   - Credit scores and ratings
   - Risk levels and financial health scores
   - Lending recommendations

2. **Risk Analysis Sheet**
   - Detailed risk assessment
   - Risk factors and scores
   - Risk mitigation strategies

3. **Lending Recommendations Sheet**
   - Loan approval decisions
   - Recommended amounts and interest rates
   - Conditions and terms

## 🎯 Banker Benefits

### Decision Support
- **Clear Risk Assessment**: Understand customer risk levels at a glance
- **Automated Recommendations**: Get lending suggestions with reasoning
- **Financial Health Insights**: Comprehensive view of customer financial status
- **Portfolio Management**: Manage multiple customers efficiently

### Efficiency
- **Batch Processing**: Analyze multiple customers simultaneously
- **Standardized Analysis**: Consistent evaluation criteria
- **Export Capabilities**: Generate reports for documentation
- **Interactive Interface**: Easy-to-use dashboard for daily operations

### Compliance
- **Transparent Criteria**: Clear lending decision rationale
- **Documentation**: Comprehensive audit trail
- **Consistent Standards**: Uniform evaluation across customers
- **Risk Mitigation**: Built-in risk management strategies

## 🔧 Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models
- **plotly**: Interactive visualizations
- **streamlit**: Web dashboard interface
- **openpyxl**: Excel file handling

### Architecture
- **Modular Design**: Separate components for different functions
- **Extensible**: Easy to add new analysis features
- **Configurable**: Adjustable parameters for different requirements
- **Scalable**: Handle large customer datasets

## 📞 Support

For questions or issues:
1. Check the CSV template format
2. Verify all required columns are present
3. Ensure data types are correct
4. Review validation warnings

## 🔄 Updates

The model can be enhanced with:
- Additional financial indicators
- Custom scoring algorithms
- Integration with external data sources
- Advanced machine learning models
- Real-time data processing

---

**Built for bankers, by financial analysts, to provide clarity and confidence in customer decisions.** 