import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import io
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Add project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load enhanced model for transaction classification
try:
    enhanced_model = pickle.load(open('transaction_classifier_enhanced.pkl', 'rb'))
    enhanced_vectorizer = pickle.load(open('transaction_vectorizer_enhanced.pkl', 'rb'))
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

def enhanced_classify_transactions(df):
    """Enhanced transaction classification using both ML model and rule-based savings detection"""
    if enhanced_model is None or enhanced_vectorizer is None:
        st.error("Enhanced model not loaded. Please ensure transaction_classifier_enhanced.pkl and transaction_vectorizer_enhanced.pkl exist.")
        return df
    
    # Prepare text features
    df['text_features'] = df['Description'].fillna('') + ' ' + df['Type'].fillna('')
    
    # Get ML predictions
    X_vec = enhanced_vectorizer.transform(df['text_features'])
    ml_predictions = enhanced_model.predict(X_vec)
    
    # Apply rule-based savings detection
    final_labels = []
    label_sources = []
    
    for i, row in df.iterrows():
        description = str(row.get('Description', ''))
        transaction_type = str(row.get('Type', '')).upper()
        ml_pred = ml_predictions[i]
        
        # 1. Check for savings using rule-based detector
        if detect_savings_keywords(description):
            final_labels.append('SAVINGS')
            label_sources.append('Rule-Savings-Detector')
        # 2. Apply type-based overrides
        elif transaction_type == 'CREDIT':
            if ml_pred == 'EXPENSE':
                final_labels.append('INCOME')
                label_sources.append('Rule-Income-Override')
            else:
                final_labels.append(ml_pred)
                label_sources.append('Enhanced-Model')
        elif transaction_type == 'DEBIT':
            if ml_pred == 'INCOME':
                final_labels.append('EXPENSE')
                label_sources.append('Rule-Expense-Override')
            else:
                final_labels.append(ml_pred)
                label_sources.append('Enhanced-Model')
        else:
            final_labels.append(ml_pred)
            label_sources.append('Enhanced-Model')
    
    df['Predicted_Label'] = final_labels
    df['Label_Source'] = label_sources
    
    return df

def save_user_corrections(df, corrections):
    """Save user corrections for model improvement"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"user_corrections_{timestamp}.csv"
    
    # Create corrected dataset
    corrected_df = df.copy()
    for idx, new_label in corrections.items():
        corrected_df.loc[idx, 'Predicted_Label'] = new_label
        corrected_df.loc[idx, 'Label_Source'] = 'User-Correction'
    
    # Save to file
    corrected_df.to_csv(filename, index=False)
    st.success(f"✅ User corrections saved to {filename}")
    return filename

def main():
    st.set_page_config(
        page_title="Continuous Learning Financial Dashboard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏦 Continuous Learning Financial Dashboard")
    st.markdown("### Advanced Transaction Classification with User Feedback Learning")
    
    # Sidebar for file upload
    st.sidebar.header("📁 Data Input")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload transaction CSV file",
        type=['csv'],
        help="CSV should contain: Date, Description, Amount, Type columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df)} transactions")
            
            # Standardize column names
            df.columns = [c.strip() for c in df.columns]
            rename_map = {
                'date': 'Date', 'description': 'Description', 'remarks': 'Description',
                'narration': 'Description', 'amount': 'Amount', 'type': 'Type'
            }
            df = df.rename(columns=rename_map)
            
            # Clean data
            if 'Type' in df.columns:
                df['Type'] = df['Type'].astype(str).str.strip().str.upper().replace({'DR': 'DEBIT', 'CR': 'CREDIT'})
            df['Description'] = df['Description'].fillna('').astype(str)
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            
            # Remove rows with missing critical data
            required_cols = ['Amount', 'Type', 'Description']
            df = df.dropna(subset=required_cols)
            
            # Classify transactions using enhanced model
            df = enhanced_classify_transactions(df)
            
            # Display results
            st.header("📊 Transaction Analysis Results")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                income_total = df[df['Predicted_Label'] == 'INCOME']['Amount'].sum()
                st.metric("💰 Total Income", f"₹{income_total:,.2f}")
            
            with col2:
                expense_total = df[df['Predicted_Label'] == 'EXPENSE']['Amount'].sum()
                st.metric("💸 Total Expenses", f"₹{expense_total:,.2f}")
            
            with col3:
                savings_total = df[df['Predicted_Label'] == 'SAVINGS']['Amount'].sum()
                st.metric("🏦 Total Savings", f"₹{savings_total:,.2f}")
            
            with col4:
                net_flow = income_total - expense_total - savings_total
                st.metric("📈 Net Flow", f"₹{net_flow:,.2f}")
            
            # User Correction Interface
            st.header("🔧 Correct Classifications")
            st.write("Review and correct any misclassified transactions:")
            
            corrections = {}
            
            # Show transactions by category for easy review
            for label in ['SAVINGS', 'INCOME', 'EXPENSE']:
                label_df = df[df['Predicted_Label'] == label].copy()
                if not label_df.empty:
                    st.subheader(f"{label} Transactions")
                    
                    for idx, row in label_df.iterrows():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"**{row['Description']}** - ₹{row['Amount']:,.2f}")
                        
                        with col2:
                            current_label = row['Predicted_Label']
                            st.write(f"Current: {current_label}")
                        
                        with col3:
                            # Correction dropdown
                            options = ['SAVINGS', 'INCOME', 'EXPENSE']
                            if current_label in options:
                                options.remove(current_label)
                            
                            if options:
                                new_label = st.selectbox(
                                    f"Correct to:",
                                    options=['Keep Current'] + options,
                                    key=f"correction_{idx}"
                                )
                                
                                if new_label != 'Keep Current':
                                    corrections[idx] = new_label
            
            # Save corrections button
            if corrections:
                if st.button("💾 Save Corrections for Model Improvement"):
                    filename = save_user_corrections(df, corrections)
                    st.info(f"📝 Corrections saved! Run 'python retrain_with_user_data.py' to improve the model.")
            
            # Show sample transactions
            st.subheader("🔍 Sample Transactions by Category")
            
            for label in ['INCOME', 'EXPENSE', 'SAVINGS']:
                label_df = df[df['Predicted_Label'] == label].head(10)
                if not label_df.empty:
                    st.write(f"**{label} Transactions:**")
                    st.dataframe(label_df[['Date', 'Description', 'Amount', 'Type', 'Label_Source']])
            
            # Debug information
            with st.expander("🔧 Debug Information"):
                st.write("**Enhanced Model Status:**", "✅ Loaded" if enhanced_model else "❌ Not Loaded")
                st.write("**Total Transactions Processed:**", len(df))
                
                # Show savings detection examples
                st.write("**Savings Detection Examples:**")
                savings_examples = df[df['Predicted_Label'] == 'SAVINGS']['Description'].head(5).tolist()
                for example in savings_examples:
                    st.write(f"  - {example}")
                
                # Show label distribution
                st.write("**Label Distribution:**")
                st.write(df['Predicted_Label'].value_counts())
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main() 