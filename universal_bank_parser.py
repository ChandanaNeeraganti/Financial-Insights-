import pandas as pd
import numpy as np
import csv
import io
import tempfile
from datetime import datetime
import os
from utils.bank_statement_parser import BankStatementParser

print("UNIVERSAL PARSER FILE PATH:", os.path.abspath(__file__))

class UniversalBankStatementParser:
    """
    Robust parser for Indian bank statement CSVs (ICICI, SBI, PNB, APGB, etc.)
    - Auto-detects header row
    - Maps common column name variations
    - Standardizes output: Date, Description, Amount, Type, Balance, Mode
    - Handles file paths and file-like objects
    """
    def __init__(self):
        pass

    def parse(self, file_input):
        print("DEBUG: Entered UniversalBankStatementParser.parse")
        # Support both file path and file-like object
        if isinstance(file_input, str):
            df = pd.read_csv(file_input)
        else:
            file_input.seek(0)
            df = pd.read_csv(file_input)
        print("DEBUG: Raw columns in uploaded file:", df.columns.tolist())
        df.columns = [c.strip().lower().replace('txn date', 'date').replace('description', 'details').replace('ref no./cheque no.', 'ref no./cheque no').replace('  ', ' ').replace('.', '').replace('value date', 'value date').replace('debit', 'debit').replace('credit', 'credit').replace('balance', 'balance') for c in df.columns]
        print("DEBUG: Universal parser columns:", df.columns.tolist())
        print("DEBUG: Universal parser head:\n", df.head())
        # Only use the ICICI parser if the columns match
        required_cols = set(['date', 'description', 'amount', 'type'])
        apgb_cols = set(['post date', 'value date', 'narration', 'cheque details', 'debit', 'credit', 'balance'])
        pnb_cols = set(['date', 'instrument id', 'amount', 'type', 'balance', 'remarks'])
        sbi_cols = set(['date', 'value date', 'details', 'ref no/cheque no', 'debit', 'credit', 'balance'])
        if required_cols.issubset(df.columns):
            df['type'] = df['type'].replace({'dr': 'DEBIT', 'cr': 'CREDIT'})
            result = BankStatementParser()._parse_icici(df)
            print("DEBUG: DataFrame returned to dashboard:\n", result.head())
            print("DEBUG: DataFrame columns:", result.columns.tolist())
            print("DEBUG: DataFrame shape:", result.shape)
            return result
        elif apgb_cols.issubset(df.columns):
            print("DEBUG: Detected APGB format, calling _parse_apgb")
            parsed = BankStatementParser()._parse_apgb(df)
            # Convert parsed dict to DataFrame with required columns
            # _parse_apgb returns a dict of summary fields, so we need to reconstruct transactions
            # Instead, let's convert APGB to the required transaction format
            # Map APGB to standard transaction DataFrame
            txns = []
            for i, row in df.iterrows():
                if not pd.isna(row['debit']) and row['debit'] != 0:
                    txns.append({
                        'Date': row['post date'],
                        'Description': row['narration'],
                        'Amount': row['debit'],
                        'Type': 'DEBIT'
                    })
                if not pd.isna(row['credit']) and row['credit'] != 0:
                    txns.append({
                        'Date': row['post date'],
                        'Description': row['narration'],
                        'Amount': row['credit'],
                        'Type': 'CREDIT'
                    })
            result = pd.DataFrame(txns)
            print("DEBUG: APGB-mapped DataFrame to dashboard:\n", result.head())
            print("DEBUG: DataFrame columns:", result.columns.tolist())
            print("DEBUG: DataFrame shape:", result.shape)
            return result
        elif pnb_cols.issubset(df.columns):
            print("DEBUG: Detected PNB format, mapping to standard format")
            df['type'] = df['type'].replace({'dr': 'DEBIT', 'cr': 'CREDIT'})
            txns = []
            for i, row in df.iterrows():
                if row['type'] == 'DEBIT' and not pd.isna(row['amount']) and row['amount'] != 0:
                    txns.append({
                        'Date': row['date'],
                        'Description': row['remarks'],
                        'Amount': float(str(row['amount']).replace(',', '')),
                        'Type': 'DEBIT'
                    })
                if row['type'] == 'CREDIT' and not pd.isna(row['amount']) and row['amount'] != 0:
                    txns.append({
                        'Date': row['date'],
                        'Description': row['remarks'],
                        'Amount': float(str(row['amount']).replace(',', '')),
                        'Type': 'CREDIT'
                    })
            result = pd.DataFrame(txns)
            print("DEBUG: PNB-mapped DataFrame to dashboard:\n", result.head())
            print("DEBUG: DataFrame columns:", result.columns.tolist())
            print("DEBUG: DataFrame shape:", result.shape)
            return result
        elif sbi_cols.issubset(df.columns):
            print("DEBUG: Detected SBI format, mapping to standard format")
            import re
            for col in ['debit', 'credit']:
                cleaned = df[col].astype(str).str.replace(',', '').str.strip()
                non_numeric = cleaned[~cleaned.str.match(r'^-?\d*\.?\d+$') & (cleaned != '0')]
                if not non_numeric.empty:
                    print(f"Non-numeric values in {col} column:", non_numeric.unique())
                df[col] = cleaned.replace({'': '0', 'nan': '0', 'None': '0', 'NaN': '0', '\xa0': '0'})
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except Exception as e:
                    print(f"ERROR converting {col} to numeric! First 10 values: {df[col].head(10).tolist()}")
                    raise
                df[col] = df[col].fillna(0)
            txns = []
            for i, row in df.iterrows():
                if not pd.isna(row['debit']) and row['debit'] != 0:
                    txns.append({
                        'Date': row['date'],
                        'Description': row['details'],
                        'Amount': float(row['debit']),
                        'Type': 'DEBIT'
                    })
                if not pd.isna(row['credit']) and row['credit'] != 0:
                    txns.append({
                        'Date': row['date'],
                        'Description': row['details'],
                        'Amount': float(row['credit']),
                        'Type': 'CREDIT'
                    })
            result = pd.DataFrame(txns)
            print("DEBUG: SBI-mapped DataFrame to dashboard:\n", result.head())
            print("DEBUG: DataFrame columns:", result.columns.tolist())
            print("DEBUG: DataFrame shape:", result.shape)
            return result
        else:
            print("ERROR: Uploaded file does not have required columns for ICICI or APGB parser.")
            print("Columns found:", df.columns.tolist())
            raise ValueError("Uploaded file does not have required columns: ['Date', 'Description', 'Amount', 'Type']")

# Example usage:
# parser = UniversalBankStatementParser()
# df = parser.parse('chandana.csv')
# print(df.head()) 