import pandas as pd
import numpy as np
import os
import pdfplumber
import tabula
import pytesseract
from PIL import Image
import io
import csv

def parse_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.csv':
        # Robustly find the header row (look for 'Date' or 'Txn Date' in the first few lines)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = list(csv.reader(f))
        header_row = 0
        for i, row in enumerate(lines):
            if any(x.strip().lower() in [col.strip().lower() for col in row] for x in ['date', 'txn date', 'post date']):
                header_row = i
                break
        # Reconstruct CSV for pandas
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='') as tmp:
            writer = csv.writer(tmp)
            for row in lines[header_row:]:
                writer.writerow(row)
            tmp_path = tmp.name
        df = pd.read_csv(tmp_path)
    elif ext == '.xlsx':
        df = pd.read_excel(file_path)
    elif ext == '.pdf':
        try:
            dfs = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            df = pd.concat(dfs)
        except Exception:
            with pdfplumber.open(file_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() or ''
            df = custom_text_to_df(text)
    else:
        raise ValueError('Unsupported file type')
    return normalize_columns(df)

def normalize_columns(df):
    col_map = {
        'date': ['Date', 'Txn Date', 'Post Date', 'Value Date'],
        'desc': ['Description', 'Narration', 'Details', 'Particulars', 'Remarks'],
        'debit': ['Debit', 'Withdrawal Amt.', 'Withdrawals'],
        'credit': ['Credit', 'Deposit Amt.', 'Deposits'],
        'amount': ['Amount', 'Transaction Amount'],
        'type': ['Type', 'Transaction Type'],
        'balance': ['Balance', 'Available Balance', 'Closing Balance'],
    }
    def find_col(possibles):
        for p in possibles:
            for col in df.columns:
                if col.strip().lower() == p.strip().lower():
                    return col
        return None
    df.columns = [c.strip() for c in df.columns]
    date_col = find_col(col_map['date'])
    desc_col = find_col(col_map['desc'])
    debit_col = find_col(col_map['debit'])
    credit_col = find_col(col_map['credit'])
    amount_col = find_col(col_map['amount'])
    type_col = find_col(col_map['type'])
    balance_col = find_col(col_map['balance'])
    # Normalize date formats
    if date_col:
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True, infer_datetime_format=True)
        # If still null, try with yearfirst
        if df['Date'].isnull().all():
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce', yearfirst=True, infer_datetime_format=True)
    else:
        df['Date'] = pd.NaT
    df['Description'] = df[desc_col] if desc_col else ''
    if debit_col and credit_col:
        df[debit_col] = df[debit_col].replace(r'^\s*$', '0', regex=True).replace(np.nan, '0')
        df[credit_col] = df[credit_col].replace(r'^\s*$', '0', regex=True).replace(np.nan, '0')
        df[debit_col] = pd.to_numeric(df[debit_col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)
        df[credit_col] = pd.to_numeric(df[credit_col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)
        df['Amount'] = df[credit_col] - df[debit_col]
        df['Type'] = df.apply(lambda x: 'Credit' if x[credit_col] > 0 else ('Debit' if x[debit_col] > 0 else ''), axis=1)
    elif amount_col and type_col:
        df[amount_col] = pd.to_numeric(df[amount_col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)
        df[type_col] = df[type_col].str.upper()
        df['Amount'] = df.apply(lambda x: x[amount_col] if 'CR' in str(x[type_col]) else -x[amount_col], axis=1)
        df['Type'] = df[type_col].apply(lambda x: 'Credit' if 'CR' in str(x) else 'Debit')
    else:
        df['Amount'] = 0
        df['Type'] = ''
    if balance_col:
        df['Balance'] = pd.to_numeric(df[balance_col].astype(str).str.replace('Cr|Dr', '', regex=True).str.replace(',', '').str.strip(), errors='coerce').fillna(0)
    else:
        df['Balance'] = np.nan
    df['Mode'] = df['Description'].astype(str).str.extract(r'(UPI|IMPS|NEFT|RTGS|ATM|CHEQUE|CARD|CASH)', expand=False)
    std_df = df[['Date', 'Description', 'Amount', 'Type', 'Balance', 'Mode']].dropna(subset=['Date'])
    return std_df 