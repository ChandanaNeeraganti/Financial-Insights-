import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("apgb.csv", skip_blank_lines=True, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df["Value Date"] = pd.to_datetime(df["Value Date"], dayfirst=True, errors="coerce")
    df["Debit"] = pd.to_numeric(
        df["Debit"].astype(str).str.replace(r"[^\d\.-]", "", regex=True),
        errors="coerce",
    ).fillna(0)
    df["Credit"] = pd.to_numeric(
        df["Credit"].astype(str).str.replace(r"[^\d\.-]", "", regex=True),
        errors="coerce",
    ).fillna(0)
    df = df.dropna(subset=["Value Date"])
    df["Date"] = df["Value Date"]
    return df


df = load_data()
st.title("💳 Personal Finance Dashboard")

# --- Sidebar ---
st.sidebar.title("Filters")
start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())
df = df[
    (df["Date"] >= pd.to_datetime(start_date))
    & (df["Date"] <= pd.to_datetime(end_date))
]

# --- Monthly Trends ---
st.subheader("📆 Monthly Spending Trends")
df["Month"] = df["Date"].dt.to_period("M")
monthly_summary = df.groupby("Month")[["Debit", "Credit"]].sum()

st.bar_chart(monthly_summary)

# --- Overspending Days ---
st.subheader("🚨 Overspending Days")
threshold = df["Debit"].quantile(0.90)
overspending_days = df[df["Debit"] > threshold][["Date", "Debit"]].sort_values("Date")
st.dataframe(overspending_days)
st.caption(f"Threshold: Top 10% spending days (₹{threshold:,.2f})")

# --- Predictive Feature Engineering ---
st.subheader("📈 Predictive Features")
df = df.sort_values("Date")
df["Net"] = df["Credit"] - df["Debit"]
df["Cumulative Balance"] = df["Net"].cumsum()
df["7-day Avg Debit"] = df["Debit"].rolling(window=7).mean()

st.line_chart(df.set_index("Date")[["Cumulative Balance", "7-day Avg Debit"]])

# --- Raw Data Viewer ---
with st.expander("🗃️ View Raw Data"):
    st.dataframe(
        df[["Date", "Debit", "Credit", "Net", "Cumulative Balance", "7-day Avg Debit"]]
    )

st.success("✅ Dashboard loaded successfully!")
