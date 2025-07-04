import pandas as pd
import matplotlib.pyplot as plt

file_path = "apgb.csv"

# Load with header detection (as you've already done)
df = pd.read_csv(
    file_path,
    skiprows=lambda x: x
    < 0,  # placeholder—use the same skiprows logic you've established
    header=0,
    engine="python",
    sep=",",
    thousands=",",
    skip_blank_lines=True,
    on_bad_lines="skip",
)

# Clean column names
df.columns = df.columns.str.strip()
print("🔍 Columns loaded:", df.columns.tolist())

# ✅ Ensure required columns are present
required = ["Value Date", "Debit", "Credit"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}")

# 💾 Convert dates
df["Date"] = pd.to_datetime(df["Value Date"], dayfirst=True, errors="raise")

# 🌗 Create a unified 'Amount' and 'Type' column
df["Debit"] = pd.to_numeric(
    df["Debit"].astype(str).str.replace(r"[^\d\.-]", "", regex=True), errors="coerce"
).fillna(0)
df["Credit"] = pd.to_numeric(
    df["Credit"].astype(str).str.replace(r"[^\d\.-]", "", regex=True), errors="coerce"
).fillna(0)

# Summarize totals per day
daily_summary = df.groupby("Date")[["Debit", "Credit"]].sum().sort_index()

# 🖼 Plot results
plt.figure(figsize=(10, 5))
plt.plot(
    daily_summary.index, daily_summary["Debit"], label="Debit", color="red", marker="o"
)
plt.plot(
    daily_summary.index,
    daily_summary["Credit"],
    label="Credit",
    color="green",
    marker="o",
)
plt.title("Daily Debit and Credit Amounts")
plt.xlabel("Date")
plt.ylabel("Amount (₹)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 💾 Export summary
daily_summary.to_csv("daily_debit_credit_summary.csv")
print("✅ Summary saved to daily_debit_credit_summary.csv")
