

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -----------------------------
# 0️⃣ Paths
# -----------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = SCRIPT_DIR  


# For demo, using synthetic CSVs
BOC_FILE = os.path.join(DATA_DIR, "boc_interest_rates.csv")
STATCAN_FILE = os.path.join(DATA_DIR, "statcan_household_debt.csv")
PATTERNS_FILE = os.path.join(DATA_DIR, "canadian_debt_patterns.csv")

# -----------------------------
# 1️⃣ Load CSVs (synthetic/demo)
# -----------------------------
def load_csv_with_date(path, date_keywords=["date"], sep=",", skiprows=0):
    df = pd.read_csv(path, sep=sep, skiprows=skiprows)
    df.columns = df.columns.str.strip().str.lower()
    date_col = next((c for c in df.columns if any(k in c for k in date_keywords)), None)
    if date_col is None:
        df["Date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="M")
    else:
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)

boc = load_csv_with_date(BOC_FILE)
patterns = load_csv_with_date(PATTERNS_FILE)

# -----------------------------
# 2️⃣ Simulate StatCan debt data
# -----------------------------
np.random.seed(42)
n_samples = len(boc)
avg_balance = np.random.uniform(2000, 15000, n_samples)
num_debts = np.random.randint(1, 5, n_samples)
monthly_budget = np.random.uniform(500, 3000, n_samples)
avg_interest_rate = np.random.uniform(5, 25, n_samples)
market_sentiment = np.random.normal(0, 1, n_samples)

# -----------------------------
# 3️⃣ Compute Snowball & Avalanche Interest
# -----------------------------
def compute_interest(balance_list, rate_list, min_payment, strategy="snowball"):
    balances = balance_list.copy()
    rates = rate_list.copy()
    n = len(balances)
    total_interest = [0]*n
    month = 0
    while any(b > 0 for b in balances):
        remaining_budget = sum(min_payment)
        # Pay minimums
        for i in range(n):
            if balances[i] <= 0:
                continue
            interest = balances[i]*rates[i]/100/12
            total_interest[i] += interest
            pay = min(min_payment[i], balances[i]+interest)
            balances[i] -= (pay - interest)
            remaining_budget -= pay

        # Extra payment
        if strategy == "snowball":
            order = np.argsort(balances)
        else:
            order = np.argsort(rates)[::-1]
        for i in order:
            if balances[i] <= 0 or remaining_budget <=0:
                continue
            extra = min(remaining_budget, balances[i])
            balances[i] -= extra
            remaining_budget -= extra
        month +=1
    return sum(total_interest)

# For demo, create 3 debts per sample
snowball_total = []
avalanche_total = []

for i in range(n_samples):
    balances = [np.random.uniform(500, avg_balance[i]) for _ in range(num_debts[i])]
    rates = [np.random.uniform(5, avg_interest_rate[i]) for _ in range(num_debts[i])]
    min_payment = [max(50, b*0.03) for b in balances]
    snowball_total.append(compute_interest(balances, rates, min_payment, "snowball"))
    avalanche_total.append(compute_interest(balances, rates, min_payment, "avalanche"))

# -----------------------------
# 4️⃣ Assign Best Strategy based on actual interest saved
# -----------------------------
best_strategy = ["Avalanche" if a < s else "Snowball" for s,a in zip(snowball_total, avalanche_total)]
df = pd.DataFrame({
    "avg_interest_rate": avg_interest_rate,
    "avg_balance": avg_balance,
    "num_debts": num_debts,
    "monthly_budget": monthly_budget,
    "market_sentiment": market_sentiment,
    "snowball_interest": snowball_total,
    "avalanche_interest": avalanche_total,
    "best_strategy": best_strategy
})

print("Class distribution:\n", df["best_strategy"].value_counts())

# -----------------------------
# 5️⃣ Train ML Model
# -----------------------------
features = ["avg_interest_rate", "avg_balance", "num_debts", "monthly_budget", "market_sentiment"]
X = df[features]
y = (df["best_strategy"]=="Avalanche").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 6️⃣ Evaluate
# -----------------------------
y_pred = model.predict(X_test_scaled)
print(f"✅ Model accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=["Snowball","Avalanche"]))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Snowball","Avalanche"]).plot(cmap=plt.cm.Blues)
plt.show()

plt.figure(figsize=(6,4))
plt.bar(features, model.feature_importances_)
plt.title("Feature Importance")
plt.show()

# -----------------------------
# 7️⃣ Save Model
# -----------------------------
MODEL_DIR = SCRIPT_DIR
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "debt_strategy_model_realistic_v5.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "debt_scaler_realistic_v5.pkl"))
print("💾 Model saved successfully.")

