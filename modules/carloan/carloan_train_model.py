# modules/carloan/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# -----------------------------
# 1️⃣ Generate Synthetic Dataset
# -----------------------------
def generate_carloan_dataset(n_samples=5000):
    """
    Generate realistic car loan dataset with multiple features.
    """
    np.random.seed(42)
    
    # Price range: $15k - $120k
    car_prices = np.random.uniform(15000, 120000, n_samples)
    
    # Down payment: 5% - 30% of price
    down_payment_pcts = np.random.uniform(0.05, 0.30, n_samples)
    down_payments = car_prices * down_payment_pcts
    
    # Interest rates: 3% - 18% based on credit
    credit_scores = np.random.randint(550, 850, n_samples)
    
    # Calculate interest rate based on credit score
    interest_rates = []
    for score in credit_scores:
        if score >= 750:
            rate = np.random.uniform(3.5, 5.5)
        elif score >= 700:
            rate = np.random.uniform(5.5, 7.5)
        elif score >= 650:
            rate = np.random.uniform(7.5, 10.5)
        else:
            rate = np.random.uniform(10.5, 16.0)
        interest_rates.append(rate)
    interest_rates = np.array(interest_rates)
    
    # Loan terms: 36, 48, 60, 72 months
    loan_terms = np.random.choice([36, 48, 60, 72], n_samples)
    
    # Extra payments: 0 - $500
    extra_payments = np.random.choice([0, 50, 100, 150, 200, 300, 500], n_samples, 
                                     p=[0.5, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02])
    
    # Calculate total interest (target variable)
    total_interests = []
    for i in range(n_samples):
        principal = car_prices[i] - down_payments[i]
        rate = interest_rates[i]
        term = loan_terms[i]
        extra = extra_payments[i]
        
        # Calculate monthly payment
        r = rate / 12 / 100
        if r == 0:
            monthly = principal / term
        else:
            monthly = principal * (r * (1 + r)**term) / ((1 + r)**term - 1)
        
        # Simulate payoff with extra payment
        balance = principal
        total_int = 0
        months = 0
        while balance > 0 and months < 600:
            months += 1
            interest = balance * r
            payment = monthly + extra
            principal_paid = payment - interest
            if principal_paid > balance:
                principal_paid = balance
                interest = balance * r
            balance -= principal_paid
            total_int += interest
        
        total_interests.append(total_int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Car_Price': car_prices,
        'Down_Payment': down_payments,
        'Interest_Rate': interest_rates,
        'Loan_Term_Months': loan_terms,
        'Extra_Payment': extra_payments,
        'Credit_Score': credit_scores,
        'Total_Interest': total_interests
    })
    
    return df

# -----------------------------
# 2️⃣ Load or Generate Dataset
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "carloan_data.csv")

# Check if dataset exists and has correct columns
regenerate = False
if os.path.exists(DATA_PATH):
    print("📂 Checking existing dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
        required_cols = ['Car_Price', 'Down_Payment', 'Interest_Rate', 'Loan_Term_Months', 
                        'Extra_Payment', 'Credit_Score', 'Total_Interest']
        
        if not all(col in df.columns for col in required_cols):
            print("⚠️  Dataset missing required columns. Regenerating...")
            regenerate = True
        else:
            print("✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"⚠️  Error loading dataset: {e}. Regenerating...")
        regenerate = True
else:
    regenerate = True

if regenerate:
    print("🔨 Generating new synthetic dataset...")
    df = generate_carloan_dataset(5000)
    df.to_csv(DATA_PATH, index=False)
    print(f"💾 Dataset saved to {DATA_PATH}")

print("\n📊 Dataset Info:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nStatistics:\n{df.describe()}")

# -----------------------------
# 3️⃣ Feature Engineering
# -----------------------------
# Add derived features
df['Principal'] = df['Car_Price'] - df['Down_Payment']
df['Down_Payment_Pct'] = (df['Down_Payment'] / df['Car_Price']) * 100
df['Debt_To_Income_Proxy'] = df['Principal'] / 60000  # Assume avg income 60k

# Select features for training
feature_cols = ['Car_Price', 'Down_Payment', 'Interest_Rate', 'Loan_Term_Months', 
                'Extra_Payment', 'Credit_Score']

print(f"\n🎯 Training features: {feature_cols}")

X = df[feature_cols]
y = df['Total_Interest']

# -----------------------------
# 4️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✂️ Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -----------------------------
# 5️⃣ Build Enhanced Pipeline
# -----------------------------
print("\n🤖 Training Random Forest model...")

pipeline_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ))
])

print("⏳ Training in progress...")
pipeline_rf.fit(X_train, y_train)
print("✅ Training complete!")

# -----------------------------
# 6️⃣ Evaluate Model
# -----------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Comprehensive model evaluation."""
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Performance")
    print(f"{'='*50}")
    print(f"\n🎯 Training Metrics:")
    print(f"   MAE:  ${train_mae:,.2f}")
    print(f"   RMSE: ${np.sqrt(train_mse):,.2f}")
    print(f"   R²:   {train_r2:.4f}")
    
    print(f"\n🎯 Test Metrics:")
    print(f"   MAE:  ${test_mae:,.2f}")
    print(f"   RMSE: ${np.sqrt(test_mse):,.2f}")
    print(f"   R²:   {test_r2:.4f}")
    
    # Check for overfitting
    overfit_diff = train_r2 - test_r2
    if overfit_diff > 0.1:
        print(f"\n⚠️  Warning: Potential overfitting detected (diff: {overfit_diff:.4f})")
    else:
        print(f"\n✅ Model generalizes well (diff: {overfit_diff:.4f})")
    
    return {
        'test_mae': test_mae,
        'test_r2': test_r2,
        'predictions': y_test_pred
    }

results_rf = evaluate_model(pipeline_rf, X_train, X_test, y_train, y_test, "Random Forest")

# -----------------------------
# 7️⃣ Cross-Validation
# -----------------------------
print("\n🔄 Running 5-Fold Cross-Validation...")
cv_scores = cross_val_score(
    pipeline_rf, X_train, y_train, 
    cv=5, scoring='r2', n_jobs=-1
)
print(f"CV R² Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# -----------------------------
# 8️⃣ Feature Importance
# -----------------------------
feature_importance = pipeline_rf.named_steps['model'].feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance:")
for idx, row in feature_importance_df.iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"   {row['Feature']:20s} {bar} {row['Importance']:.4f}")

# -----------------------------
# 9️⃣ Save Model
# -----------------------------
MODEL_PATH = os.path.join(SCRIPT_DIR, "carloan_model.pkl")
joblib.dump(pipeline_rf, MODEL_PATH)
print(f"\n💾 Model saved successfully to: {MODEL_PATH}")

# -----------------------------
# 🔟 Test Predictions
# -----------------------------
def calculate_monthly_payment(principal, rate, months):
    """Calculate monthly payment."""
    r = rate / 12 / 100
    if r == 0:
        return principal / months
    return principal * (r * (1 + r)**months) / ((1 + r)**months - 1)

print("\n🧪 Sample Predictions:")
sample_features = [
    [35000, 7000, 6.5, 60, 0, 720],      # Mid-range car
    [80000, 16000, 4.5, 48, 200, 780],   # Luxury car with extra payment
    [25000, 2500, 12.0, 72, 0, 620],     # Budget car, poor credit
    [50000, 10000, 7.0, 60, 100, 700],   # Average car with extra payment
]

sample_labels = [
    "Mid-range car, good credit",
    "Luxury car, excellent credit + extra $200/mo",
    "Budget car, fair credit",
    "Average car, good credit + extra $100/mo"
]

for i, (features, label) in enumerate(zip(sample_features, sample_labels), 1):
    prediction = pipeline_rf.predict([features])[0]
    monthly_payment = calculate_monthly_payment(features[0] - features[1], features[2], features[3])
    
    print(f"\n📝 Sample {i}: {label}")
    print(f"   Price: ${features[0]:,} | Down: ${features[1]:,} ({features[1]/features[0]*100:.0f}%)")
    print(f"   Rate: {features[2]}% | Term: {features[3]} months | Extra: ${features[4]}/mo | Credit: {features[5]}")
    print(f"   💰 Predicted Total Interest: ${prediction:,.2f}")
    print(f"   📅 Estimated Monthly Payment: ${monthly_payment + features[4]:,.2f}")

print("\n" + "="*50)
print("✅ Training Complete! Model ready for deployment.")
print("="*50)
print("\n💡 Next Steps:")
print("   1. The model file 'carloan_model.pkl' is ready")
print("   2. Start your Flask app to use the AI predictions")
print("   3. Model will automatically load when accessing /carLoanCalc")