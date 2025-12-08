# -*- coding: utf-8 -*-
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

#   CONFIG 

ASSET_UNIVERSE = [
    "ZAG.TO",   # CAD Bonds
    "ZFL.TO",   # Long Bonds
    "XIC.TO",   # Canadian Equity
    "XAW.TO",   # Global Equity
    "XEQT.TO",  # All-Equity
    "QQQ",      # US Tech
]

YEARS_BACK = 5           # training history
PRED_HORIZON_DAYS = 60   # predict 60d forward return

MODEL_PATH = "modules/investment/ml_xgb_model.pkl"
SCALER_PATH = "modules/investment/ml_xgb_scaler.pkl"

#  2. DATA DOWNLOAD 

end = dt.date.today()
start = end - dt.timedelta(days=365 * YEARS_BACK)

print(f"Downloading price history from {start} to {end}...")
data = yf.download(ASSET_UNIVERSE, start=start, end=end, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    if "Adj Close" in data.columns.get_level_values(0):
        prices = data.xs("Adj Close", axis=1, level=0)
    else:
        prices = data.xs("Close", axis=1, level=0)
else:
    prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

prices = prices.dropna(how="all")

# FEATURE ENGINEERING 

def build_features(prices, horizon=PRED_HORIZON_DAYS):
    """
    Builds a supervised dataset:
      X: features from window [t-90..t]
      y: forward return over next 'horizon' days
    Features per date per ticker:
      - 5d, 20d, 60d returns
      - 20d volatility
      - 60d drawdown
    """
    log_ret = np.log(prices / prices.shift(1))
    r5 = prices.pct_change(5)
    r20 = prices.pct_change(20)
    r60 = prices.pct_change(60)
    vol20 = log_ret.rolling(20).std()
    roll_max60 = prices.rolling(60).max()
    dd60 = (prices - roll_max60) / roll_max60

    # target: future horizon return
    future_price = prices.shift(-horizon)
    fwd_ret = (future_price - prices) / prices

    rows = []
    for ticker in prices.columns:
        df = pd.DataFrame({
            "ticker": ticker,
            "r5": r5[ticker],
            "r20": r20[ticker],
            "r60": r60[ticker],
            "vol20": vol20[ticker],
            "dd60": dd60[ticker],
            "target": fwd_ret[ticker],
        })
        rows.append(df)

    all_df = pd.concat(rows)
    all_df = all_df.dropna()

    X = all_df[["r5", "r20", "r60", "vol20", "dd60"]].values
    y = all_df["target"].values
    return X, y

X, y = build_features(prices)
print("Dataset shape:", X.shape, y.shape)

#  TRAIN / SAVE 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
)
print("Training XGBoost model...")
model.fit(X_train_scaled, y_train)

# simple evaluation
from sklearn.metrics import r2_score, mean_squared_error
y_pred = model.predict(X_test_scaled)
print("R2 Score:", r2_score(y_test, y_pred))

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("RMSE:", rmse)

# save model + scaler
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print("Saved model to", MODEL_PATH)
print("Saved scaler to", SCALER_PATH)

