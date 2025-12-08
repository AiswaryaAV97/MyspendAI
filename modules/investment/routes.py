import datetime as dt
import os

import numpy as np
import pandas as pd
import yfinance as yf
from flask import request, render_template, redirect, url_for, flash

import plotly.express as px
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from pymongo import MongoClient

# ----------------------------------------
# MongoDB Setup
# ----------------------------------------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["SmartSpendAI"]
investment_collection = db["investment_history"]

print("✓ MongoDB connected for Investment module")


from . import bp


# Config
TICKERS = ["VEQT.TO", "XEQT.TO", "XIC.TO", "XBB.TO", "VAB.TO", "ZAG.TO", "XSB.TO"]

ASSET_CLASS = {
    "VEQT.TO": "Equity",
    "XEQT.TO": "Equity",
    "XIC.TO": "Equity",
    "XBB.TO": "Bond",
    "VAB.TO": "Bond",
    "ZAG.TO": "Bond",
    "XSB.TO": "Bond",
}


# XGBoost model (inline training)

XGB_MODEL = None
XGB_SCALER = None

def train_xgb_inline():
    """
    Train a small XGBoost model on the fly using the same tickers (TICKERS)
    and save it into global XGB_MODEL + XGB_SCALER.
    This runs automatically if no model is loaded.
    """
    global XGB_MODEL, XGB_SCALER

    try:
        print("🔧 Training XGBoost model inline for investment module...")
        # use 5 years to have enough history
        prices = fetch_price_data(TICKERS, years=5)
    except Exception as e:
        print("⚠ Could not fetch data to train XGBoost:", e)
        return

    # build supervised dataset similar to the train_xgb.py logic
    # daily log returns
    log_ret = np.log(prices / prices.shift(1))
    r5 = prices.pct_change(5)
    r20 = prices.pct_change(20)
    r60 = prices.pct_change(60)
    vol20 = log_ret.rolling(20).std()
    roll_max60 = prices.rolling(60).max()
    dd60 = (prices - roll_max60) / roll_max60

    # target: 60-day forward return
    future_price = prices.shift(-60)
    fwd_ret = (future_price - prices) / prices

    rows = []
    for t in prices.columns:
        df = pd.DataFrame({
            "ticker": t,
            "r5": r5[t],
            "r20": r20[t],
            "r60": r60[t],
            "vol20": vol20[t],
            "dd60": dd60[t],
            "target": fwd_ret[t],
        })
        rows.append(df)

    all_df = pd.concat(rows).dropna()

    if all_df.empty:
        print("⚠ Not enough data to train XGBoost.")
        return

    X = all_df[["r5", "r20", "r60", "vol20", "dd60"]].values
    y = all_df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_scaled, y)

    XGB_MODEL = model
    XGB_SCALER = scaler
    print("✅ Inline XGBoost model trained successfully.")


# -----------------------------
# ML model paths (XGBoost)
# -----------------------------


XGB_MODEL = None
XGB_SCALER = None

# -----------------------------
# Data + Feature Engineering
# -----------------------------
def fetch_price_data(tickers, years=3):
    """
    Fetch latest price data for the past N years.
    Works with different yfinance DataFrame formats.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)

    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
    )

    if data is None or len(data) == 0:
        raise ValueError("No price data returned from yfinance.")

    # If columns are MultiIndex (field, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        if "Adj Close" in level0:
            prices = data.xs("Adj Close", axis=1, level=0)
        elif "Close" in level0:
            prices = data.xs("Close", axis=1, level=0)
        else:
            prices = data.select_dtypes(include=[np.number])
    else:
        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            prices = data.select_dtypes(include=[np.number])

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("Price data is empty after cleaning.")
    return prices


def compute_returns(price_df):
    rets = price_df.pct_change().dropna()
    return rets


def build_features(returns_df):
    """
    Build features per ticker:
      - annualized return
      - annualized volatility
      - Sharpe-like ratio
      - 6-month momentum
      - max drawdown
    (Used for clustering, not XGBoost.)
    """
    if isinstance(returns_df, pd.Series):
        returns_df = returns_df.to_frame()

    trading_days = 252
    prices = (1 + returns_df).cumprod()

    features = []
    for col in returns_df.columns:
        r = returns_df[col].dropna()
        if r.empty:
            continue

        mean_daily = r.mean()
        vol_daily = r.std()

        ann_return = (1 + mean_daily) ** trading_days - 1
        ann_vol = vol_daily * np.sqrt(trading_days)
        sharpe_like = ann_return / ann_vol if ann_vol > 0 else 0.0

        # 6-month momentum
        last_126 = prices[col].iloc[-126:] if len(prices[col]) >= 126 else prices[col]
        if len(last_126) > 1:
            momentum_6m = last_126.iloc[-1] / last_126.iloc[0] - 1
        else:
            momentum_6m = 0.0

        cum = prices[col]
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_dd = drawdown.min()

        features.append(
            {
                "Ticker": col,
                "ann_return": float(ann_return),
                "ann_vol": float(ann_vol),
                "sharpe": float(sharpe_like),
                "mom_6m": float(momentum_6m),
                "max_drawdown": float(max_dd),
            }
        )

    feat_df = pd.DataFrame(features).set_index("Ticker")
    return feat_df



# Extra feature builder for XGBoost (per ticker latest snapshot)
def latest_features_for_ticker(price_series):
    """
    Build the single feature row [r5, r20, r60, vol20, dd60]
    matching the XGBoost training script.
    """
    p = price_series.dropna()
    if len(p) < 90:
        return None

    r5 = p.pct_change(5).iloc[-1]
    r20 = p.pct_change(20).iloc[-1]
    r60 = p.pct_change(60).iloc[-1]

    log_ret = np.log(p / p.shift(1))
    vol20 = log_ret.rolling(20).std().iloc[-1]

    roll_max60 = p.rolling(60).max()
    dd60_series = (p - roll_max60) / roll_max60
    dd60 = dd60_series.iloc[-1]

    return np.array([r5, r20, r60, vol20, dd60], dtype=float)


def predict_ml_expected_returns(prices_df):
    """
    Used XGBoost model to predict forward returns for each ticker.
    If model/scaler are missing, train them inline once.
    """
    global XGB_MODEL, XGB_SCALER

    
    if XGB_MODEL is None or XGB_SCALER is None:
        train_xgb_inline()
        if XGB_MODEL is None or XGB_SCALER is None:
            # still failed -> give up and fallback
            return None

    preds = {}
    for t in prices_df.columns:
        series = prices_df[t].dropna()
        feats = latest_features_for_ticker(series)
        if feats is None:
            continue
        try:
            x_scaled = XGB_SCALER.transform(feats.reshape(1, -1))
            fwd_60d = XGB_MODEL.predict(x_scaled)[0]
            # approx annualize
            annual = (1 + fwd_60d) ** (252 / 60) - 1
            preds[t] = float(annual)
        except Exception:
            continue

    if not preds:
        return None
    return pd.Series(preds)



# ML: Clustering into risk buckets
def cluster_assets(feature_df, n_clusters=3):
    """
    Use KMeans to cluster assets into risk buckets.
    Then label clusters as Conservative / Balanced / Aggressive
    based on centroid volatility + return.
    """
    if feature_df.empty:
        raise ValueError("No features to cluster.")

    X = feature_df[["ann_return", "ann_vol", "sharpe", "mom_6m", "max_drawdown"]].values
    k = min(n_clusters, len(feature_df))
    if k <= 1:
        return pd.Series(["Balanced"] * len(feature_df), index=feature_df.index)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    feature_df = feature_df.copy()
    feature_df["cluster"] = labels

    cluster_stats = (
        feature_df.groupby("cluster")[["ann_vol", "ann_return"]].mean().reset_index()
    )

    cluster_stats = cluster_stats.sort_values("ann_vol")
    mapping = {}
    if len(cluster_stats) == 1:
        mapping[cluster_stats.iloc[0]["cluster"]] = "Balanced"
    elif len(cluster_stats) == 2:
        mapping[cluster_stats.iloc[0]["cluster"]] = "Conservative"
        mapping[cluster_stats.iloc[1]["cluster"]] = "Aggressive"
    else:
        mapping[cluster_stats.iloc[0]["cluster"]] = "Conservative"
        mapping[cluster_stats.iloc[1]["cluster"]] = "Balanced"
        for _, row in cluster_stats.iloc[2:].iterrows():
            mapping[row["cluster"]] = "Aggressive"

    bucket_labels = feature_df["cluster"].map(mapping)
    return bucket_labels  # index = ticker, value = bucket


def build_portfolio_weights(bucket_labels, risk_pref):
    """
    Original rule-based weights using risk buckets only.
    Used as fallback when ML model is not available.
    """
    risk_pref = (risk_pref or "medium").lower()

    if risk_pref == "low":
        bucket_target = {"Conservative": 0.6, "Balanced": 0.3, "Aggressive": 0.1}
    elif risk_pref == "high":
        bucket_target = {"Conservative": 0.1, "Balanced": 0.3, "Aggressive": 0.6}
    else:
        bucket_target = {"Conservative": 0.3, "Balanced": 0.4, "Aggressive": 0.3}

    w = pd.Series(0.0, index=bucket_labels.index)
    for bucket, target_weight in bucket_target.items():
        members = bucket_labels[bucket_labels == bucket].index
        if len(members) == 0 or target_weight <= 0:
            continue
        per_asset = target_weight / len(members)
        w.loc[members] = per_asset

    total = w.sum()
    if total > 0:
        w = w / total
    else:
        w[:] = 1 / len(w)
    return w


def build_portfolio_weights_ml(bucket_labels, predicted_returns, risk_pref):
    """
    New ML-based weights:
    - Use ML-predicted annualized returns per ticker as base score
    - Modulate by risk bucket & user risk preference
    """
    risk_pref = (risk_pref or "medium").lower()
    tickers = bucket_labels.index

    scores = []
    for t in tickers:
        base_r = float(predicted_returns.get(t, 0.0))
        bucket = bucket_labels[t]  # Conservative / Balanced / Aggressive
        score = max(base_r, 0.0)

        if risk_pref == "low":
            if bucket == "Aggressive":
                score *= 0.25
            elif bucket == "Balanced":
                score *= 0.6
        elif risk_pref == "high":
            if bucket == "Aggressive":
                score *= 1.3

        scores.append(score)

    arr = np.array(scores, dtype=float)
    if arr.sum() <= 0:
        # fallback to original bucket-only weights
        return build_portfolio_weights(bucket_labels, risk_pref)

    arr = arr / arr.sum()
    return pd.Series(arr, index=tickers)


# -----------------------------
# Portfolio simulation / metrics
# -----------------------------
def simulate_portfolio_growth(returns_df, weights, initial_amount=1000.0):
    returns_df = returns_df[weights.index]
    port_daily = (returns_df * weights.values).sum(axis=1)
    cumulative = (1 + port_daily).cumprod()
    series = initial_amount * cumulative
    return series


def compute_portfolio_metrics(returns_df, weights):
    returns_df = returns_df[weights.index]
    port_daily = (returns_df * weights.values).sum(axis=1)

    trading_days = 252
    mean_daily = port_daily.mean()
    vol_daily = port_daily.std()

    exp_return = (1 + mean_daily) ** trading_days - 1
    ann_vol = vol_daily * np.sqrt(trading_days)

    cumulative = (1 + port_daily).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    return {
        "expected_return": float(exp_return),
        "volatility": float(ann_vol),
        "max_drawdown": float(max_dd),
    }


# -----------------------------
# Wrapper allocation (RRSP / TFSA / FHSA)
# -----------------------------
def wrapper_allocation(amount, fhsa_room, home_plan, risk_level, portfolio_type):
    """
    Wrapper logic:

    - If home_plan = "no":
        * Ignore FHSA completely (FHSA = 0).
        * Put 100% into the chosen account (RRSP or TFSA).
        * If FHSA somehow selected, fall back to TFSA.

    - If home_plan = "yes":
        * If FHSA chosen:
            - Allocate up to fhsa_room into FHSA.
            - Any leftover goes to TFSA by default.
        * If RRSP chosen: 100% to RRSP.
        * If TFSA chosen: 100% to TFSA.
    """
    amount = float(amount)
    fhsa_room = max(float(fhsa_room or 0.0), 0.0)
    home_plan = (home_plan or "").lower() == "yes"
    portfolio_type = (portfolio_type or "").upper()

    rrsp_alloc = 0.0
    tfsa_alloc = 0.0
    fhsa_alloc = 0.0

    if not home_plan:
        if portfolio_type == "RRSP":
            rrsp_alloc = amount
        else:
            tfsa_alloc = amount
    else:
        if portfolio_type == "FHSA":
            if fhsa_room > 0:
                fhsa_alloc = min(amount, fhsa_room)
                tfsa_alloc = amount - fhsa_alloc
            else:
                tfsa_alloc = amount
        elif portfolio_type == "RRSP":
            rrsp_alloc = amount
        else:
            tfsa_alloc = amount

    total = rrsp_alloc + tfsa_alloc + fhsa_alloc
    return {
        "RRSP": rrsp_alloc,
        "TFSA": tfsa_alloc,
        "FHSA": fhsa_alloc,
        "total": total,
    }


# -----------------------------
# Routes
# -----------------------------
@bp.route("/", methods=["GET"])
def investment_home():
    return redirect(url_for("investment.investment_planner"))


@bp.route("/planner", methods=["GET", "POST"], endpoint="investment_planner")
def investment_planner():
    if request.method == "GET":
        # use your existing template path
        return render_template("investment.html")

    # ---------- Parse inputs ----------
    try:
        investment_amount = float(request.form.get("investment_amount", 0))
    except ValueError:
        investment_amount = 0.0

    try:
        fhsa_contrib = float(request.form.get("fhsa_contrib", 0))
    except ValueError:
        fhsa_contrib = 0.0

    home_plan = request.form.get("home_plan", "no")  # yes/no
    risk_pref = request.form.get("risk_pref", "medium")
    horizon = request.form.get("horizon", "5")
    horizon_unit = request.form.get("horizon_unit", "years")
    portfolio_type = request.form.get("portfolio_type", "TFSA")

    if investment_amount <= 0:
        flash("Please enter a valid investment amount.", "error")
        return redirect(url_for("investment.investment_planner"))

    # convert horizon to years (for future use if needed)
    try:
        horizon_val = float(horizon)
    except ValueError:
        horizon_val = 5.0
    if horizon_unit == "months":
        horizon_years = horizon_val / 12.0
    else:
        horizon_years = horizon_val

    # ---------- Fetch latest market data + build features ----------
    try:
        prices = fetch_price_data(TICKERS, years=3)
        returns = compute_returns(prices)
        feat_df = build_features(returns)
        feat_df = feat_df.loc[feat_df.index.intersection(returns.columns)]
        returns = returns[feat_df.index]
        prices = prices[feat_df.index]  # keep same ticker set for ML
    except Exception as e:
        flash(f"Error fetching or processing market data: {e}", "error")
        return redirect(url_for("investment.investment_planner"))

    # ---------- ML: clustering into risk buckets ----------
    bucket_labels = cluster_assets(feat_df, n_clusters=3)

    # ---------- ML: expected returns with XGBoost (if available) ----------
    predicted_ret = predict_ml_expected_returns(prices)

    # ---------- Build portfolio weights ----------
    if predicted_ret is not None:
        # use ML-based weights
        w_adj = build_portfolio_weights_ml(bucket_labels, predicted_ret, risk_pref)
    else:
        # fallback: original bucket-only weights
        w_adj = build_portfolio_weights(bucket_labels, risk_pref)

    # align returns/prices with final weights index
    returns = returns[w_adj.index]

    # ---------- Portfolio metrics ----------
    metrics = compute_portfolio_metrics(returns, w_adj)
    if predicted_ret is not None:
        # overwrite expected_return with ML-based portfolio expected return
        aligned_pred = predicted_ret.reindex(w_adj.index).fillna(0.0)
        metrics["expected_return"] = float((aligned_pred * w_adj).sum())

    # growth_series is kept for internal logic if needed; not used on page now
    growth_series = simulate_portfolio_growth(
        returns, w_adj, initial_amount=investment_amount
    )

    # ---------- Wrapper-level allocation (RRSP / TFSA / FHSA) ----------
    wrappers = wrapper_allocation(
        amount=investment_amount,
        fhsa_room=fhsa_contrib,
        home_plan=home_plan,
        risk_level=risk_pref,
        portfolio_type=portfolio_type,
    )

    wrapper_perc = {
        k: (v / wrappers["total"] * 100 if wrappers["total"] > 0 else 0)
        for k, v in wrappers.items()
        if k != "total"
    }
    best_wrapper = max(wrapper_perc, key=wrapper_perc.get) if wrapper_perc else None

    # ---------- Charts ----------
    # Allocation bar chart
    alloc_df = pd.DataFrame(
        {
            "Ticker": w_adj.index,
            "Weight": (w_adj.values * 100.0),
            "AssetClass": [ASSET_CLASS[t] for t in w_adj.index],
            "RiskBucket": [bucket_labels[t] for t in w_adj.index],
        }
    )
    alloc_fig = px.bar(
        alloc_df,
        x="Ticker",
        y="Weight",
        color="RiskBucket",
        title="ML-Based Portfolio Allocation by Asset",
        labels={"Weight": "Weight (%)"},
        text=alloc_df["Weight"].map(lambda x: f"{x:.1f}%"),
    )
    alloc_chart_json = pio.to_json(alloc_fig)

    # Pie chart: split of chosen account (RRSP / TFSA / FHSA) across ETFs/bonds
    selected_account = portfolio_type.upper()
    account_amount = wrappers.get(selected_account, 0.0)

    pie_df = pd.DataFrame(
        {
            "Ticker": w_adj.index,
            "Amount": w_adj.values * account_amount,
            "AssetClass": [ASSET_CLASS[t] for t in w_adj.index],
        }
    )
    pie_df = pie_df[pie_df["Amount"] > 0]

    pie_title = f"{selected_account} allocation across ETFs/Bonds"
    pie_fig = px.pie(
        pie_df,
        names="Ticker",
        values="Amount",
        title=pie_title,
        hover_data=["AssetClass"],
    )
    pie_chart_json = pio.to_json(pie_fig)

        # ---------- Prepare data for template ----------
    def fmt_pct(x):
        return round(x * 100, 2)

    metrics_display = {
        "expected_return_pct": fmt_pct(metrics["expected_return"]),
        "volatility_pct": fmt_pct(metrics["volatility"]),
        "max_drawdown_pct": fmt_pct(metrics["max_drawdown"]),
    }

    wrapper_display = {k: round(v, 2) for k, v in wrappers.items() if k != "total"}
    wrapper_display_pct = {
        k: round((v / wrappers["total"] * 100), 2) if wrappers["total"] > 0 else 0
        for k, v in wrappers.items()
        if k != "total"
    }

    # New calculations: per-asset cost + expected future value
    annual_return = metrics["expected_return"]  # already computed
    horizon_years = float(horizon) if horizon_unit == "years" else float(horizon) / 12

    asset_analysis = []
    for ticker in w_adj.index:
        weight = float(w_adj[ticker])
        invest_amount = investment_amount * weight

        # expected value after horizon
        fv = invest_amount * ((1 + annual_return) ** horizon_years)

        asset_analysis.append({
            "Ticker": ticker,
            "AssetClass": ASSET_CLASS[ticker],
            "WeightPct": round(weight * 100, 2),
            "InvestAmount": round(invest_amount, 2),
            "FutureValue": round(fv, 2),
            "ExpectedGain": round(fv - invest_amount, 2)
        })

    # Determine best ETF/Bond
    best_asset = max(asset_analysis, key=lambda x: x["FutureValue"])

        # ----------------------------------------
    # Save results to MongoDB
    # ----------------------------------------
    try:
        from flask import session  # ensure session is imported

        username = session.get("user", "guest")

        investment_collection.insert_one({
            "username": username,
            "investment_amount": investment_amount,
            "fhsa_contrib": fhsa_contrib,
            "home_plan": home_plan,
            "risk_pref": risk_pref,
            "horizon": horizon,
            "horizon_unit": horizon_unit,
            "portfolio_type": portfolio_type,
            "weights": w_adj.to_dict(),
            "metrics": metrics_display,
            "wrapper_allocation": wrappers,
            "wrapper_percentages": wrapper_display_pct,
            "asset_analysis": asset_analysis,
            "best_asset": best_asset,
            "created_at": dt.datetime.utcnow()
        })

        print("✓ Investment record saved to MongoDB")

    except Exception as e:
        print("⚠ Error saving investment data:", e)
        flash("Unable to save your investment results.", "error")

    # ----------------------------------------
    # Return final results to template
    # ----------------------------------------
    return render_template(
        "result.html",
        investment_amount=investment_amount,
        fhsa_contrib=fhsa_contrib,
        home_plan=home_plan,
        risk_pref=risk_pref,
        horizon=horizon,
        horizon_unit=horizon_unit,
        portfolio_type=portfolio_type,
        weights=w_adj.to_dict(),
        asset_classes=ASSET_CLASS,
        metrics=metrics_display,
        wrapper_amounts=wrapper_display,
        wrapper_perc=wrapper_display_pct,
        best_wrapper=best_wrapper,
        alloc_chart_json=alloc_chart_json,
        pie_chart_json=pie_chart_json,
        asset_analysis=asset_analysis,
        best_asset=best_asset,
    )