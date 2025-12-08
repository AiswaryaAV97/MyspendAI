# ...existing code...
from pymongo import MongoClient
from .mortgage_calculator import (
    calculate_mortgage_with_cmhc,
    calculate_min_down_payment,
)
import datetime
from dotenv import load_dotenv
from flask import current_app
import os
import joblib
import pandas as pd
import logging

# Load environment variables from .env
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

# Connect to MongoDB (lazy safe connect)
client = MongoClient(MONGODB_URI) if MONGODB_URI else None
db = client[DB_NAME] if (client is not None and DB_NAME) else None
collection = db.mortgage_calculations if db is not None else None



def _to_bool(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    sval = str(val).strip().lower()
    return sval in ("1", "true", "yes", "y", "on")


MODEL_PATH = os.path.join(os.path.dirname(__file__), "mortgage_chain.pkl")
_mortgage_model = None

def load_mortgage_model():
    """
    Lazy-load the mortgage_chain.pkl model. Returns model or None.
    """
    global _mortgage_model
    if _mortgage_model is not None:
        return _mortgage_model

    if not os.path.exists(MODEL_PATH):
        logging.warning("Mortgage model not found at %s", MODEL_PATH)
        return None

    try:
        _mortgage_model = joblib.load(MODEL_PATH)
        logging.info("Loaded mortgage model from %s", MODEL_PATH)
    except Exception:
        logging.exception("Failed to load mortgage model from %s", MODEL_PATH)
        _mortgage_model = None
    return _mortgage_model

def process_affordability(payload):
    """
    Run the trained pipeline (mortgage_chain.pkl) and return a JSON-friendly dict:
      { predicted_max_home_price, predicted_monthly_payment, debt_to_income_ratio, affordable }
    """
    mdl = load_mortgage_model()
    if mdl is None:
        return {"error": "Affordability model not available"}

    try:
        # Build input row with the same columns used in training
        row = {
            "Province": payload.get("province", "ON"),
            "Household_Income": float(payload.get("household_income", 0)),
            "Downpayment": float(payload.get("down_payment", 0)),
            "Debts_per_month": float(payload.get("debts_per_month", 0)),
            "Credit_total": float(payload.get("credit_total", 0)),
            "Monthly_condo_fees": float(payload.get("monthly_condo_fees", 0)),
            "Amortization_period": float(payload.get("amortization_period", payload.get("years", 0))),
            "Interest_rate": float(payload.get("interest_rate", payload.get("rate", 0))),
        }
        X = pd.DataFrame([row])
        pred = mdl.predict(X)  # RegressorChain -> array shape (n_samples, n_targets) or (n_samples,)

        if hasattr(pred, "shape") and getattr(pred, "ndim", 1) == 2 and pred.shape[1] >= 2:
            max_price = float(pred[0, 0])
            monthly_payment = float(pred[0, 1])
        else:
            # fallback if single-target output
            max_price = float(pred[0])
            monthly_payment = round(max_price * 0.005, 2)

        monthly_income = row["Household_Income"] / 12.0 if row["Household_Income"] else None
        debt_ratio = None
        if monthly_income and monthly_income > 0:
            debt_ratio = (row["Debts_per_month"] + monthly_payment) / monthly_income

        affordable = (debt_ratio is not None and debt_ratio < 0.45)

        return {
            "predicted_max_home_price": max_price,
            "predicted_monthly_payment": monthly_payment,
            "debt_to_income_ratio": debt_ratio,
            "affordable": affordable
        }
    except Exception:
        logging.exception("Affordability prediction failed")
        return {"error": "Affordability prediction failed"}


def process_mortgage(data):
    """
    Validate input, compute mortgage values, store record in MongoDB, and return result.
    Accepts fields: price, down_payment, rate, years, province (optional), city (optional),
    first_time or first_time_buyer (optional boolean).
    Returns a dict containing monthly_payment, cmhc_fee, ltt breakdown, min_down_payment, etc.
    If down payment is below minimum, returns {"error": "..."} (caller should handle).
    """
    if not isinstance(data, dict):
        raise ValueError("Input data must be a JSON object")

    # Coerce and validate numeric inputs
    try:
        price = float(data.get("price"))
        down_payment = float(data.get("down_payment"))
        rate = float(data.get("rate"))
        years = int(data.get("years"))
    except Exception:
        raise ValueError("Invalid numeric inputs: price, down_payment, rate, years are required")

    province = (data.get("province") or "ON").strip()
    city = (data.get("city") or "").strip() or None

    # accept several possible field names / values for first-time flag
    first_time_raw = data.get("first_time")
    if first_time_raw is None:
        first_time_raw = data.get("first_time_buyer")
    if first_time_raw is None:
        first_time_raw = data.get("first_time_yes")

    first_time = _to_bool(first_time_raw)

    # Enforce minimum down payment
    min_dp = calculate_min_down_payment(price)
    if down_payment < min_dp:
        return {"error": f"Minimum down payment for this price is ${min_dp:,.2f}"}

    # Use high-level calculator to compute CMHC, monthly & biweekly payments, and LTT breakdown
    try:
        result = calculate_mortgage_with_cmhc(
            price=price,
            down_payment=down_payment,
            rate=rate,
            years=years,
            province=province,
            city=city,
            first_time=first_time,
        )
    except Exception as e:
        return {"error": f"Calculation failed: {e}"}
    
    # Try to attach affordability prediction information if model present.
    mdl = load_mortgage_model()
    if mdl is not None:
        try:
            inp = pd.DataFrame([{
                "Province": province,
                "Household_Income": float(data.get("household_income", 0)),
                "Downpayment": float(down_payment),
                "Debts_per_month": float(data.get("debts_per_month", 0)),
                "Credit_total": float(data.get("credit_total", 0)),
                "Monthly_condo_fees": float(data.get("monthly_condo_fees", 0)),
                "Amortization_period": int(years),
                "Interest_rate": float(rate),
            }])
            pred = mdl.predict(inp)
            # handle different output shapes:
            if hasattr(pred, "shape") and getattr(pred, "ndim", 1) == 2 and pred.shape[1] >= 2:
                row_pred = pred[0]
                result["predicted_max_home_price"] = float(row_pred[0])
                result["predicted_monthly_payment"] = float(row_pred[1])
            else:
                first_val = pred[0]
                result["predicted_max_home_price"] = float(first_val)
                result["predicted_monthly_payment"] = round(float(first_val) * 0.005, 2)
        except Exception:
            logging.exception("Failed to run affordability model during process_mortgage")
            # do not fail the main mortgage calculation; model info is optional

    # attach metadata and min down
    result.setdefault("meta", {})
    result["meta"].update({"province": province, "city": city, "first_time_buyer": first_time})
    result["min_down_payment"] = round(min_dp, 2)

    # Store input + output in MongoDB if available
    try:
        if collection:
            record = {
                "timestamp": datetime.datetime.utcnow(),
                "input": data,
                "output": result
            }
            collection.insert_one(record)
    except Exception:
        # Do not fail the calculation if DB write fails; caller can inspect logs
        logging.exception("Failed to write calculation record to MongoDB")
        pass

    return result