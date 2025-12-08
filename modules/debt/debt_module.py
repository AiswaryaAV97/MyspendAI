# modules/debt/debt_module.py - FULLY CORRECTED VERSION

import os
import json
from datetime import datetime
from flask import Blueprint, render_template, request, session, flash, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
from bson import ObjectId

from modules.db import debt_collection, users_collection

# Blueprint
debt_bp = Blueprint(
    "debt",
    __name__,
    template_folder="../templates/debt",
    static_folder="../static"
)

# Load ML model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "debt_strategy_model_realistic_v5.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "debt_scaler_realistic_v5.pkl")

debt_model = joblib.load(MODEL_PATH)
debt_scaler = joblib.load(SCALER_PATH)
print("✅ Debt model and scaler loaded successfully")

# ==================== AUTO SUGGESTED DEBT TYPES ====================
DEBT_TYPES = {
    "Credit Card": {"typical_rate": 19.99},
    "Personal Loan": {"typical_rate": 10.5},
    "Student Loan": {"typical_rate": 5.5},
    "Auto Loan": {"typical_rate": 6.5},
    "Medical Debt": {"typical_rate": 8.0},
    "Payday Loan": {"typical_rate": 400.0},
    "Mortgage": {"typical_rate": 7.0}
}

# ==================== AI STRATEGY PREDICTION ====================
def predict_strategy_with_confidence(debts, monthly_budget):
    X = np.array([[np.mean([d['rate'] for d in debts]),
                   np.mean([d['balance'] for d in debts]),
                   len(debts),
                   monthly_budget,
                   0]])
    X_scaled = debt_scaler.transform(X)

    pred = debt_model.predict(X_scaled)[0]
    proba = debt_model.predict_proba(X_scaled)[0]

    strategy = "Avalanche" if pred == 1 else "Snowball"
    confidence = round(max(proba) * 100, 1)

    # reasoning
    avg_rate = np.mean([d['rate'] for d in debts])
    rate_variance = np.std([d['rate'] for d in debts])
    balance_variance = np.std([d['balance'] for d in debts])

    if strategy == "Avalanche":
        if rate_variance > 5:
            reason = f"High interest rate variance ({rate_variance:.1f}%). Avalanche saves more on interest."
        else:
            reason = "Avalanche minimizes interest cost."
    else:
        reason = "Snowball creates faster psychological wins."

    return {
        "strategy": strategy,
        "confidence": confidence,
        "reason": reason,
        "alternative": "Snowball" if strategy == "Avalanche" else "Avalanche"
    }

# ==================== DEBT PAYOFF CALCULATOR ====================
def calculate_debt_payoff(debts, monthly_budget, strategy="snowball"):
    debts = [d.copy() for d in debts]
    total_interest = [0] * len(debts)
    monthly_snapshots = []
    payment_history = []

    month = 0
    while any(d['balance'] > 0 for d in debts):
        month += 1
        remaining_budget = monthly_budget
        month_payments = []

        # minimum + interest
        for i, d in enumerate(debts):
            if d['balance'] <= 0:
                continue

            interest = d['balance'] * d['rate'] / 100 / 12
            total_interest[i] += interest

            min_payment = min(d['min_payment'], d['balance'] + interest)
            principal = min_payment - interest

            d['balance'] -= principal
            remaining_budget -= min_payment

            month_payments.append({
                "debt": d['type'],
                "interest": round(interest, 2),
                "principal": round(principal, 2),
                "total": round(min_payment, 2)
            })

        # extra payments
        if remaining_budget > 0:
            if strategy == "snowball":
                sorted_debts = sorted([d for d in debts if d['balance'] > 0],
                                      key=lambda x: x['balance'])
            else:
                sorted_debts = sorted([d for d in debts if d['balance'] > 0],
                                      key=lambda x: x['rate'],
                                      reverse=True)

            for d in sorted_debts:
                if remaining_budget <= 0:
                    break
                extra = min(remaining_budget, d['balance'])
                d['balance'] -= extra
                remaining_budget -= extra

        monthly_snapshots.append([d['balance'] for d in debts])

        if month > 600:
            break

    return {
        "months": month,
        "total_interest": round(sum(total_interest), 2),
        "interest_per_debt": [round(i, 2) for i in total_interest],
        "monthly_snapshots": monthly_snapshots
    }

# ==================== PAYOFF ORDER FUNCTION 📌 NEW ====================
def build_payoff_order(debts, strategy):
    if strategy == "Avalanche":
        ordered = sorted(debts, key=lambda d: d["rate"], reverse=True)
    else:
        ordered = sorted(debts, key=lambda d: d["balance"])

    return [{
        "position": i + 1,
        "name": d["type"],
        "rate": d["rate"],
        "balance": d["balance"]
    } for i, d in enumerate(ordered)]

# ==================== INSIGHTS GENERATOR ====================
def generate_ai_insights(debts, snowball, avalanche, recommended):
    insights = []
    total_debt = sum(d['balance'] for d in debts)
    avg_rate = np.mean([d['rate'] for d in debts])

    insights.append({
        "type": "info",
        "icon": "💰",
        "title": "Total Debt Overview",
        "message": f"You have ${total_debt:,.2f} total debt at an average {avg_rate:.1f}% interest."
    })

    return insights

# ==================== CHART BUILDER ====================
def create_comparison_chart(debts, snowball, avalanche, recommended):
    months_range = list(range(1, max(len(snowball['monthly_snapshots']),
                                     len(avalanche['monthly_snapshots'])) + 1))

    fig = go.Figure()

    for i, d in enumerate(debts):
        snowball_balances = [m[i] for m in snowball['monthly_snapshots']]
        fig.add_trace(go.Scatter(x=months_range[:len(snowball_balances)],
                                 y=snowball_balances,
                                 mode='lines',
                                 name=f"{d['type']} (Snowball)",
                                 line=dict(width=2)))

        avalanche_balances = [m[i] for m in avalanche['monthly_snapshots']]
        fig.add_trace(go.Scatter(x=months_range[:len(avalanche_balances)],
                                 y=avalanche_balances,
                                 mode='lines',
                                 name=f"{d['type']} (Avalanche)",
                                 line=dict(width=2, dash="dot")))

    fig.update_layout(title=f"AI Recommended: {recommended}",
                      xaxis_title="Month",
                      yaxis_title="Balance ($)",
                      template="plotly_white",
                      height=500)

    return json.dumps(fig, cls=PlotlyJSONEncoder)

# ==================== MAIN ROUTE ====================
@debt_bp.route("/planner/debt", methods=["GET", "POST"])
def planner():
    if "user" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    username = session["user"]

    if request.method == "POST":
        action = request.form.get("action")

        # ---------- ADD DEBT (WITH AUTO-NUMBERING) ----------
        if action == "add_debt":
            try:
                base_type = request.form.get("debt_type")
                balance = float(request.form.get("balance"))
                rate = float(request.form.get("rate"))
                min_payment = float(request.form.get("min_payment"))

                if balance <= 0 or min_payment <= 0:
                    flash("⚠️ Invalid values.", "error")
                    return redirect(url_for("debt.planner"))

                existing = list(debt_collection.find({
                    "username": username,
                    "debt_type": {"$regex": f"^{base_type}"}
                }))

                debt_type = f"{base_type} {len(existing) + 1}"

                debt_collection.insert_one({
                    "username": username,
                    "debt_type": debt_type,
                    "balance": balance,
                    "rate": rate,
                    "min_payment": min_payment,
                    "added_date": datetime.utcnow()
                })

                flash(f"✅ {debt_type} added!", "success")

            except:
                flash("⚠️ Invalid number format.", "error")

        # ---------- SET BUDGET ----------
        elif action == "set_budget":
            try:
                amount = float(request.form.get("monthly_budget"))
                users_collection.update_one(
                    {"username": username},
                    {"$set": {"debt_budget": amount}},
                    upsert=True
                )
                flash("Monthly budget updated!", "success")
            except:
                flash("⚠️ Invalid amount.", "error")

        # ---------- REMOVE ----------
        elif action == "remove_debt":
            debt_collection.delete_one({
                "_id": ObjectId(request.form["debt_id"]),
                "username": username
            })
            flash("Debt removed.", "success")

        # ---------- CLEAR ALL ----------
        elif action == "clear_all":
            debt_collection.delete_many({"username": username})
            flash("All debts cleared.", "success")

        return redirect(url_for("debt.planner"))

    # ---------- GET REQUEST ----------
    user_debts = list(debt_collection.find({"username": username}))
    user_doc = users_collection.find_one({"username": username})
    monthly_budget = user_doc.get("debt_budget", 0) if user_doc else 0

    debts = [{
        "id": str(d["_id"]),
        "type": d["debt_type"],
        "balance": float(d["balance"]),
        "rate": float(d["rate"]),
        "min_payment": float(d["min_payment"])
    } for d in user_debts]

    analysis = None
    if debts and monthly_budget > 0:
        snowball = calculate_debt_payoff(debts, monthly_budget, "snowball")
        avalanche = calculate_debt_payoff(debts, monthly_budget, "avalanche")
        ai = predict_strategy_with_confidence(debts, monthly_budget)
        recommended = snowball if ai["strategy"] == "Snowball" else avalanche

        payoff_order = build_payoff_order(debts, ai["strategy"])

        chart_json = create_comparison_chart(debts, snowball, avalanche, ai["strategy"])

        analysis = {
            "snowball": snowball,
            "avalanche": avalanche,
            "recommendation": ai,
            "chart_json": chart_json,
            "payoff_order": payoff_order,
            "total_debt": sum(d["balance"] for d in debts)
        }

    return render_template(
        "debt.html",
        debts=debts,
        monthly_budget=monthly_budget,
        analysis=analysis,
        debt_types=list(DEBT_TYPES.keys())
    )
