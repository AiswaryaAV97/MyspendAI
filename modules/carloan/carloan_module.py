# ===============================
# modules/carloan/carloan_module.py
# ===============================

import os
from datetime import datetime

import joblib
import plotly.graph_objs as go
import pandas as pd
from flask import (
    Blueprint,
    render_template,
    request,
    session,
    flash,
    redirect,
    url_for,
)
from pymongo import MongoClient

# ---------------------------------------------------
# Import Canadian car database (REAL prices + rates)
# ---------------------------------------------------
try:
    from .canadian_car_database import (
        CANADIAN_CAR_DATABASE,
        get_car_price,
        get_manufacturer_rate,
        get_all_years,
        get_all_companies,
        get_all_models,
        get_all_trims,
        get_rate_source,
        get_last_rate_update,
    )

    CANADIAN_DB_LOADED = True
except Exception as e:
    print("⚠️ canadian_car_database.py not found, using fallback:", e)
    CANADIAN_DB_LOADED = False
    CANADIAN_CAR_DATABASE = {}

# ---------------------------------------------------
# Flask Blueprint & MongoDB
# ---------------------------------------------------
carloan_bp = Blueprint("carloan", __name__, template_folder="../../templates")

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "SmartSpendAI")

client = MongoClient(MONGO_URI) if MONGO_URI else MongoClient()
db = client[DB_NAME]

carloans_collection = db["carLoans"]
users_collection = db["users"]

# ---------------------------------------------------
# Load ML model (RandomForest pipeline)
# ---------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "carloan_model.pkl")
carloan_model = None

try:
    carloan_model = joblib.load(MODEL_PATH)
    print("✅ AI model loaded: carloan_model.pkl")
except Exception as e:
    print("⚠️ Failed to load model:", e)


# ============================
# Helper Functions
# ============================
def estimate_credit_score(income: float, debt_budget: float) -> int:
    """Generate fallback credit score."""
    base_score = 650

    if income > 150_000:
        income_points = 100
    elif income > 100_000:
        income_points = 80
    elif income > 70_000:
        income_points = 60
    else:
        income_points = 40

    dti_ratio = debt_budget / income if income > 0 else 1.0

    if dti_ratio < 0.15:
        dti_points = 100
    elif dti_ratio < 0.25:
        dti_points = 80
    elif dti_ratio < 0.35:
        dti_points = 60
    else:
        dti_points = 40

    score = base_score + (income_points * 0.5) + (dti_points * 0.5)
    return max(300, min(850, int(score)))


def get_credit_tier(credit_score: int) -> str:
    if credit_score >= 750:
        return "excellent"
    elif credit_score >= 700:
        return "good"
    elif credit_score >= 650:
        return "fair"
    return "poor"


def adjust_rate_by_credit(base_rate: float, credit_score: int, term: int) -> float:
    """Adjust rate based on credit + term."""
    if credit_score >= 780:
        credit_adj = -0.5
    elif credit_score >= 720:
        credit_adj = 0.0
    elif credit_score >= 680:
        credit_adj = 1.0
    elif credit_score >= 650:
        credit_adj = 2.0
    else:
        credit_adj = 3.5

    term_adj = {24: -0.5, 36: 0.0, 48: 0.25, 60: 0.5, 72: 0.75, 84: 1.0}

    final_rate = base_rate + credit_adj + term_adj.get(term, 0.0)
    return round(max(2.99, final_rate), 2)


def calculate_monthly_payment(principal: float, rate: float, months: int) -> float:
    r = rate / 12.0 / 100.0
    if r == 0:
        return principal / months
    return principal * (r * (1 + r) ** months) / ((1 + r) ** months - 1)


def simulate_loan(principal: float, rate: float, months: int, extra: float = 0.0):
    """Simulate loan payoff and compute total interest."""
    r = rate / 12.0 / 100.0
    base_monthly = calculate_monthly_payment(principal, rate, months)

    balance = principal
    interest_total = 0.0
    month_count = 0

    while balance > 0 and month_count < 600:
        month_count += 1
        interest = balance * r
        payment = base_monthly + extra
        principal_paid = payment - interest

        if principal_paid > balance:
            principal_paid = balance
            interest = balance * r

        balance -= principal_paid
        interest_total += interest

    return {
        "months": month_count,
        "monthly_payment": round(base_monthly + extra, 2),
        "total_interest": round(interest_total, 2),
        "total_paid": round(principal + interest_total, 2),
    }


def generate_insights(
    car_key: str,
    price: float,
    monthly: float,
    income: float,
    rate: float,
    term: int,
    principal: float,
    base,
    early=None,
):
    """Build friendly text insights for the UI."""
    insights = []
    monthly_income = income / 12.0 if income > 0 else 1.0
    payment_ratio = (monthly / monthly_income) * 100.0

    # Affordability
    if payment_ratio > 15:
        insights.append(
            {
                "type": "warning",
                "icon": "⚠️",
                "title": "Affordability Alert",
                "message": f"Payment is {payment_ratio:.1f}% of your monthly income. "
                f"Most advisors suggest keeping car payments under 15%.",
            }
        )
    elif payment_ratio < 10:
        insights.append(
            {
                "type": "success",
                "icon": "✅",
                "title": "Comfortable Payment",
                "message": f"Payment is only {payment_ratio:.1f}% of your monthly income. "
                "This looks very manageable for your budget.",
            }
        )

    # Luxury warning
    if price > 60_000:
        insights.append(
            {
                "type": "info",
                "icon": "💎",
                "title": "Higher-End Vehicle",
                "message": f"{car_key} is a higher-end vehicle. "
                "Remember to consider insurance and maintenance costs too.",
            }
        )

    # Rate quality
    if rate > 10:
        insights.append(
            {
                "type": "warning",
                "icon": "📈",
                "title": "High Interest Rate",
                "message": f"{rate:.2f}% is relatively high. "
                "Improving your credit score or a bigger down payment could help you find a better rate.",
            }
        )
    elif rate < 5:
        insights.append(
            {
                "type": "success",
                "icon": "🎉",
                "title": "Strong Interest Rate",
                "message": f"You locked in a solid {rate:.2f}% rate. "
                "This keeps interest costs lower over the life of the loan.",
            }
        )

    # Long term
    if term > 72:
        insights.append(
            {
                "type": "info",
                "icon": "⏰",
                "title": "Long Loan Term",
                "message": f"{term} months keeps your payment lower but increases total interest. "
                "If your income grows later, you can consider paying extra to finish sooner.",
            }
        )

    # Early payoff
    if early and early["months"] < base["months"]:
        savings = base["total_interest"] - early["total_interest"]
        months_saved = base["months"] - early["months"]
        insights.append(
            {
                "type": "success",
                "icon": "💰",
                "title": "Early Payoff Option",
                "message": f"Adding extra payments could save about ${savings:,.0f} in interest "
                f"and shorten your loan by around {months_saved} months.",
            }
        )

    # High interest cost vs principal
    if base["total_interest"] > principal * 0.4:
        insights.append(
            {
                "type": "warning",
                "icon": "🔴",
                "title": "Interest Cost Warning",
                "message": f"You'll pay about ${base['total_interest']:,.0f} in interest, "
                "which is a large share of the loan. A bigger down payment or shorter term could help.",
            }
        )

    return insights


# ===============================
# Recommendation Engine
# ===============================
def flatten_car_database():
    """Convert nested DB into a flat list."""
    cars = []
    if not CANADIAN_DB_LOADED:
        return cars

    for year, brands in CANADIAN_CAR_DATABASE.items():
        for company, models in brands.items():
            for model, trims in models.items():
                for trim, price in trims.items():
                    cars.append(
                        {
                            "year": year,
                            "company": company,
                            "model": model,
                            "trim": trim,
                            "price": float(price),
                        }
                    )
    return cars


def ai_recommend_cars(income, credit_score, preferred_payment=None, max_results=3):
    """Return top affordable car recommendations."""
    if not CANADIAN_DB_LOADED:
        return []

    cars = flatten_car_database()
    if not cars:
        return []

    monthly_income = income / 12.0 if income > 0 else 1.0
    scored = []

    for car in cars:
        base_term = 60
        base_rate = get_manufacturer_rate(car["company"], car["model"], base_term)
        adj_rate = adjust_rate_by_credit(base_rate, credit_score, base_term)

        price = car["price"]
        principal = price - (price * 0.20)

        monthly = calculate_monthly_payment(principal, adj_rate, base_term)
        payment_ratio = (monthly / monthly_income) * 100.0

        penalty = max(0, payment_ratio - 15)
        if preferred_payment:
            pay_gap = abs(monthly - preferred_payment)
        else:
            ideal = monthly_income * 0.10
            pay_gap = abs(monthly - ideal)

        score = penalty * 2 + pay_gap

        scored.append(
            {
                "car_key": f"{car['year']} {car['company']} {car['model']} {car['trim']}",
                "year": car["year"],
                "company": car["company"],
                "model": car["model"],
                "trim": car["trim"],
                "price": price,
                "term": base_term,
                "rate": adj_rate,
                "base_rate": base_rate,
                "estimated_monthly": round(monthly, 2),
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"])
    top = scored[:max_results]

    # ML interest prediction
    if carloan_model:
        for rec in top:
            try:
                df = pd.DataFrame(
                    [
                        [
                            rec["price"],
                            rec["price"] * 0.20,
                            rec["rate"],
                            rec["term"],
                            0.0,
                            credit_score,
                        ]
                    ],
                    columns=[
                        "Car_Price",
                        "Down_Payment",
                        "Interest_Rate",
                        "Loan_Term_Months",
                        "Extra_Payment",
                        "Credit_Score",
                    ],
                )
                rec["ml_total_interest"] = float(carloan_model.predict(df)[0])
            except Exception:
                rec["ml_total_interest"] = None

    return top


# ===============================
# Main Route – Calculator + Recommender
# ===============================
@carloan_bp.route("/carLoanCalc", methods=["GET", "POST"])
def car_loan_calc():
    if "user" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    username = session["user"]

    # Get last saved data for defaults
    saved_data = carloans_collection.find_one(
        {"username": username}, sort=[("timestamp", -1)]
    )
    income = float(saved_data.get("user_income", 60000)) if saved_data else 60000.0
    debt_budget = float(saved_data.get("debt_budget", 0.0)) if saved_data else 0.0
    saved_credit = int(saved_data.get("credit_score", 0)) if saved_data else 0

    # Also check profile table for income
    profile = users_collection.find_one({"username": username}) or {}
    if profile.get("income"):
        try:
            income = float(profile["income"])
        except Exception:
            pass

    # Default score
    credit_score = (
        saved_credit if saved_credit > 0 else estimate_credit_score(income, debt_budget)
    )
    credit_tier = get_credit_tier(credit_score)

    # Data lists (empty on initial load)
    available_years = get_all_years() if CANADIAN_DB_LOADED else [2025]
    ai_recs = session.get("ai_recommendations", [])

    # Defaults for template
    chart_json = None
    car_key = None
    price = None
    selected_year = None
    selected_company = None
    selected_model = None
    selected_trim = None

    trims = []
    models = []
    companies = []
    insights = []
    ml_interest = None
    base_scenario = None
    early_scenario = None
    biweekly = None

    if request.method == "POST":
        action = request.form.get("action")

        # Update credit score if user entered manually
        if request.form.get("credit_score"):
            try:
                credit_score = int(request.form.get("credit_score"))
            except Exception:
                pass
        credit_tier = get_credit_tier(credit_score)

        # 1️⃣ Step-1: Recommendations
        if action == "ai_recommend":
            preferred = request.form.get("ai_budget")
            preferred = float(preferred) if preferred else None

            ai_recs = ai_recommend_cars(
                income=income,
                credit_score=credit_score,
                preferred_payment=preferred,
                max_results=3,
            )
            session["ai_recommendations"] = ai_recs

            return render_template(
                "carloan.html",
                years=available_years,
                companies=[],
                models=[],
                trims=[],
                ai_recommendations=ai_recs,
                credit_score=credit_score,
                credit_tier=credit_tier,
                user_income=income,
                user_debt=debt_budget,
            )

        # 2️⃣ From recommendation card → prefill Step-2
        if action == "prefill_selection":
            selected_year = int(request.form["year"])
            selected_company = request.form["company"]
            selected_model = request.form["model"]

            companies = get_all_companies(selected_year) if CANADIAN_DB_LOADED else []
            models = (
                get_all_models(selected_year, selected_company)
                if CANADIAN_DB_LOADED
                else []
            )
            trims = (
                get_all_trims(selected_year, selected_company, selected_model)
                if CANADIAN_DB_LOADED
                else []
            )

            selected_trim = trims[0] if trims else None

            return render_template(
                "carloan.html",
                years=available_years,
                selected_year=selected_year,
                companies=companies,
                selected_company=selected_company,
                models=models,
                selected_model=selected_model,
                trims=trims,
                selected_trim=selected_trim,
                ai_recommendations=ai_recs,
                credit_score=credit_score,
                credit_tier=credit_tier,
                user_income=income,
                user_debt=debt_budget,
            )

        # 3️⃣ Dropdown cascade: Year → Company
        if action == "select_year":
            selected_year = int(request.form["year"])
            companies = get_all_companies(selected_year) if CANADIAN_DB_LOADED else []

            return render_template(
                "carloan.html",
                years=available_years,
                selected_year=selected_year,
                companies=companies,
                models=[],
                trims=[],
                ai_recommendations=ai_recs,
                credit_score=credit_score,
                credit_tier=credit_tier,
                user_income=income,
                user_debt=debt_budget,
            )

        # 4️⃣ Dropdown cascade: Company → Model
        if action == "select_company":
            selected_year = int(request.form["year"])
            selected_company = request.form["company"]

            companies = get_all_companies(selected_year) if CANADIAN_DB_LOADED else []
            models = (
                get_all_models(selected_year, selected_company)
                if CANADIAN_DB_LOADED
                else []
            )

            return render_template(
                "carloan.html",
                years=available_years,
                selected_year=selected_year,
                companies=companies,
                selected_company=selected_company,
                models=models,
                trims=[],
                ai_recommendations=ai_recs,
                credit_score=credit_score,
                credit_tier=credit_tier,
                user_income=income,
                user_debt=debt_budget,
            )

        # 5️⃣ Dropdown cascade: Model → Trim
        if action == "select_model":
            selected_year = int(request.form["year"])
            selected_company = request.form["company"]
            selected_model = request.form["model"]

            companies = get_all_companies(selected_year) if CANADIAN_DB_LOADED else []
            models = (
                get_all_models(selected_year, selected_company)
                if CANADIAN_DB_LOADED
                else []
            )
            trims = (
                get_all_trims(selected_year, selected_company, selected_model)
                if CANADIAN_DB_LOADED
                else []
            )

            return render_template(
                "carloan.html",
                years=available_years,
                selected_year=selected_year,
                companies=companies,
                selected_company=selected_company,
                models=models,
                selected_model=selected_model,
                trims=trims,
                ai_recommendations=ai_recs,
                credit_score=credit_score,
                credit_tier=credit_tier,
                user_income=income,
                user_debt=debt_budget,
            )

        # 6️⃣ Final calculation (Step-2)
        if action == "calculate":
            selected_year = int(request.form["year"])
            selected_company = request.form["company"]
            selected_model = request.form["model"]
            selected_trim = request.form["trim"]

            term = int(request.form["term"])
            down_pct = float(request.form["down_payment_pct"])
            extra_payment = float(request.form["extra_payment"])

            # Price + base rate with safe fallbacks
            if CANADIAN_DB_LOADED:
                price = get_car_price(
                    selected_year, selected_company, selected_model, selected_trim
                )
                if price is None:
                    flash(
                        f"Price not found for {selected_year} {selected_company} "
                        f"{selected_model} {selected_trim}",
                        "error",
                    )
                    return redirect(url_for("carloan.car_loan_calc"))

                base_rate = get_manufacturer_rate(selected_company, selected_model, term)
            else:
                price = 35_000.0
                base_rate = 6.99

            rate = adjust_rate_by_credit(base_rate, credit_score, term)

            down_payment = price * (down_pct / 100)
            principal = price - down_payment

            # Base scenario (no extra)
            base_scenario = simulate_loan(principal, rate, term, extra=0.0)
            # Early payoff scenario (with extra)
            early_scenario = (
                simulate_loan(principal, rate, term, extra=extra_payment)
                if extra_payment > 0
                else None
            )

            biweekly = round(base_scenario["monthly_payment"] * 12 / 26, 2)

            # ML prediction
            if carloan_model:
                try:
                    df = pd.DataFrame(
                        [
                            [
                                price,
                                down_payment,
                                rate,
                                term,
                                extra_payment,
                                credit_score,
                            ]
                        ],
                        columns=[
                            "Car_Price",
                            "Down_Payment",
                            "Interest_Rate",
                            "Loan_Term_Months",
                            "Extra_Payment",
                            "Credit_Score",
                        ],
                    )
                    ml_interest = float(carloan_model.predict(df)[0])
                except Exception:
                    ml_interest = None

            car_key = (
                f"{selected_year} {selected_company} {selected_model} {selected_trim}"
            )

            insights = generate_insights(
                car_key,
                price,
                base_scenario["monthly_payment"],
                income,
                rate,
                term,
                principal,
                base_scenario,
                early_scenario,
            )

            # Build payoff chart (standard line)
            fig = go.Figure()
            months = list(range(1, base_scenario["months"] + 1))
            balance = []
            bal = principal

            for _m in months:
                interest = bal * (rate / 12.0 / 100.0)
                principal_paid = base_scenario["monthly_payment"] - interest
                bal -= principal_paid
                balance.append(max(0.0, bal))

            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=balance,
                    name="Standard Payment",
                    line=dict(width=3),
                    hovertemplate="Month: %{x}<br>Balance: $%{y:,.0f}<extra></extra>",
                )
            )

            # Extra-payment line
            if early_scenario:
                months_e = list(range(1, early_scenario["months"] + 1))
                balance_e = []
                bal = principal
                for _m in months_e:
                    interest = bal * (rate / 12.0 / 100.0)
                    principal_paid = early_scenario["monthly_payment"] - interest
                    bal -= principal_paid
                    balance_e.append(max(0.0, bal))

                fig.add_trace(
                    go.Scatter(
                        x=months_e,
                        y=balance_e,
                        name=f"With ${extra_payment:.0f}/mo Extra",
                        line=dict(width=3, dash="dot"),
                        hovertemplate="Month: %{x}<br>Balance: $%{y:,.0f}<extra></extra>",
                    )
                )

            fig.update_layout(
                template="plotly_white",
                title=f"{car_key} – Loan Payoff Timeline",
                xaxis_title="Month",
                yaxis_title="Remaining Balance ($)",
                hovermode="x unified",
            )
            chart_json = fig.to_json()

            # Save to Mongo
            carloans_collection.insert_one(
                {
                    "username": username,
                    "year": selected_year,
                    "car": car_key,
                    "price": price,
                    "down_payment": down_payment,
                    "principal": principal,
                    "rate": rate,
                    "base_rate": base_rate,
                    "term": term,
                    "monthly": base_scenario["monthly_payment"],
                    "credit_score": credit_score,
                    "user_income": income,
                    "debt_budget": debt_budget,
                    "timestamp": datetime.utcnow(),
                }
            )

            return render_template(
                "carloan.html",
                years=available_years,
                selected_year=selected_year,
                companies=get_all_companies(selected_year)
                if CANADIAN_DB_LOADED
                else [],
                selected_company=selected_company,
                models=get_all_models(selected_year, selected_company)
                if CANADIAN_DB_LOADED
                else [],
                selected_model=selected_model,
                trims=get_all_trims(selected_year, selected_company, selected_model)
                if CANADIAN_DB_LOADED
                else [],
                selected_trim=selected_trim,
                car_key=car_key,
                price=price,
                down_payment=down_payment,
                down_payment_pct=down_pct,
                principal=principal,
                rate=rate,                     # 👉 now passed to template
                term=term,
                base_scenario=base_scenario,
                early_scenario=early_scenario,
                biweekly_payment=biweekly,
                extra_payment=extra_payment,   # 👉 slider value preserved
                insights=insights,
                chart_json=chart_json,
                credit_score=credit_score,
                credit_tier=credit_tier,
                ml_interest=ml_interest,
                ai_recommendations=ai_recs,
                user_income=income,
                user_debt=debt_budget,
            )

    # GET request (first time page load)
    return render_template(
        "carloan.html",
        years=available_years,
        companies=[],
        models=[],
        trims=[],
        ai_recommendations=ai_recs,
        credit_score=credit_score,
        credit_tier=credit_tier,
        user_income=income,
        user_debt=debt_budget,
    )
