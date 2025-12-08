# modules/insurance/insurance_ml.py

import math

def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def predict_insurance(user: dict) -> dict:
    """
    Rule-based / AI-inspired engine that recommends coverage
    for 8 insurance types and a combined risk score.

    Uses simple, explainable formulas based on:
    - Age, income, dependents
    - Assets & liabilities
    - Health (BMI, smoker, exercise)
    - Risk tolerance
    """

    # -----------------------------
    # Parse inputs from user form
    # -----------------------------
    age = safe_float(user.get("age"))
    income = safe_float(user.get("income"))
    dependents = safe_float(user.get("dependents"))

    height_cm = safe_float(user.get("height"))
    weight_kg = safe_float(user.get("weight"))

    smoker = 1 if (user.get("smoker") or "").lower() == "yes" else 0
    exercise = (user.get("exercise") or "medium").lower()
    risk_tol = (user.get("risk_tol") or "medium").lower()

    has_home = (user.get("has_home") or "no").lower()
    home_value = safe_float(user.get("home_value"))
    savings = safe_float(user.get("savings"))
    investments = safe_float(user.get("investments"))

    mortgage = safe_float(user.get("mortgage"))
    car_loan = safe_float(user.get("car_loan"))
    credit_debt = safe_float(user.get("credit_debt"))
    student_loan = safe_float(user.get("student_loan"))

    assets = home_value + savings + investments
    liabilities = mortgage + car_loan + credit_debt + student_loan

    # -----------------------------
    # Health: BMI
    # -----------------------------
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0.0

    # Avoid weird behaviour if income is 0
    if income <= 0:
        income = 50_000

    # -----------------------------
    # Global Risk Score (0–100)
    # -----------------------------
    risk_score = 0.0

    # Age: older → slightly higher risk
    risk_score += (age - 30) * 0.7

    # BMI: further from ~24 → higher risk
    risk_score += (bmi - 24) * 1.5

    # Smoking
    risk_score += smoker * 15

    # Exercise
    if exercise == "low":
        risk_score += 6
    elif exercise == "high":
        risk_score -= 4

    # Financial stress: more debt than assets
    risk_score += (liabilities - assets) / 40_000.0 * 6.0

    # Behavioural / risk tolerance
    if risk_tol == "high":
        risk_score += 4
    elif risk_tol == "low":
        risk_score -= 4

    # Base it around 35 and clamp 0–100
    risk_score = 35 + risk_score
    risk_score = max(0, min(100, risk_score))

    if risk_score < 33:
        risk_label = "Low"
    elif risk_score < 66:
        risk_label = "Medium"
    else:
        risk_label = "High"

    # Helper to build each plan
    def make_plan(name, cov, prem, priority, note):
        return {
            "name": name,
            "coverage": max(0, round(cov)),
            "premium": round(prem, 2),
            "priority": priority,  # Critical / High / Medium / Low
            "note": note,
        }

    # -----------------------------
    # 1) Life Insurance
    # -----------------------------
    life_cov = income * 10 + liabilities + dependents * 60_000 - assets * 0.3
    life_cov = max(50_000, life_cov)
    life_prem = life_cov * 0.0012  # ~0.12% / year

    life_priority = "Critical" if dependents > 0 else "High"

    life_plan = make_plan(
        "Life insurance",
        life_cov,
        life_prem,
        life_priority,
        "Income replacement + debt coverage based on your income, debts and dependents."
    )

    # -----------------------------
    # 2) Health / Extended Health
    # -----------------------------
    extra_health = max(0.0, bmi - 25) * 2_000 + smoker * 30_000
    health_cov = 80_000 + extra_health
    health_prem = health_cov * 0.003  # ~0.3% / year

    health_priority = "High" if (smoker or bmi >= 27) else "Medium"

    health_plan = make_plan(
        "Health / extended health",
        health_cov,
        health_prem,
        health_priority,
        "Covers drugs, dental, vision and paramedical costs not fully covered by provincial health plans."
    )

    # -----------------------------
    # 3) Disability Insurance
    # -----------------------------
    disability_cov = income * 0.6 * 3  # 60% income for 3 years
    disability_prem = disability_cov * 0.002

    disability_priority = "High"

    disability_plan = make_plan(
        "Disability insurance",
        disability_cov,
        disability_prem,
        disability_priority,
        "Targets about 60% of your income for ~3 years if you cannot work due to illness or injury."
    )

    # -----------------------------
    # 4) Home / Tenant Insurance
    # -----------------------------
    if has_home == "yes" and home_value > 0:
        property_cov = max(150_000, home_value * 0.8)
    else:
        # Contents + liability for renters / living with family
        property_cov = 40_000 + dependents * 10_000

    property_prem = property_cov * 0.001

    property_priority = "High" if has_home == "yes" else "Medium"

    property_plan = make_plan(
        "Home / tenant insurance",
        property_cov,
        property_prem,
        property_priority,
        "Protects your home or contents and your personal liability."
    )

    # -----------------------------
    # 5) Auto Insurance (Liability-focused)
    # -----------------------------
    has_car = car_loan > 0  # simple heuristic: if car loan exists → car owned

    if has_car:
        auto_cov = 1_000_000  # liability limit
        base_auto_prem = 900
        # Adjust based on risk_score (but keep reasonable)
        auto_prem = base_auto_prem * (1 + (risk_score - 40) / 100.0 * 0.5)
        auto_prem = max(700, auto_prem)
        auto_priority = "Critical"
    else:
        auto_cov = 0
        auto_prem = 0
        auto_priority = "Low"

    auto_plan = make_plan(
        "Auto insurance (liability focus)",
        auto_cov,
        auto_prem,
        auto_priority,
        "Basic liability-focused estimate. Real premiums depend on car type and driving record."
    )

    # -----------------------------
    # 6) Critical Illness
    # -----------------------------
    crit_cov = income * 3   # 3 years of income
    crit_prem = crit_cov * 0.0015

    crit_priority = "Medium" if age >= 30 else "Low"

    crit_plan = make_plan(
        "Critical illness insurance",
        crit_cov,
        crit_prem,
        crit_priority,
        "Lump-sum payout if you are diagnosed with a major illness (e.g., cancer, heart attack, stroke)."
    )

    # -----------------------------
    # 7) Travel Medical
    # -----------------------------
    travel_cov = 1_000_000   # common emergency medical limit
    travel_prem = 180 + max(0, age - 30) * 4 + smoker * 20

    travel_priority = "Medium"

    travel_plan = make_plan(
        "Travel medical insurance",
        travel_cov,
        travel_prem,
        travel_priority,
        "Emergency medical coverage when travelling outside your home province or Canada."
    )

    # -----------------------------
    # 8) Long-Term Care
    # -----------------------------
    base_ltc_monthly = 1_500 + max(0, age - 45) * 25 + risk_score * 5
    ltc_cov = base_ltc_monthly * 12 * 2  # ~2 years
    ltc_prem = base_ltc_monthly * 0.4

    ltc_priority = "High" if age >= 55 else "Low"

    ltc_plan = make_plan(
        "Long-term care insurance",
        ltc_cov,
        ltc_prem,
        ltc_priority,
        "Helps with nursing home or in-home care costs later in life."
    )

    # -----------------------------
    # Combine plans & totals
    # -----------------------------
    plans = [
        life_plan,
        health_plan,
        disability_plan,
        property_plan,
        auto_plan,
        crit_plan,
        travel_plan,
        ltc_plan,
    ]

    total_premium_yearly = round(sum(p["premium"] for p in plans), 2)
    total_premium_monthly = round(total_premium_yearly / 12.0, 2)

    return {
        "risk_score": int(round(risk_score)),
        "risk_label": risk_label,
        "plans": plans,
        "total_premium_yearly": total_premium_yearly,
        "total_premium_monthly": total_premium_monthly,
    }
