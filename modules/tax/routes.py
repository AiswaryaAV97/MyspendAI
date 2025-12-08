import os
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    session,
)
import requests
from pymongo import MongoClient
from datetime import datetime

# ----------------------------------------
# MongoDB Setup (same style as Insurance)
# ----------------------------------------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db_mongo = client["SmartSpendAI"]

tax_collection = db_mongo["tax_history"]
print("✓ MongoDB connected for Tax module")

from .engine import (
    get_db,
    load_user_profile_from_db,
    build_tax_estimate,
    suggest_account_contributions,
)


bp = Blueprint(
    "tax",
    __name__,
    template_folder="../../templates",
    static_folder="../../static",
)


@bp.route("/tax", methods=["GET", "POST"])
def tax_planner():
    db = get_db()
    user_id = session.get("user_id")

    profile = load_user_profile_from_db(db, user_id)
    estimate = None
    advice = None
    account_recommendations = None   # 👈 initialise

    if request.method == "POST":
        # Override profile from form fields
        def f(name, default=0.0):
            return float(request.form.get(name, default) or default)

        profile.update({
            "income": f("income", profile["income"]),
            "province": request.form.get("province", profile["province"] or "ON"),
            "tuition_paid": f("tuition_paid", profile["tuition_paid"]),
            "rrsp_contrib": f("rrsp_contrib", profile["rrsp_contrib"]),
            "tfsa_contrib": f("tfsa_contrib", profile["tfsa_contrib"]),
            "fhsa_contrib": f("fhsa_contrib", profile["fhsa_contrib"]),
            "tax_paid": f("tax_paid", profile["tax_paid"]),
            "childcare_expenses": f("childcare_expenses", profile["childcare_expenses"]),
            "donations": f("donations", profile["donations"]),
            "medical_expenses": f("medical_expenses", profile["medical_expenses"]),
            "union_dues": f("union_dues", profile["union_dues"]),
            "employment_expenses": f("employment_expenses", profile["employment_expenses"]),
            "rent_or_property_tax": f("rent_or_property_tax", profile["rent_or_property_tax"]),
        })

        # Booleans / ints
        profile["is_student"] = request.form.get("is_student") == "on"
        profile["has_spouse"] = request.form.get("has_spouse") == "on"
        profile["spouse_income"] = f("spouse_income", profile["spouse_income"])
        profile["num_dependants"] = int(request.form.get("num_dependants", profile["num_dependants"] or 0))
        age_val = request.form.get("age", "")
        profile["age"] = int(age_val) if age_val else profile["age"]
        profile["is_disabled"] = request.form.get("is_disabled") == "on"
        profile["has_disabled_dependant"] = request.form.get("has_disabled_dependant") == "on"

        year = int(request.form.get("year", 2025))
        province = profile["province"]

        # build normal tax estimate
        estimate = build_tax_estimate(profile, year, province)

        #  build FHSA / RRSP / TFSA recommendations
        account_recommendations = suggest_account_contributions(profile, estimate)

        #  LLM advice 
        try:
            advice = generate_llm_advice(profile, estimate)
        except Exception as e:
            advice = f"Could not generate AI advice: {e}"

        # ----------------------------------------
        # Save tax results to MongoDB  (FIXED INDENTATION)
        # ----------------------------------------
        try:
            tax_collection.insert_one({
                "username": session.get("user", "guest"),
                "profile": profile,
                "estimate": estimate,
                "recommendations": account_recommendations,
                "advice": advice,
                "created_at": datetime.utcnow(),
            })
            print("✓ Tax record saved to MongoDB")
        except Exception as e:
            print("⚠ Error saving tax data:", e)

    return render_template(
        "tax.html",
        profile=profile,
        estimate=estimate,
        advice=advice,
        account_recommendations=account_recommendations,  
    )


@bp.route("/api/tax/estimate", methods=["POST"])
def api_tax_estimate():
    """
    Pure JSON tax estimate (no LLM).
    """
    db = get_db()
    user_id = session.get("user_id")
    body = request.get_json(silent=True) or {}

    profile = load_user_profile_from_db(db, user_id)

    def get_float(key, default):
        return float(body.get(key, default) or default)

    for k in [
        "income", "tuition_paid", "rrsp_contrib", "tfsa_contrib", "fhsa_contrib",
        "tax_paid", "childcare_expenses", "donations", "medical_expenses",
        "union_dues", "employment_expenses", "rent_or_property_tax",
    ]:
        if k in body:
            profile[k] = get_float(k, profile.get(k, 0.0))

    if "province" in body:
        profile["province"] = body["province"]
    if "is_student" in body:
        profile["is_student"] = bool(body["is_student"])
    if "has_spouse" in body:
        profile["has_spouse"] = bool(body["has_spouse"])
    if "spouse_income" in body:
        profile["spouse_income"] = get_float("spouse_income", profile["spouse_income"])
    if "num_dependants" in body:
        profile["num_dependants"] = int(body["num_dependants"])
    if "age" in body:
        profile["age"] = int(body["age"])
    if "is_disabled" in body:
        profile["is_disabled"] = bool(body["is_disabled"])
    if "has_disabled_dependant" in body:
        profile["has_disabled_dependant"] = bool(body["has_disabled_dependant"])

    year = int(body.get("year", 2025))
    province = profile["province"]

    estimate = build_tax_estimate(profile, year, province)

    return jsonify({
        "success": True,
        "profile_used": profile,
        "estimate": estimate,
    })


@bp.route("/api/tax/recommend", methods=["POST"])
def api_tax_recommend():
    """
    JSON: tax estimate + LLM-generated advice.
    """
    db = get_db()
    user_id = session.get("user_id")
    body = request.get_json(silent=True) or {}

    profile = load_user_profile_from_db(db, user_id)

    def get_float(key, default):
        return float(body.get(key, default) or default)

    for k in [
        "income", "tuition_paid", "rrsp_contrib", "tfsa_contrib", "fhsa_contrib",
        "tax_paid", "childcare_expenses", "donations", "medical_expenses",
        "union_dues", "employment_expenses", "rent_or_property_tax",
    ]:
        if k in body:
            profile[k] = get_float(k, profile.get(k, 0.0))

    if "province" in body:
        profile["province"] = body["province"]
    if "is_student" in body:
        profile["is_student"] = bool(body["is_student"])
    if "has_spouse" in body:
        profile["has_spouse"] = bool(body["has_spouse"])
    if "spouse_income" in body:
        profile["spouse_income"] = get_float("spouse_income", profile["spouse_income"])
    if "num_dependants" in body:
        profile["num_dependants"] = int(body["num_dependants"])
    if "age" in body:
        profile["age"] = int(body["age"])
    if "is_disabled" in body:
        profile["is_disabled"] = bool(body["is_disabled"])
    if "has_disabled_dependant" in body:
        profile["has_disabled_dependant"] = bool(body["has_disabled_dependant"])

    year = int(body.get("year", 2025))
    province = profile["province"]

    estimate = build_tax_estimate(profile, year, province)
    advice_text = generate_llm_advice(profile, estimate)

    return jsonify({
        "success": True,
        "estimate": estimate,
        "llm_advice": advice_text,
    })


def generate_llm_advice(profile, estimate) -> str:
   
    income = estimate["income"]
    tax_payable = estimate["estimated_tax_payable"]
    tax_paid = estimate.get("tax_paid", 0.0)
    refund = estimate.get("refund", 0.0)
    province = estimate["province"]
    year = estimate["year"]

    rrsp = profile.get("rrsp_contrib", 0.0)
    tfsa = profile.get("tfsa_contrib", 0.0)
    fhsa = profile.get("fhsa_contrib", 0.0)
    tuition = profile.get("tuition_paid", 0.0)

    credits = estimate["credits_breakdown"]

    prompt = f"""
You are a helpful Canadian tax assistant (not a CRA agent) specializing in
federal + Ontario personal income tax for {year}.
You are given a trusted tax calculation result from another system.
DO NOT recompute tax – assume the numbers are correct.

User situation:
- Year: {year}
- Province: {province}
- Employment/other income: ${income:,.2f}
- Estimated total tax payable: ${tax_payable:,.2f}
- Tax already paid (source deductions): ${tax_paid:,.2f}
- Refund (positive means money back, negative means owing): ${refund:,.2f}

Registered accounts this year:
- RRSP contributions: ${rrsp:,.2f}
- TFSA contributions: ${tfsa:,.2f}
- FHSA contributions: ${fhsa:,.2f}

Education:
- Tuition paid: ${tuition:,.2f}
- Is student: {profile.get("is_student", False)}

Credits & benefits breakdown (approximate):
{credits}

In 4–7 bullet points:
1. Say clearly if they are getting a refund or owing tax, and briefly why.
2. Suggest realistic RRSP / TFSA / FHSA strategies for next year based on this income level.
3. Mention likely federal benefits (GST/HST credit, Canada Workers Benefit) if the hints suggest.
4. For Ontario, mention LIFT or the Ontario Trillium Benefit if the income range and hints suggest.
5. If you see non-zero amounts (spousal, dependant, age, disability, medical, donations),
   briefly explain what those credits represent in simple language.
6. Keep the tone friendly and educational. This is NOT official tax or legal advice.
"""

    try:
        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "llama3",   
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 600,
            },
            timeout=60,
        )
        data = response.json()

        text = data["choices"][0]["message"]["content"]
        return text.replace("\n", "<br>")

        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        html = "<ul>"
        for ln in lines:
               html += f"<li>{ln}</li>"
        html += "</ul>"

        return html

    except Exception as e:
        return (
            "AI advice is temporarily unavailable (local LLM error). "
            f"Technical details: {e}"
        )
