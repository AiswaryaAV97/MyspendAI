# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Dict, Any
from flask import current_app
from bson import ObjectId


#  DB helper 

def get_db():
    
    mongo = current_app.config.get("MONGO_CLIENT")
    if not mongo:
        raise RuntimeError("MONGO_CLIENT not configured on app")
    return mongo.get_default_database()


#  Data access

def load_tax_brackets(db, year: int, province_code: str):
    """
    Load federal + provincial tax brackets from Mongo.

    tax_brackets collection structure:

    Federal:
    {
      "year": 2025,
      "type": "federal",
      "province": null,
      "brackets": [
        {"threshold": 0, "rate": 0.15},
        ...
      ]
    }

    Provincial (Ontario):
    {
      "year": 2025,
      "type": "provincial",
      "province": "ON",
      "brackets": [...],
      "credits_meta": { "basic_personal_amount": 12747, ... }
    }
    """
    federal_doc = db.tax_brackets.find_one({
        "year": year,
        "type": "federal",
        "province": None,
    })

    provincial_doc = db.tax_brackets.find_one({
        "year": year,
        "type": "provincial",
        "province": province_code,
    })

    if not federal_doc or not provincial_doc:
        raise RuntimeError(
            f"Tax brackets not configured in DB for year={year}, province={province_code}. "
            f"Please insert them into 'tax_brackets'."
        )

    return federal_doc["brackets"], provincial_doc["brackets"]


def load_basic_credits(db, year: int, province_code: str) -> Dict[str, float]:
    """
    Load key credit parameters from DB.

    Preferred: tax_credits collection:
    {
      "year": 2025,
      "province": "ON",
      "basic_personal_amount": 12747,
      "spouse_max": 10823,
      "eligible_dep_max": 10823,
      "age_amount_max": 6223,
      "disability_amount": 10298,
      "disability_supp_max": 6007,
      "med_threshold_max": 2885
      ...
    }

    Fallback: provincial tax_brackets.credits_meta.
    """
    doc = db.tax_credits.find_one({
        "year": year,
        "province": province_code,
    })

    basic_personal_amount = None
    spouse_max = 10823
    eligible_dep_max = 10823
    age_amount_max = 6223
    disability_amount = 10298
    disability_supp = 6007
    med_threshold_max = 2885

    if doc:
        basic_personal_amount = doc.get("basic_personal_amount", None)
        spouse_max = doc.get("spouse_max", spouse_max)
        eligible_dep_max = doc.get("eligible_dep_max", eligible_dep_max)
        age_amount_max = doc.get("age_amount_max", age_amount_max)
        disability_amount = doc.get("disability_amount", disability_amount)
        disability_supp = doc.get("disability_supp_max", disability_supp)
        med_threshold_max = doc.get("med_threshold_max", med_threshold_max)

    if basic_personal_amount is None:
        prov = db.tax_brackets.find_one({
            "year": year,
            "type": "provincial",
            "province": province_code,
        })
        if prov and "credits_meta" in prov:
            basic_personal_amount = prov["credits_meta"].get("basic_personal_amount")

    if basic_personal_amount is None:
        # Last resort fallback: still allow app to run
        basic_personal_amount = 15000

    return {
        "basic_personal_amount": float(basic_personal_amount),
        "spouse_max": float(spouse_max),
        "eligible_dep_max": float(eligible_dep_max),
        "age_amount_max": float(age_amount_max),
        "disability_amount": float(disability_amount),
        "disability_supp_max": float(disability_supp),
        "med_threshold_max": float(med_threshold_max),
    }


# =============== Core calculations ===============

def calculate_tax_from_brackets(taxable_income: float, brackets: list) -> float:
    """
    Progressive tax calculation: apply each bracket in order of threshold.
    """
    tax = 0.0
    prev_threshold = 0.0

    for b in sorted(brackets, key=lambda x: x["threshold"]):
        threshold = float(b["threshold"])
        rate = float(b["rate"])

        if taxable_income > threshold:
            amount = taxable_income - max(prev_threshold, threshold)
            if amount < 0:
                amount = 0
            tax += amount * rate
            prev_threshold = threshold
        else:
            break

    return max(tax, 0.0)


def compute_taxable_income(profile: Dict[str, Any]) -> float:
    """
    Approximate taxable income by subtracting major deductions from gross income.
    """
    income = float(profile.get("income", 0.0))

    rrsp = float(profile.get("rrsp_contrib", 0.0))
    childcare = float(profile.get("childcare_expenses", 0.0))
    union_dues = float(profile.get("union_dues", 0.0))
    employment_exp = float(profile.get("employment_expenses", 0.0))

    deductions = rrsp + childcare + union_dues + employment_exp

    taxable_income = max(income - deductions, 0.0)
    return taxable_income


def estimate_non_refundable_credits(
    profile: Dict[str, Any],
    year: int,
    province_code: str,
    db,
) -> Tuple[float, Dict[str, Any]]:
    """
    Approximate key non-refundable credits:
    - Basic personal
    - Spousal / eligible dependant
    - Age amount
    - Disability
    - Tuition
    - Medical
    - Donations
    Plus benefit hints (GST/HST, CWB, ON LIFT, OTB).
    """
    income = float(profile.get("income", 0.0))
    tuition = float(profile.get("tuition_paid", 0.0))
    age = profile.get("age", None)
    has_spouse = bool(profile.get("has_spouse", False))
    spouse_income = float(profile.get("spouse_income", 0.0))
    num_dep = int(profile.get("num_dependants", 0))
    donations = float(profile.get("donations", 0.0))
    medical = float(profile.get("medical_expenses", 0.0))
    is_disabled = bool(profile.get("is_disabled", False))
    has_disabled_dep = bool(profile.get("has_disabled_dependant", False))

    cfg = load_basic_credits(db, year, province_code)
    basic_personal_amount = cfg["basic_personal_amount"]
    spouse_max = cfg["spouse_max"]
    eligible_dep_max = cfg["eligible_dep_max"]
    age_amount_max = cfg["age_amount_max"]
    disability_amount_base = cfg["disability_amount"]
    disability_supp_max = cfg["disability_supp_max"]
    med_threshold_max = cfg["med_threshold_max"]

    # Use 20% as a rough combined federal+prov credit rate
    r = 0.20

    total = 0.0

    # 1) Basic personal
    basic_credit = basic_personal_amount * r
    total += basic_credit

    # 2) Spousal / eligible dependant
    spouse_amount = 0.0
    eligible_dep_amount = 0.0

    if has_spouse:
        spouse_amount = spouse_max
        if spouse_income > 0:
            spouse_amount = max(spouse_amount - spouse_income, 0)
        total += spouse_amount * r
    elif num_dep > 0:
        eligible_dep_amount = eligible_dep_max
        total += eligible_dep_amount * r

    # 3) Age amount (65+)
    age_amount = 0.0
    if age is not None and age >= 65:
        age_amount = age_amount_max
        total += age_amount * r

    # 4) Disability
    disability_amount = 0.0
    if is_disabled:
        disability_amount += disability_amount_base
    if has_disabled_dep:
        disability_amount += disability_supp_max
    total += disability_amount * r

    # 5) Tuition
    tuition_credit = tuition * 0.15
    total += tuition_credit

    # 6) Medical � above lesser of 3% of income and med_threshold_max
    med_credit = 0.0
    if medical > 0 and income > 0:
        threshold = min(0.03 * income, med_threshold_max)
        med_eligible = max(medical - threshold, 0)
        med_credit = med_eligible * 0.15
        total += med_credit

    # 7) Donations � 20% on first 200, 40% above
    don_credit = 0.0
    if donations > 0:
        first = min(donations, 200)
        rest = max(donations - 200, 0)
        don_credit = first * 0.20 + rest * 0.40
        total += don_credit

    # Benefit hints (very rough)
    gst_hst_likely = income < 55000
    cwb_likely = 8000 < income < 40000
    on_liiftc_likely = province_code == "ON" and income < 45000
    on_otb_possible = province_code == "ON" and income < 50000

    breakdown = {
        "basic_personal_amount": round(basic_personal_amount, 2),
        "basic_personal_credit": round(basic_credit, 2),
        "spousal_amount": round(spouse_amount, 2),
        "eligible_dep_amount": round(eligible_dep_amount, 2),
        "age_amount": round(age_amount, 2),
        "disability_amount": round(disability_amount, 2),
        "tuition_amount": round(tuition, 2),
        "tuition_credit": round(tuition_credit, 2),
        "medical_expenses": round(medical, 2),
        "medical_credit": round(med_credit, 2),
        "donations": round(donations, 2),
        "donation_credit": round(don_credit, 2),
        "gst_hst_likely": gst_hst_likely,
        "cwb_likely": cwb_likely,
        "on_liiftc_likely": on_liiftc_likely,
        "on_otb_possible": on_otb_possible,
    }

    return total, breakdown


#  Profile loading 

def load_user_profile_from_db(db, user_id: Optional[str]) -> Dict[str, Any]:
    """
    Auto-fill profile from DB collections. You can simplify if you want.
    """
    profile = {
        "income": 0.0,
        "province": "ON",
        "tuition_paid": 0.0,
        "rrsp_contrib": 0.0,
        "tfsa_contrib": 0.0,
        "fhsa_contrib": 0.0,
        "is_student": False,
        "tax_paid": 0.0,

        "has_spouse": False,
        "spouse_income": 0.0,
        "num_dependants": 0,
        "age": None,
        "is_disabled": False,
        "has_disabled_dependant": False,
        "childcare_expenses": 0.0,
        "donations": 0.0,
        "medical_expenses": 0.0,
        "union_dues": 0.0,
        "employment_expenses": 0.0,
        "rent_or_property_tax": 0.0,
    }

    if not user_id:
        return profile

    user_doc = db.users.find_one({"_id": ObjectId(user_id)})
    if user_doc:
        profile["income"] = float(user_doc.get("annual_income", profile["income"]))
        profile["province"] = user_doc.get("province", profile["province"])
        profile["age"] = user_doc.get("age", profile["age"])
        profile["tax_paid"] = float(user_doc.get("tax_paid", profile["tax_paid"]))
        profile["has_spouse"] = bool(user_doc.get("has_spouse", profile["has_spouse"]))
        profile["spouse_income"] = float(user_doc.get("spouse_income", profile["spouse_income"]))
        profile["num_dependants"] = int(user_doc.get("num_dependants", profile["num_dependants"]))

    edu_doc = db.education.find_one({"user_id": user_id})
    if edu_doc:
        profile["tuition_paid"] = float(edu_doc.get("tuition_paid", profile["tuition_paid"]))
        profile["is_student"] = bool(edu_doc.get("is_student", profile["is_student"]))

    acc_doc = db.registered_accounts.find_one({"user_id": user_id})
    if acc_doc:
        profile["rrsp_contrib"] = float(acc_doc.get("rrsp_contribution", profile["rrsp_contrib"]))
        profile["tfsa_contrib"] = float(acc_doc.get("tfsa_contribution", profile["tfsa_contrib"]))
        profile["fhsa_contrib"] = float(acc_doc.get("fhsa_contribution", profile["fhsa_contrib"]))

    # Other details could be in a separate finances collection, etc.
    fin_doc = db.finances.find_one({"user_id": user_id})
    if fin_doc:
        profile["childcare_expenses"] = float(fin_doc.get("childcare_expenses", profile["childcare_expenses"]))
        profile["donations"] = float(fin_doc.get("donations", profile["donations"]))
        profile["medical_expenses"] = float(fin_doc.get("medical_expenses", profile["medical_expenses"]))
        profile["union_dues"] = float(fin_doc.get("union_dues", profile["union_dues"]))
        profile["employment_expenses"] = float(fin_doc.get("employment_expenses", profile["employment_expenses"]))
        profile["rent_or_property_tax"] = float(fin_doc.get("rent_or_property_tax", profile["rent_or_property_tax"]))

    return profile


#  Main estimate 

def build_tax_estimate(profile: Dict[str, Any], year: int, province_code: str) -> Dict[str, Any]:
    """
    Combine brackets, deductions, credits, and tax paid to estimate
    tax payable and refund/balance owing.
    """
    db = get_db()

    federal_brackets, provincial_brackets = load_tax_brackets(db, year, province_code)

    income = float(profile.get("income", 0.0))
    taxable_income = compute_taxable_income(profile)

    federal_tax = calculate_tax_from_brackets(taxable_income, federal_brackets)
    provincial_tax = calculate_tax_from_brackets(taxable_income, provincial_brackets)
    gross_tax = federal_tax + provincial_tax

    credits_total, credits_breakdown = estimate_non_refundable_credits(
        profile, year, province_code, db
    )

    estimated_tax_payable = max(gross_tax - credits_total, 0.0)

    tax_paid = float(profile.get("tax_paid", 0.0))
    refund = tax_paid - estimated_tax_payable

    result = {
        "year": year,
        "province": province_code,
        "income": round(income, 2),
        "taxable_income": round(taxable_income, 2),
        "federal_tax": round(federal_tax, 2),
        "provincial_tax": round(provincial_tax, 2),
        "gross_tax_before_credits": round(gross_tax, 2),
        "non_refundable_credits": round(credits_total, 2),
        "estimated_tax_payable": round(estimated_tax_payable, 2),
        "tax_paid": round(tax_paid, 2),
        "refund": round(refund, 2),
        "effective_tax_rate": round(
            estimated_tax_payable / income, 4
        ) if income > 0 else 0.0,
        "credits_breakdown": credits_breakdown,
        "bracket_source": "mongo",
    }

    return result
def suggest_account_contributions(profile: Dict[str, Any], estimate: Dict[str, Any]):
    """
    Very simple rule-of-thumb suggestions for FHSA, RRSP, TFSA.

    - Aim for 15% of gross income going into registered accounts.
    - Fill FHSA first (up to $8,000 in the year).
    - Then split the rest 60% RRSP / 40% TFSA.

    Returns a list of dicts for the template table.
    """
    income = float(profile.get("income", 0.0))

    # Treat these as annual contributions
    rrsp_current = float(profile.get("rrsp_contrib", 0.0))
    tfsa_current = float(profile.get("tfsa_contrib", 0.0))
    fhsa_current = float(profile.get("fhsa_contrib", 0.0))

    # Target: 15% of income
    target_total = income * 0.15 if income > 0 else 0.0
    already_planned = rrsp_current + tfsa_current + fhsa_current
    extra_needed = max(target_total - already_planned, 0.0)

    recommendations = []

    # 1) FHSA � prioritize for first home (simple cap 8k / year)
    fhsa_cap = 8000.0
    fhsa_room = max(fhsa_cap - fhsa_current, 0.0)
    fhsa_extra = min(extra_needed, fhsa_room)
    extra_needed -= fhsa_extra
    fhsa_recommended_annual = fhsa_current + fhsa_extra

    recommendations.append({
        "account": "FHSA",
        "recommended_annual": round(fhsa_recommended_annual, 2),
        "recommended_monthly": round(fhsa_recommended_annual / 12.0, 2),
        "reason": "Use FHSA first for home savings; contributions are deductible and growth is tax-free.",
    })

    # 2) Split remaining between RRSP and TFSA
    rrsp_extra = extra_needed * 0.60
    tfsa_extra = extra_needed * 0.40

    rrsp_recommended_annual = rrsp_current + rrsp_extra
    tfsa_recommended_annual = tfsa_current + tfsa_extra

    recommendations.append({
        "account": "RRSP",
        "recommended_annual": round(rrsp_recommended_annual, 2),
        "recommended_monthly": round(rrsp_recommended_annual / 12.0, 2),
        "reason": "RRSP helps reduce your taxable income at your current marginal tax rate.",
    })

    recommendations.append({
        "account": "TFSA",
        "recommended_annual": round(tfsa_recommended_annual, 2),
        "recommended_monthly": round(tfsa_recommended_annual / 12.0, 2),
        "reason": "TFSA gives tax-free growth and flexible withdrawals for medium-term goals.",
    })

    return recommendations

