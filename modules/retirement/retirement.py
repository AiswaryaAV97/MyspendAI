
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
)
from datetime import datetime
from math import pow
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

bp = Blueprint(
    "retirement",
    __name__,
    template_folder="../../templates",
    static_folder="../../static",
)

# ML MODEL: Risk profile classifier



class RiskProfileModel:
    def __init__(self):
        # Synthetic training data:
        # [age, years_to_retire, income, riskTolerance]
        X = np.array([
            # Conservative examples,Usually older
            [55, 10, 40000, 1],
            [60, 5, 60000, 1],
            [50, 15, 35000, 2],
            # Balanced examples,Middle-aged
            [40, 25, 50000, 3],
            [35, 30, 60000, 3],
            [45, 20, 70000, 3],
            # Aggressive examples,Younger (more time to recover from losses)
            [25, 40, 45000, 4],
            [30, 35, 55000, 5],
            [28, 37, 80000, 4],
        ])

        y = np.array([
            "Conservative",
            "Conservative",
            "Conservative",
            "Balanced",
            "Balanced",
            "Balanced",
            "Aggressive",
            "Aggressive",
            "Aggressive",
        ])

        # Small pipeline: scaling + decision tree
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ])
        self.model.fit(X, y)

    def predict_profile(self, age, retire_age, income, risk_tolerance):
        years_to_retire = max(retire_age - age, 1)
        x = np.array([[age, years_to_retire, income, risk_tolerance]])
        label = self.model.predict(x)[0]

        if label == "Conservative":
            expected_return = 0.04
        elif label == "Aggressive":
            expected_return = 0.08
        else:
            expected_return = 0.06

        return label, expected_return


risk_model = RiskProfileModel()

# Helpers: CPP / OAS / GIS


def estimate_cpp(current_age, retire_age, current_income, cpp_start_age=65):
    """
    Very rough CPP estimate at cpp_start_age.
    """
    YMPE = 75000  # assumed Year's Maximum Pensionable Earnings
    max_cpp_65 = 1300  # max monthly CPP at 65 (approx)

    earnings_ratio = min(1.0, current_income / YMPE)
    years = max(retire_age - current_age, 0)
    years_ratio = min(1.0, years / 39.0)

    cpp_65 = max_cpp_65 * earnings_ratio * years_ratio

    # Age adjustment
    months_diff = int((cpp_start_age - 65) * 12)
    # -0.6%/month before 65, +0.7%/month after 65
    if months_diff < 0:
        # earlier than 65: reduction
        factor = 1 + (months_diff * 0.006)
    else:
        factor = 1 + (months_diff * 0.007)

    cpp_monthly = cpp_65 * factor
    cpp_monthly = max(cpp_monthly, 0)
    return cpp_monthly


def estimate_oas(retire_age, years_in_canada_after_18=None, oas_start_age=65):
    """
    Rough OAS estimate, ignoring clawback.
    """
    full_oas_65 = 715  # monthly

    if years_in_canada_after_18 is None:
        # assume lived in Canada from 18 to retire_age
        years_in_canada_after_18 = max(retire_age - 18, 0)

    ratio = min(1.0, years_in_canada_after_18 / 40.0)
    oas_65 = full_oas_65 * ratio

    if oas_start_age != 65:
        # For now treat same as CPP adjustment
        months_diff = int((oas_start_age - 65) * 12)
        if months_diff < 0:
            factor = 1 + (months_diff * 0.006)
        else:
            factor = 1 + (months_diff * 0.007)
        oas_monthly = oas_65 * factor
    else:
        oas_monthly = oas_65

    return max(oas_monthly, 0)


def estimate_gis(total_gov_income_annual):
    """
    Extremely rough GIS approximation (for low-income seniors).
    """
    threshold = 20000  # annual
    if total_gov_income_annual >= threshold:
        return 0.0
    # Somewhat arbitrary formula just to show effect
    gis_annual = max(0.0, 10000 - 0.5 * total_gov_income_annual)
    return gis_annual


# Simulation logic


def project_retirement(
    current_age,
    retire_age,
    life_expectancy,
    current_income,
    income_growth_rate,
    rrsp_balance,
    tfsa_balance,
    fhsa_balance,
    nonreg_balance,
    rrsp_monthly,
    tfsa_monthly,
    fhsa_monthly,
    nonreg_monthly,
    expected_return_rate,
    inflation_rate,
    target_replacement_rate,
    cpp_monthly_65,
    oas_monthly_65,
):
    """
    Returns:
      series: list of yearly entries
      coverage_ratio, projected_real_income_at_retirement, target_real_income
    """
    series = []

    current_year = datetime.now().year
    age = current_age
    year = current_year

    income = current_income  # nominal

    # Convert monthly contributions to annual
    rrsp_annual_c = rrsp_monthly * 12
    tfsa_annual_c = tfsa_monthly * 12
    fhsa_annual_c = fhsa_monthly * 12
    nonreg_annual_c = nonreg_monthly * 12

    def to_real(nominal_value, years_from_now):
        return nominal_value / pow((1 + inflation_rate), years_from_now)

    retirement_real_income = None
    target_real_income = None

    for t in range(life_expectancy - current_age + 1):
        is_working = age < retire_age
        years_from_now = t

        # Government benefits: start at 65 only
        if age >= 65:
            cpp_annual = cpp_monthly_65 * 12
            oas_annual = oas_monthly_65 * 12
            gis_annual = estimate_gis(cpp_annual + oas_annual)
        else:
            cpp_annual = 0.0
            oas_annual = 0.0
            gis_annual = 0.0

        # Accumulation vs retirement phase
        if is_working:
            # Contributions add, then growth
            rrsp_balance = (rrsp_balance + rrsp_annual_c) * (1 + expected_return_rate)
            tfsa_balance = (tfsa_balance + tfsa_annual_c) * (1 + expected_return_rate)
            fhsa_balance = (fhsa_balance + fhsa_annual_c) * (1 + expected_return_rate)
            nonreg_balance = (nonreg_balance + nonreg_annual_c) * (1 + expected_return_rate)

            total_portfolio_withdrawal = 0.0
        else:
            # Retirement: withdraw using a safe withdrawal rate (4% of total at retirement)
            safe_withdrawal_rate = 0.04

            total_portfolio = rrsp_balance + tfsa_balance + fhsa_balance + nonreg_balance
            total_portfolio_withdrawal = total_portfolio * safe_withdrawal_rate

            # Withdraw proportionally
            total_before = total_portfolio if total_portfolio > 0 else 1
            rrsp_withdraw = total_portfolio_withdrawal * (rrsp_balance / total_before)
            tfsa_withdraw = total_portfolio_withdrawal * (tfsa_balance / total_before)
            fhsa_withdraw = total_portfolio_withdrawal * (fhsa_balance / total_before)
            nonreg_withdraw = total_portfolio_withdrawal * (nonreg_balance / total_before)

            rrsp_balance = (rrsp_balance - rrsp_withdraw) * (1 + expected_return_rate)
            tfsa_balance = (tfsa_balance - tfsa_withdraw) * (1 + expected_return_rate)
            fhsa_balance = (fhsa_balance - fhsa_withdraw) * (1 + expected_return_rate)
            nonreg_balance = (nonreg_balance - nonreg_withdraw) * (1 + expected_return_rate)

        total_retirement_income_nominal = cpp_annual + oas_annual + gis_annual + total_portfolio_withdrawal
        total_retirement_income_real = to_real(total_retirement_income_nominal, years_from_now)

        # Determine target based on final salary at retirement
        if age == retire_age - 1:
            final_salary_at_retirement = income * (1 + income_growth_rate)
            target_real_income = target_replacement_rate * to_real(
                final_salary_at_retirement, years_from_now + 1
            )

        if age == retire_age:
            retirement_real_income = total_retirement_income_real

        series.append({
            "age": age,
            "year": year,
            "rrsp": rrsp_balance,
            "tfsa": tfsa_balance,
            "fhsa": fhsa_balance,
            "nonRegistered": nonreg_balance,
            "cppAnnual": cpp_annual,
            "oasAnnual": oas_annual,
            "gisAnnual": gis_annual,
            "totalRetirementIncomeNominal": total_retirement_income_nominal,
            "totalRetirementIncomeReal": total_retirement_income_real,
        })

        age += 1
        year += 1
        income = income * (1 + income_growth_rate)

    if retirement_real_income is None or target_real_income is None or target_real_income == 0:
        coverage_ratio = 0.0
    else:
        coverage_ratio = retirement_real_income / target_real_income

    return series, coverage_ratio, retirement_real_income, target_real_income


def find_required_contribution(
    base_inputs,
    cpp_monthly_65,
    oas_monthly_65,
    max_total_monthly=2000,
    step=50,
):
    """
    Simple search over total monthly contribution to reach coverage_ratio >= 1.
    Splits contributions as 50% RRSP / 50% TFSA (v1).
    """
    (
        current_age,
        retire_age,
        life_expectancy,
        current_income,
        income_growth_rate,
        rrsp_balance,
        tfsa_balance,
        fhsa_balance,
        nonreg_balance,
        expected_return_rate,
        inflation_rate,
        target_replacement_rate,
    ) = base_inputs

    required = None
    last_series = None
    last_cov = None
    last_ret_income = None
    last_target = None

    for total_monthly in range(0, max_total_monthly + step, step):
        rrsp_m = total_monthly * 0.5
        tfsa_m = total_monthly * 0.5

        series, cov, ret_real, target_real = project_retirement(
            current_age,
            retire_age,
            life_expectancy,
            current_income,
            income_growth_rate,
            rrsp_balance,
            tfsa_balance,
            fhsa_balance,
            nonreg_balance,
            rrsp_m,
            tfsa_m,
            0.0,
            0.0,
            expected_return_rate,
            inflation_rate,
            target_replacement_rate,
            cpp_monthly_65,
            oas_monthly_65,
        )

        last_series, last_cov, last_ret_income, last_target = series, cov, ret_real, target_real

        if cov >= 1.0:
            required = total_monthly
            break

    if required is None:
        required = max_total_monthly

    return required, last_series, last_cov, last_ret_income, last_target


def evaluate_allocation_schemes(
    base_inputs,
    cpp_monthly_65,
    oas_monthly_65,
    total_monthly_contribution,
):
    """
    Try 3 allocation schemes:
      - RRSP-heavy
      - Balanced
      - TFSA-heavy
    and pick one with highest retirement_real_income.
    """
    (
        current_age,
        retire_age,
        life_expectancy,
        current_income,
        income_growth_rate,
        rrsp_balance,
        tfsa_balance,
        fhsa_balance,
        nonreg_balance,
        expected_return_rate,
        inflation_rate,
        target_replacement_rate,
    ) = base_inputs

    schemes = [
        ("RRSP heavy", 0.7, 0.3),
        ("Balanced", 0.5, 0.5),
        ("TFSA heavy", 0.3, 0.7),
    ]

    best_scheme = None
    best_income = -1
    best_series = None
    best_cov = None
    best_target = None

    for name, rrsp_ratio, tfsa_ratio in schemes:
        rrsp_m = total_monthly_contribution * rrsp_ratio
        tfsa_m = total_monthly_contribution * tfsa_ratio

        series, cov, ret_real, target_real = project_retirement(
            current_age,
            retire_age,
            life_expectancy,
            current_income,
            income_growth_rate,
            rrsp_balance,
            tfsa_balance,
            fhsa_balance,
            nonreg_balance,
            rrsp_m,
            tfsa_m,
            0.0,
            0.0,
            expected_return_rate,
            inflation_rate,
            target_replacement_rate,
            cpp_monthly_65,
            oas_monthly_65,
        )

        if ret_real is not None and ret_real > best_income:
            best_income = ret_real
            best_scheme = name
            best_series = series
            best_cov = cov
            best_target = target_real

    return best_scheme, best_series, best_cov, best_income, best_target


# Routes


@bp.route("/retirement", methods=["GET"])
def retirement_planner():
    # Simple session-based protection (similar to your other planners)
    if "user" not in session:
        return redirect(url_for("login"))

    username = session["user"]
    return render_template("retirement.html", current_user=username)


@bp.route("/retirementPlan", methods=["POST"])
def retirement_plan():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "success": False,
            "error": "Invalid or missing JSON payload.",
        }), 400

    try:
        # 1) Read inputs from frontend
        current_age = int(data["currentAge"])
        retire_age = int(data["retireAge"])
        life_expectancy = int(data.get("lifeExpectancy", 90))

        current_income = float(data["currentAnnualIncome"])

        income_growth = float(data.get("expectedIncomeGrowthRate") or 0.02)     # decimal
        inflation = float(data.get("inflationRate") or 0.02)                     # decimal
        target_replacement = float(data.get("targetReplacementRate") or 0.7)     # decimal

        rrsp_balance = float(data.get("rrspBalance") or 0.0)
        tfsa_balance = float(data.get("tfsaBalance") or 0.0)
        fhsa_balance = float(data.get("fhsaBalance") or 0.0)
        nonreg_balance = float(data.get("nonRegisteredBalance") or 0.0)

        rrsp_m = float(data.get("rrspMonthlyContribution") or 0.0)
        tfsa_m = float(data.get("tfsaMonthlyContribution") or 0.0)
        fhsa_m = float(data.get("fhsaMonthlyContribution") or 0.0)
        nonreg_m = float(data.get("nonRegisteredMonthlyContribution") or 0.0)
        total_monthly_contrib = rrsp_m + tfsa_m + fhsa_m + nonreg_m

        risk_tol = int(data.get("riskTolerance") or 3)
        explicit_ret = data.get("expectedReturnRate", None)  # decimal or None

        cpp_flag = bool(data.get("cppContributing", True))
        oas_flag = bool(data.get("oasEligible", True))
        # gis handled automatically inside project_retirement

        # 2) ML risk profile + expected return
        if explicit_ret is not None:
            expected_ret = float(explicit_ret)
            risk_label = "Custom (override)"
        else:
            risk_label, expected_ret = risk_model.predict_profile(
                current_age, retire_age, current_income, risk_tol
            )

        # 3) CPP & OAS estimates
        cpp_monthly_65 = estimate_cpp(current_age, retire_age, current_income) if cpp_flag else 0.0
        oas_monthly_65 = estimate_oas(retire_age) if oas_flag else 0.0

        # 4) Prepare base inputs tuple
        base_inputs = (
            current_age,
            retire_age,
            life_expectancy,
            current_income,
            income_growth,
            rrsp_balance,
            tfsa_balance,
            fhsa_balance,
            nonreg_balance,
            expected_ret,
            inflation,
            target_replacement,
        )

        # 5) Current plan: use actual contributions
        current_series, current_cov, current_ret_income, target_income_real = project_retirement(
            current_age,
            retire_age,
            life_expectancy,
            current_income,
            income_growth,
            rrsp_balance,
            tfsa_balance,
            fhsa_balance,
            nonreg_balance,
            rrsp_m,
            tfsa_m,
            fhsa_m,
            nonreg_m,
            expected_ret,
            inflation,
            target_replacement,
            cpp_monthly_65,
            oas_monthly_65,
        )

        # 6) Required monthly contribution to hit 100% coverage
        required_total_monthly, req_series, req_cov, req_ret_income, req_target = find_required_contribution(
            base_inputs,
            cpp_monthly_65,
            oas_monthly_65,
        )

        # 7) Best allocation scheme for that required monthly amount
        best_scheme_name, best_series, best_cov, best_income, best_target = evaluate_allocation_schemes(
            base_inputs,
            cpp_monthly_65,
            oas_monthly_65,
            required_total_monthly,
        )

        message = (
            "You are on track for your target retirement income."
            if current_cov >= 1.0
            else "You may need to increase your monthly savings to fully reach your target."
        )

        return jsonify({
            "success": True,
            "riskProfile": risk_label,
            "expectedReturnRateUsed": expected_ret,
            "currentPlan": {
                "coverageRatio": current_cov,
                "totalMonthlyContribution": total_monthly_contrib,
                "series": current_series,
            },
            "recommendedPlan": {
                "requiredTotalMonthlyContribution": required_total_monthly,
                "bestSchemeName": best_scheme_name,
                "series": best_series,
            },
            "message": message,
        }), 200

    except Exception as e:
        print("Error in /retirementPlan:", e)
        return jsonify({
            "success": False,
            "error": "Server error while computing plan.",
        }), 500
