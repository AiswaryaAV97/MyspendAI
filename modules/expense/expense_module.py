"""
EXPENSE MODULE – SmartSpendAI
Handles:
- Income & savings goal
- Manual expense entry
- Receipt OCR upload
- Bank statement PDF extraction
- Automatic analysis pipeline (run_full_pipeline)
- Charts, trends, recommendations
"""

import os
import re
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from flask import (
    Blueprint, render_template, request, session, flash,
    redirect, url_for
)
from werkzeug.utils import secure_filename
from bson import ObjectId
from pymongo import MongoClient

# ML Analyzer
from modules.expense.expense_ml import (
    run_full_pipeline,
    categorizer as SMART_CATEGORIZER
)

# ------------------------------------------------------------------
# PDF Parsing
# ------------------------------------------------------------------
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# ------------------------------------------------------------------
# Flask Blueprint
# ------------------------------------------------------------------
expense_bp = Blueprint(
    "expense",
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# MongoDB Connection
# ------------------------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = client["SmartSpendAI"]

users_collection = db["users"]
expense_collection = db["expenses"]
analysis_collection = db["expense_analysis"]

print("✓ MongoDB connected for Expense module")

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
CATEGORIES = [
    "Groceries", "Dining", "Transportation", "Housing",
    "Utilities", "Shopping", "Health", "Entertainment",
    "Insurance", "Education", "Others"
]

FREQ_MULTIPLIER = {
    "Weekly": 4.33,
    "Bi-Weekly": 2.16,
    "Monthly": 1.0,
    "One-Time": 1.0,
}

# ------------------------------------------------------------------
# OCR Setup
# ------------------------------------------------------------------
OCR_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image

    t_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract"
    ]
    for p in t_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            OCR_AVAILABLE = True
            print(f"✓ Tesseract OCR found at: {p}")
            break
except Exception as e:
    print("⚠ OCR unavailable:", e)

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def extract_receipt_data(image_path):
    """OCR extract total + AI category."""
    result = {
        "amount": 0.0,
        "date": str(date.today()),
        "category": "Others",
    }
    if not OCR_AVAILABLE:
        return result

    try:
        text = pytesseract.image_to_string(Image.open(image_path))

        # capture all float amounts
        prices = re.findall(r"\$?\s*(\d+\.\d{2})", text)
        if prices:
            result["amount"] = max(float(p) for p in prices)

        # AI category
        try:
            cinfo = SMART_CATEGORIZER.categorize(text, result["amount"])
            result["category"] = cinfo.get("category", "Others")
        except Exception:
            pass

    except Exception as e:
        print("OCR Error:", e)

    return result


# ------------------------------------------------------------------
# TD PDF STATEMENT PARSER
# ------------------------------------------------------------------
def parse_td_pdf_statement(file_path):
    """
    Clean TD Credit Card Statement Parser.
    Returns list of {date, amount, description}
    """

    if not PDF_AVAILABLE:
        return []

    try:
        reader = PdfReader(file_path)
    except Exception:
        return []

    txns = []
    in_table = False
    current = None

    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
    }

    def td_to_date(x):
        m = re.match(r"([A-Z]{3})(\d{1,2})", x)
        if not m:
            return str(date.today())
        return f"2025-{month_map[m.group(1)]}-{int(m.group(2)):02d}"

    header_pattern = re.compile(r"TRANSACTION\s+POSTING", re.I)
    stop_pattern = re.compile(r"TOTAL\s+NEW\s+BALANCE", re.I)
    noise = ["ACCOUNT", "SUMMARY", "PAYMENT", "CREDIT", "ANNUAL", "AVAILABLE"]

    line_regex = re.compile(
        r"^([A-Z]{3}\d{1,2})\s+([A-Z]{3}\d{1,2})?\s*\$?\s*([\d,]+\.\d{2})\s+(.+)$"
    )

    for page in reader.pages:
        text = page.extract_text() or ""

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue

            # start of table
            if not in_table and header_pattern.search(line):
                in_table = True
                continue

            # end of table
            if in_table and stop_pattern.search(line):
                if current:
                    txns.append(current)
                in_table = False
                current = None
                continue

            if not in_table:
                continue

            if any(n in line.upper() for n in noise):
                continue

            m = line_regex.match(line)
            if m:
                if current:
                    txns.append(current)

                trans_date, post_date, amt_str, desc = m.groups()

                amount = safe_float(amt_str.replace(",", ""), 0)

                current = {
                    "date": td_to_date(trans_date),
                    "amount": amount,
                    "description": desc.strip(),
                }

    if current:
        txns.append(current)

    print(f"✓ Extracted {len(txns)} TD transactions")
    return txns


# ------------------------------------------------------------------
# RECOMPUTE INSIGHTS
# ------------------------------------------------------------------
def recompute_insights(username, income):
    """
    Compute analysis using expense_ml and store result.

    IMPORTANT:
    - Automatically detects how many distinct months of data exist.
    - Adjusts income = monthly_income × detected_months for ML analysis.
    """
    try:
        txns = list(expense_collection.find({"username": username}))
        if len(txns) < 3:
            analysis_collection.delete_many({"username": username})
            return

        df = pd.DataFrame(txns)

        # Use lowercase column names expected by ML engine
        df["amount"] = df.get("Amount").astype(float)
        df["category"] = df.get("Category")
        df["monthly"] = df.get("monthly", df["amount"])
        df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
        df["frequency"] = df.get("frequency", df.get("Frequency", "One-Time"))

        # ------------------------------------------------------------
        # AUTO-DETECT number of months from uploaded expenses
        # ------------------------------------------------------------
        try:
            df_clean = df.dropna(subset=["date"]).copy()
            df_clean["month"] = df_clean["date"].dt.to_period("M")
            detected_months = len(df_clean["month"].unique())
            if detected_months < 1:
                detected_months = 1
        except Exception:
            detected_months = 1

        adjusted_income = income * detected_months

        # Run ML pipeline
        result = run_full_pipeline(df, adjusted_income)
        result["detected_months"] = detected_months

        # Save new analysis
        analysis_collection.delete_many({"username": username})
        analysis_collection.insert_one({
            "username": username,
            "generated_at": datetime.utcnow(),
            **result
        })

    except Exception as e:
        print("Insight Error:", e)


# ------------------------------------------------------------------
# MAIN ROUTE
# ------------------------------------------------------------------
@expense_bp.route("/planner/expense", methods=["GET", "POST"])
def planner():
    if "user" not in session:
        return redirect(url_for("login"))

    username = session["user"]

    # user profile
    profile = users_collection.find_one({"username": username}) or {}
    income = safe_float(profile.get("income", 5000))          # Monthly income
    savings_goal = safe_float(profile.get("savings_goal", 0))

    # ------------------------------------------------------------------
    # POST HANDLING
    # ------------------------------------------------------------------
    if request.method == "POST":
        action = request.form.get("action")

        # ---------------- Income ----------------
        if action == "save_income":
            try:
                new_income = safe_float(request.form.get("income"))
                if new_income > 0:
                    users_collection.update_one(
                        {"username": username},
                        {"$set": {"income": new_income}},
                        upsert=True
                    )
                    income = new_income
                    flash("Income updated", "success")
                    recompute_insights(username, income)
                else:
                    flash("Invalid income", "error")
            except Exception:
                flash("Error updating income", "error")

        # ---------------- Savings Goal ----------------
        elif action == "save_goal":
            try:
                goal = safe_float(request.form.get("savings_goal"))
                if goal >= 0:
                    users_collection.update_one(
                        {"username": username},
                        {"$set": {"savings_goal": goal}},
                        upsert=True
                    )
                    savings_goal = goal
                    flash("Savings goal saved", "success")
            except Exception:
                flash("Error saving goal", "error")

        # ---------------- Add Expense ----------------
        elif action == "add_expense":
            try:
                amount = safe_float(request.form.get("amount"))
                category = request.form.get("category") or "Others"
                freq = request.form.get("frequency")
                desc = request.form.get("description", "")
                monthly = amount * FREQ_MULTIPLIER.get(freq, 1)

                expense_collection.insert_one({
                    "username": username,
                    "date": str(date.today()),
                    "Amount": amount,
                    "Category": category,
                    "frequency": freq,
                    "monthly": monthly,
                    "description": desc,
                    "created_at": datetime.utcnow()
                })

                flash("Expense added", "success")
                recompute_insights(username, income)

            except Exception as e:
                flash(f"Error: {e}", "error")

        # ---------------- Edit ----------------
        elif action == "edit_expense":
            try:
                eid = request.form.get("expense_id")
                amount = safe_float(request.form.get("amount"))
                category = request.form.get("category") or "Others"
                freq = request.form.get("frequency")
                desc = request.form.get("description", "")

                monthly = amount * FREQ_MULTIPLIER.get(freq, 1)

                expense_collection.update_one(
                    {"_id": ObjectId(eid), "username": username},
                    {"$set": {
                        "Amount": amount,
                        "Category": category,
                        "frequency": freq,
                        "monthly": monthly,
                        "description": desc
                    }}
                )

                flash("Updated successfully", "success")
                recompute_insights(username, income)

            except Exception:
                flash("Update failed", "error")

        # ---------------- Delete ----------------
        elif action == "remove_expense":
            try:
                rid = request.form.get("remove_id")
                expense_collection.delete_one({"_id": ObjectId(rid), "username": username})
                flash("Deleted", "success")
                recompute_insights(username, income)
            except Exception:
                flash("Delete failed", "error")

        # ---------------- Delete All ----------------
        elif action == "remove_all":
            expense_collection.delete_many({"username": username})
            analysis_collection.delete_many({"username": username})
            flash("All expenses cleared", "success")

        # ---------------- Receipt Upload ----------------
        elif action == "upload_receipt":
            try:
                file = request.files.get("receipt_file")
                if file and file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    name = secure_filename(file.filename)
                    path = UPLOAD_FOLDER / f"{username}_{int(time.time())}_{name}"
                    file.save(path)

                    data = extract_receipt_data(path)

                    expense_collection.insert_one({
                        "username": username,
                        "date": data["date"],
                        "Amount": data["amount"],
                        "Category": data["category"],
                        "frequency": "One-Time",
                        "monthly": data["amount"],
                        "description": "Receipt",
                        "created_at": datetime.utcnow()
                    })

                    flash("Receipt added", "success")
                    recompute_insights(username, income)
                else:
                    flash("Invalid image", "error")

            except Exception as e:
                flash(f"OCR Error: {e}", "error")

        # ---------------- PDF Upload ----------------
        elif action == "upload_pdf":
            try:
                if not PDF_AVAILABLE:
                    flash("PDF parsing unavailable", "error")
                else:
                    file = request.files.get("statement_file")
                    if file and file.filename.lower().endswith(".pdf"):
                        name = secure_filename(file.filename)
                        path = UPLOAD_FOLDER / f"{username}_{int(time.time())}_{name}"
                        file.save(path)

                        txns = parse_td_pdf_statement(str(path))
                        if not txns:
                            flash("No transactions found", "warning")
                        else:
                            added = 0
                            for t in txns:
                                amount = t["amount"]
                                desc = t["description"]

                                cat = "Others"
                                try:
                                    cinfo = SMART_CATEGORIZER.categorize(desc, amount)
                                    cat = cinfo.get("category", "Others")
                                except Exception:
                                    pass

                                expense_collection.insert_one({
                                    "username": username,
                                    "date": t["date"],
                                    "Amount": amount,
                                    "Category": cat,
                                    "frequency": "One-Time",
                                    "monthly": amount,
                                    "description": desc,
                                    "source": "pdf",
                                    "created_at": datetime.utcnow()
                                })
                                added += 1

                            flash(f"Imported {added} transactions", "success")
                            recompute_insights(username, income)
                    else:
                        flash("Invalid PDF", "error")
            except Exception as e:
                flash(f"PDF Error: {e}", "error")

        return redirect(url_for("expense.planner"))

    # ------------------------------------------------------------------
    # GET → DISPLAY PAGE
    # ------------------------------------------------------------------
    raw_expenses = list(expense_collection.find({"username": username}).sort("date", -1))

    expenses = [{
        "_id": str(e["_id"]),
        "Category": e.get("Category", "Others"),
        "Amount": safe_float(e.get("Amount")),
        "frequency": e.get("frequency", "One-Time"),
        "monthly": safe_float(e.get("monthly")),
        "date": str(e.get("date", "")),
        "description": e.get("description", "")
    } for e in raw_expenses]

    df = pd.DataFrame(expenses) if expenses else pd.DataFrame()

    # ---------------- Charts ----------------
    charts = {}
    if not df.empty:
        try:
            import plotly.express as px

            cat_totals = df.groupby("Category")["monthly"].sum().reset_index()

            pie = px.pie(cat_totals, names="Category", values="monthly")
            charts["pie"] = pie.to_json()

            bar = px.bar(cat_totals, x="Category", y="monthly")
            charts["bar"] = bar.to_json()

        except Exception as e:
            print("Chart Error:", e)

    # ---------------- Trend Detection ----------------
    trends = None
    if not df.empty:
        try:
            import numpy as np
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            monthly_series = df.groupby(df["date"].dt.to_period("M"))["monthly"].sum()

            if len(monthly_series) >= 2:
                x = np.arange(len(monthly_series))
                slope, _ = np.polyfit(x, monthly_series.values, 1)

                trends = {
                    "trend": "increasing" if slope > 0 else "decreasing",
                    "trend_percentage": round(
                        abs(slope) / (monthly_series.mean() + 1e-8) * 100, 1
                    ),
                    "average_monthly": round(monthly_series.mean(), 2),
                    "last_month": round(monthly_series.iloc[-1], 2)
                }
        except Exception as e:
            print("Trend Error:", e)

    # ---------------- Insights (from ML Engine) ----------------
    analysis = None
    latest = analysis_collection.find_one(
        {"username": username},
        sort=[("generated_at", -1)]
    )

    # Default months = 1
    detected_months_for_ui = 1

    if latest:
        detected_months_for_ui = int(latest.get("detected_months", 1)) or 1

        analysis = {
            "summary": latest.get("summary", ""),
            "forecast": latest.get("forecast", {}),
            "recurring_expenses": latest.get("recurring_expenses", []),
            "recommendations": latest.get("recommendations", []),
            "category_analysis": latest.get("category_analysis", []),
            "total_spent": latest.get("total_spent", 0),
            "savings": latest.get("savings", 0),
            "savings_rate": latest.get("savings_rate", 0),
            "detected_months": detected_months_for_ui,
        }
    else:
        # If no analysis doc, we can still guess months from df
        if not df.empty and "date" in df.columns:
            try:
                tmp = df.copy()
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
                tmp = tmp.dropna(subset=["date"])
                if not tmp.empty:
                    tmp["month"] = tmp["date"].dt.to_period("M")
                    detected_months_for_ui = len(tmp["month"].unique()) or 1
            except Exception:
                detected_months_for_ui = 1

    # ---------------- Summary Cards (use income × months) ----------------
    total_spending = sum(e["monthly"] for e in expenses) if expenses else 0
    period_income = income * detected_months_for_ui  # ← this is your monthly income × months

    if period_income > 0:
        current_savings = period_income - total_spending
        savings_rate_calc = round(current_savings / period_income * 100, 1)
    else:
        current_savings = 0
        savings_rate_calc = 0

    goal_progress = None
    if savings_goal > 0:
        goal_progress = max(0, min(100, (current_savings / savings_goal) * 100))

    summary_stats = {
        "total_spending": round(total_spending, 2),
        "current_savings": round(current_savings, 2),
        "savings_rate": savings_rate_calc,
        "transaction_count": len(expenses),
        "avg_transaction": round(total_spending / max(len(expenses), 1), 2),
        "savings_goal": savings_goal,
        "goal_progress": round(goal_progress, 1) if goal_progress is not None else None,
        "detected_months": detected_months_for_ui,
        "period_income": period_income,   # not used in template yet, but available
    }

    return render_template(
        "expense.html",
        categories=CATEGORIES,
        frequencies=list(FREQ_MULTIPLIER.keys()),
        expenses=expenses,
        income=income,                    # monthly income (badge at top)
        savings_goal=savings_goal,
        charts=charts,
        trends=trends,
        analysis=analysis,
        summary_stats=summary_stats,
        pdf_available=PDF_AVAILABLE,
        ocr_available=OCR_AVAILABLE
    )
