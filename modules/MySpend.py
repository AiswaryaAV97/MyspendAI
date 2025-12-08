import os
from datetime import datetime
from functools import wraps
from pathlib import Path
import secrets

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from pymongo import MongoClient, errors
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# --- Paths / .env ---
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env") 

# -----------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


# -----------------------------------------------------------------------------
app.config["APP_NAME"] = "SmartSpendAI"

@app.context_processor
def inject_globals():
    return {
        "APP_NAME": app.config.get("APP_NAME", "App"),
        "logged_in": ("user" in session)
    }


# -----------------------------------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

if not MONGODB_URI:
    raise SystemExit("Missing MONGODB_URI in .env")

if not DB_NAME:
    raise SystemExit("Missing DB_NAME in .env")

if not FLASK_SECRET_KEY or FLASK_SECRET_KEY == "FLASK_SECRET_KEY":
    raise SystemExit("Set a secure FLASK_SECRET_KEY in .env")

app.secret_key = FLASK_SECRET_KEY

# --- MongoDB setup ---
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
except errors.PyMongoError as e:
    raise SystemExit(f"Could not connect to MongoDB: {e}")

db = client[DB_NAME]
users = db["users"]
users.create_index("username", unique=True)

# --- Helpers ---
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""
        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("register"))
        try:
            users.insert_one({
                "username": username,
                "password_hash": generate_password_hash(password),
                "roles": ["user"],
                "is_active": True,
                "created_at": datetime.utcnow()
            })
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except errors.DuplicateKeyError:
            flash("Username already exists.", "error")
            return redirect(url_for("register"))
        except Exception as e:
            flash(f"Registration error: {e}", "error")
            return redirect(url_for("register"))
    return render_template("register.html", title="Create account")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""
        user = users.find_one({"username": username, "is_active": True})
        if user and check_password_hash(user.get("password_hash", ""), password):
            session["user"] = username
            flash("Logged in!", "success")
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid username or password.", "error")
        return redirect(url_for("login"))
    return render_template("login.html", title="Log in")


# -----------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    
    return render_template("dashboard.html", username=session["user"], title="Dashboard",)

@app.route("/planner/<key>")
def planner(key):
    titles = {
        "expense": "Expense & Budget Analyzer",
        "investment": "Investment Planner",
        "debt": "Debt Management",
        "realestate": "Real-estate / Mortgage Advisor",
        "carloan": "Car-loan Planner",
        "retirement": "Retirement Goal Planner",
        "insurance": "Insurance & Risk Advisor",
        "tax": "Tax Recommender",
    }
    title = titles.get(key, "Planner")
    return render_template("tool.html", title=title, key=key, username=session["user"])

@app.route("/chat")
@login_required
def chat():
    return render_template("chat.html", title="AI Chatbot", username=session["user"])

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out.", "info")
    return redirect(url_for("login"))


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import webbrowser, threading
    URL = "http://127.0.0.1:5000/"

    threading.Timer(1.5, lambda: webbrowser.open_new(URL)).start()
    app.run(debug=True,use_reloader=False)

    # --- Expense & Budget Analyzer ----------------------------------------------
import json
from bson import ObjectId

expenses_col = db["expenses"]
profiles_col = db["profiles"]

def _monthly(amount, frequency):
    if frequency == "Weekly":
        return amount * 4.33
    if frequency == "Bi-Weekly":
        return amount * 2.17
    return amount  # Monthly

@app.route("/expense", methods=["GET", "POST"],endpoint="expense")
@login_required
def expense():
    username = session["user"]

    # current profile (income)
    prof = profiles_col.find_one({"username": username}) or {}
    income = float(prof.get("income") or 0)

    # handle POST actions
    if request.method == "POST":
        # update income
        if "income" in request.form:
            try:
                income_val = float(request.form.get("income") or 0)
            except ValueError:
                income_val = 0
            profiles_col.update_one(
                {"username": username},
                {"$set": {"income": income_val}},
                upsert=True,
            )
            flash("Income updated.", "success")
            return redirect(url_for("expense"))

        # remove expense
        if "remove_id" in request.form:
            try:
                expenses_col.delete_one(
                    {"_id": ObjectId(request.form["remove_id"]), "username": username}
                )
                flash("Expense removed.", "info")
            except Exception:
                flash("Could not remove item.", "error")
            return redirect(url_for("expense"))

        # add expense
        if "category" in request.form and "amount" in request.form:
            try:
                amount = float(request.form.get("amount") or 0)
            except ValueError:
                amount = 0
            doc = {
                "username": username,
                "category": (request.form.get("category") or "Other").strip(),
                "amount": amount,
                "frequency": request.form.get("frequency") or "Monthly",
                "created_at": datetime.utcnow(),
            }
            expenses_col.insert_one(doc)
            flash("Expense added.", "success")
            return redirect(url_for("expense"))

    # fetch expenses for user
    rows = list(expenses_col.find({"username": username}).sort("created_at", 1))

    # compute monthly totals + flags
    by_cat, total_monthly, expenses = {}, 0.0, []
    for d in rows:
        m = _monthly(float(d.get("amount", 0)), d.get("frequency", "Monthly"))
        total_monthly += m
        by_cat[d["category"]] = by_cat.get(d["category"], 0.0) + m
        expenses.append({
            "id": str(d["_id"]),
            "category": d["category"],
            "amount": float(d.get("amount", 0)),
            "frequency": d.get("frequency", "Monthly"),
            "monthly": m,
            "high_spending": (income > 0 and m > 0.30 * income and d["category"].lower() not in ("rent","mortgage","housing")),
        })

    # suggestions
    suggestions = []
    if income:
        left = income - total_monthly
        if left < 0:
            suggestions.append(f"High: You are over budget by ${abs(left):.2f}. Trim variable expenses.")
        else:
            suggestions.append(f"Normal: Estimated monthly savings ${left:.2f}.")
    # category heuristics
    for cat, val in by_cat.items():
        r = (val / income) if income else 0
        if cat.lower() in ("rent","mortgage","housing") and r > 0.5:
            suggestions.append("High: Housing above 50% of income - aim for ~30-35%.")
        if cat.lower() in ("dining","eating","food","entertainment") and r > 0.15:
            suggestions.append("High: Dining/entertainment above 15%. Consider a cap.")

    # plotly pie JSON
    pie_chart = None
    if by_cat:
        pie_chart = json.dumps({
            "data": [{
                "type": "pie",
                "labels": list(by_cat.keys()),
                "values": [round(v, 2) for v in by_cat.values()],
                "hole": 0.35
            }],
            "layout": { "title": "Monthly Spending by Category", "height": 400 }
        })

    categories = [
        "Rent","Utilities","Groceries","Dining","Transport","Health",
        "Education","Entertainment","Travel","Shopping","Debt","Savings","Other"
    ]

    return render_template(
        "expense.html", title="Expense & Budget Analyzer", username=session["user"]
    )
