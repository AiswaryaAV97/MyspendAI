# Main Flask application for SmartSpendAI
import os
import threading
import webbrowser
from functools import wraps
from datetime import datetime


from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    session,
    flash,
    request,
    jsonify,
)
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
import pandas as pd
import csv

# =========================================================
# 1) Load environment variables
# =========================================================
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

FLASK_SECRET_KEY = os.getenv(
    "FLASK_SECRET_KEY", "fallback-secret-key-CHANGE-IN-PRODUCTION"
)
MONGO_URI = os.getenv("MONGODB_URI")

# =========================================================
# 2) Flask app setup
# =========================================================
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# =========================================================
# 3) Configure MongoDB Atlas
# =========================================================
if not MONGO_URI:
    raise RuntimeError("❌ MONGODB_URI missing in .env file")

mongo_client = MongoClient(MONGO_URI)
app.config["MONGO_CLIENT"] = mongo_client
print("✓ Connected to MongoDB Atlas (MONGO_CLIENT configured)")

# =========================================================
# 4) Import db and collections
# =========================================================
from modules.db import (
    db,
    users_collection,
    user_profiles,
    expenses_collection,
    debt_collection,
    investment_collection,
)

# =========================================================
# 5) Import blueprints
# =========================================================
from modules.user_profile import profile_bp
from modules.investment.routes import bp as investment_bp
from modules.debt.debt_module import debt_bp
from modules.mortgage_module.routes import mortgage_bp
from modules.expense.expense_module import expense_bp
from modules.carloan.carloan_module import carloan_bp
from modules.insurance.insurance_module import insurance_bp
from modules.tax.routes import bp as tax_bp
from modules.retirement.retirement import bp as retirement_bp

# =========================================================
# 6) Chatbot / RAG imports
# =========================================================
from modules.chatbot_share.chatbot_module.rag_service import (
    answer_user,
    ingest_chunks_for_user,
)
from modules.chatbot_share.chatbot_module.parsers import parse_document
from modules.chatbot_share.bank_statement_parser import parse_bank_statement  # PDF parser

# =========================================================
# 7) Register blueprints
# =========================================================
app.register_blueprint(profile_bp, url_prefix="/profile")
app.register_blueprint(investment_bp, url_prefix="/investment")
app.register_blueprint(debt_bp)
app.register_blueprint(mortgage_bp, url_prefix="/mortgage")
app.register_blueprint(expense_bp)
app.register_blueprint(carloan_bp, url_prefix="/carLoanCalc")
app.register_blueprint(insurance_bp, url_prefix="/planner")
app.register_blueprint(tax_bp)
app.register_blueprint(retirement_bp)

# =========================================================
# 8) Login required decorator
# =========================================================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Please login first", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return wrapper


# =========================================================
# 9) Routes: Home / Register / Login / Logout
# =========================================================
@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""
        email    = (request.form.get("email") or "").strip().lower()

        # 1) basic validation
        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("register"))

        # 2) build document (omit email if blank)
        doc = {
            "username": username,
            "password_hash": generate_password_hash(password),
            "roles": ["user"],
            "is_active": True,
            "created_at": datetime.utcnow(),
        }
        if email:
            doc["email"] = email

        # 3) insert and handle duplicates
        try:
            users_collection.insert_one(doc)
            flash("Account created. Please log in.", "success")
            return redirect(url_for("profile"))

        except errors.DuplicateKeyError as e:
            # Which unique field collided?
            flash("User already exists, please log in.", "error")
        return render_template("register.html", error="User already exists")

    # GET -> show form
    return render_template("register.html", title="Create account")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            flash("Please provide username and password", "error")
            return redirect(url_for("login"))

        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user.get("password", ""), password):
            session["user"] = username
            flash("Login successful", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password", "error")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.pop("user", None)
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))


# =========================================================
# 10) Forgot password (simple direct reset)
# =========================================================
ALLOW_INSECURE_RESET = True  # if False, will require matching email


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        email_in = (request.form.get("email") or "").strip().lower()
        pw1 = request.form.get("password") or ""
        pw2 = request.form.get("confirm_password") or ""

        # Validate new password
        if len(pw1) < 6 or pw1 != pw2:
            flash("Passwords must match and be at least 6 characters.", "error")
            return redirect(url_for("forgot_password"))

        user = users_collection.find_one({"username": username})

        if not user:
            # Do not reveal user existence
            flash("If the account exists, the password has been updated.", "success")
            return redirect(url_for("login"))

        # If secure mode enabled, check email
        if not ALLOW_INSECURE_RESET and user.get("email"):
            if not email_in or email_in != user["email"].lower():
                flash("Please enter the registered email for this account.", "error")
                return redirect(url_for("forgot_password"))

        # Update password hash
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"password": generate_password_hash(pw1)}},
        )

        flash("Password updated successfully.", "success")
        return redirect(url_for("login"))

    # GET → show reset form
    return render_template(
        "forgot_password_direct.html", title="Forgot password", card_size="form-card"
    )


# =========================================================
# 11) Dashboard + Planner Router
# =========================================================
@app.route("/dashboard")
@login_required
def dashboard():
    username = session["user"]
    if not user_profiles.find_one({"username": username}):
        return redirect(url_for("profile.profile"))

    return render_template(
        "dashboard.html",
        username=username,
        logged_in=True,
        APP_NAME="SmartSpendAI",
    )


@app.route("/planner/<key>", methods=["GET"])
@login_required
def planner_router(key):
    mapping = {
        "expense": "expense.planner",
        "investment": "investment.investment_planner",
        "debt": "debt.planner",
        "realestate": "mortgage.mortgage_page",
        "carloan": "carloan.car_loan_calc",
        "retirement": "retirement.retirement_planner",
        "insurance": "insurance.insurance_home",
        "tax": "tax.tax_planner",
    }

    target = mapping.get(key)
    if target:
        return redirect(url_for(target))

    flash("Unknown planner selected.", "error")
    return redirect(url_for("dashboard"))


# =========================================================
# 12) Chatbot route  (/chat)
# =========================================================
@app.route("/chat", methods=["GET", "POST"])
@login_required
def chat():
    username = session["user"]

    if request.method == "POST":
        # Support JSON + form-encoded
        if request.is_json:
            data = request.get_json(silent=True) or {}
            question = (data.get("message") or "").strip()
        else:
            question = (request.form.get("message") or "").strip()

        if not question:
            return jsonify({"answer": "Please enter a question.", "context": ""})

        try:
            resp = answer_user(username, question, top_k=10)
        except Exception as e:
            print(f"Error in answer_user: {e}")
            return (
                jsonify(
                    {
                        "answer": "Sorry, there was an error answering your question.",
                        "context": "",
                    }
                ),
                500,
            )

        # resp is already JSON-safe dict from answer_user
        return jsonify(resp)

    return render_template("chat.html", username=username)


# =========================================================
# 13) Upload documents (Bank Statements + Contracts)
# =========================================================
@app.route("/upload", methods=["POST"])
@login_required
def upload_document():
    username = session.get("user")
    if not username:
        return jsonify({"success": False, "message": "Not authenticated"}), 401

    file = request.files.get("file")
    if not file or file.filename.strip() == "":
        return jsonify({"success": False, "message": "No file uploaded"}), 400

    doc_type = (request.form.get("doc_type") or "bank_statement").strip()
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    # Save temporarily
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", filename)
    file.save(save_path)

    try:
        # --------------------------
        # 1) BANK STATEMENTS
        # --------------------------
        if doc_type == "bank_statement":
            transactions = []

            # PDF → use your custom parser
            if ext == ".pdf":
                data = parse_bank_statement(save_path)
                if not data.get("transactions"):
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": "No transactions found in PDF.",
                            }
                        ),
                        400,
                    )

                for t in data["transactions"]:
                    amt = float(t["amount"])
                    transactions.append(
                        {
                            "date": t["date"],
                            "description": t["description"],
                            "category": t["category"],
                            "amount": float(abs(amt)),
                            "type": "income" if amt > 0 else "expense",
                            "source": "bank_statement",
                        }
                    )

            # CSV
            elif ext == ".csv":
                with open(save_path, newline="", encoding="utf-8") as f:
                    rows = csv.DictReader(f)
                    for row in rows:
                        raw_amt = row.get("Amount") or "0"
                        try:
                            amt = float(raw_amt)
                        except ValueError:
                            # Strip $ and commas if present
                            amt = float(
                                raw_amt.replace("$", "").replace(",", "") or "0"
                            )
                        transactions.append(
                            {
                                "date": row.get("Date", ""),
                                "description": row.get("Description", ""),
                                "category": row.get("Category", "Bank"),
                                "amount": float(abs(amt)),
                                "type": "income" if amt > 0 else "expense",
                                "source": "bank_statement",
                            }
                        )

            # Excel
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(save_path)
                for _, row in df.iterrows():
                    raw_amt = row.get("Amount", 0)
                    try:
                        amt = float(raw_amt)
                    except ValueError:
                        amt = float(str(raw_amt).replace("$", "").replace(",", "") or 0)
                    transactions.append(
                        {
                            "date": str(row.get("Date", "")),
                            "description": str(row.get("Description", "")),
                            "category": row.get("Category", "Bank"),
                            "amount": float(abs(amt)),
                            "type": "income" if amt > 0 else "expense",
                            "source": "bank_statement",
                        }
                    )

            else:
                return (
                jsonify(
                    {
                        "success": False,
                        "message": "Unsupported bank statement file type.",
                    }
                ),
                400,
            )

            if not transactions:
                return jsonify({"success": False, "message": "No transactions found."}), 400

            # Save transactions to MongoDB
            for txn in transactions:
                expenses_collection.insert_one(
                    {
                        "username": username,
                        **txn,
                    }
                )

            # Also ingest RAG chunks so chatbot can use them
            chunks = []
            for txn in transactions:
                chunks.append(
                    {
                        "text": (
                            f"Transaction on {txn['date']}: {txn['description']} - "
                            f"Category: {txn['category']}, Amount: ${txn['amount']:.2f}"
                        ),
                        "source": f"bank_statement_{username}",
                        "chunk_index": len(chunks),
                    }
                )

            if chunks:
                ingest_chunks_for_user(username, chunks)

            return jsonify(
                {
                    "success": True,
                    "message": f"Bank statement uploaded! {len(transactions)} transactions saved.",
                }
            )

        # --------------------------
        # 2) CONTRACTS / POLICIES
        # --------------------------
        elif doc_type == "contract":
            chunks = parse_document(save_path)

            if not chunks:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": "Could not extract text from contract.",
                        }
                    ),
                    400,
                )

            # Tag chunks as contract for nicer audit queries
            for c in chunks:
                meta = c.get("snippet_meta") or {}
                meta["doc_type"] = "contract"
                meta["filename"] = filename
                c["snippet_meta"] = meta

            ingest_chunks_for_user(username, chunks)

            return jsonify(
                {
                    "success": True,
                    "message": f"Contract uploaded! {len(chunks)} chunks added to AI memory.",
                }
            )

        else:
            return jsonify({"success": False, "message": "Invalid document type."}), 400

    except Exception as e:
        print(f"Error in /upload: {e}")
        return (
            jsonify(
                {"success": False, "message": f"Error parsing file: {str(e)}"}
            ),
            500,
        )

    finally:
        # Clean up temp file
        if os.path.exists(save_path):
            os.remove(save_path)


# =========================================================
# 14) Run Flask app
# =========================================================
if __name__ == "__main__":

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")

    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False)
