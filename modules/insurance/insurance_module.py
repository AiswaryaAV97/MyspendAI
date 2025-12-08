from flask import Blueprint, render_template, request, session, flash
from modules.insurance.insurance_ml import predict_insurance
from pymongo import MongoClient
from datetime import datetime
import os

# ----------------------------------------
# Blueprint
# ----------------------------------------
insurance_bp = Blueprint("insurance", __name__, template_folder="../../templates")

# ----------------------------------------
# MongoDB Setup
# ----------------------------------------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["SmartSpendAI"]

insurance_collection = db["insurance_history"]
print("✓ MongoDB connected for Insurance module")

# ----------------------------------------
# Routes
# ----------------------------------------
@insurance_bp.route("/insurance", methods=["GET", "POST"])
@insurance_bp.route("/insurance/", methods=["GET", "POST"])
def insurance_home():

    # GET → Show empty form
    if request.method == "GET":
        return render_template("insurance.html")

    # POST → collect form inputs
    user_data = {k: request.form.get(k) for k in request.form.keys()}

    # Run the ML model
    results = predict_insurance(user_data)

    # Determine username (if logged in)
    username = session.get("user", "guest")

    # Save to MongoDB
    try:
        insurance_collection.insert_one({
            "username": username,
            "input": user_data,
            "results": results,
            "created_at": datetime.utcnow()
        })
        print("✓ Insurance record saved to MongoDB")
    except Exception as e:
        print("⚠ Error saving insurance data:", e)
        flash("Unable to save insurance record", "error")

    # Show results to user
    return render_template("insurance.html", results=results)
