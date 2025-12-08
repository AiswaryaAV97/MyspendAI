import os
import re
from datetime import datetime
from functools import wraps

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, session, flash
)
from flask import jsonify 
from pymongo import MongoClient, errors
from dotenv import load_dotenv

# Create Blueprint for profile functionality
profile_bp = Blueprint('profile', __name__, template_folder='templates')

# --- Load environment variables ---
load_dotenv()

# --- Configuration ---
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

# --- MongoDB setup ---
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    print("✅ MongoDB connected successfully for profile module")
except errors.PyMongoError as e:
    print(f"❌ Profile module: Could not connect to MongoDB: {e}")
    client = None

if client:
    db = client[DB_NAME]
    user_profiles = db["user_profiles"]
    # Ensure unique username index for profiles
    user_profiles.create_index("username", unique=True)
else:
    user_profiles = None

# --- Authentication Decorator ---
def login_required(view):
    """Decorator to require login for profile routes"""
    @wraps(view)
    def wrapped(*args, **kwargs):
        # Use 'username' to match your main App.py session key
        if "user" not in session:
            flash("Please log in to access your profile.", "error")
            return redirect("/login")  # Redirect to your main app's login
        return view(*args, **kwargs)
    return wrapped

# --- Routes ---
@profile_bp.route("/")
def index():
    """Homepage - redirects to profile"""
    return redirect(url_for("profile.profile"))

@profile_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """Create/Edit user profile"""
    username = session["user"]
    
    if request.method == "POST":
        # Get form data - separate first and last name
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        
        # Create full name (combine first and last name for backward compatibility)
        if first_name and last_name:
            full_name = f"{first_name} {last_name}"
        elif last_name:
            full_name = last_name
        else:
            full_name = first_name if first_name else ""
        
        profile_data = {
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
            "full_name": full_name,
            "email": email,
            "phone": phone,
            "address": address,
            "date_of_birth": request.form.get("date_of_birth", ""),
            "marital_status": request.form.get("marital_status", ""),
            "number_of_dependents": int(request.form.get("number_of_dependents", 0) or 0),
            "last_updated": datetime.utcnow()
        }
        
        # Validate required fields
        if not last_name:
            flash("Last name is required.", "error")
            return redirect("/profile/profile")
        
        if not email:
            flash("Email address is required.", "error")
            return redirect("/profile/profile")
        
        if not phone:
            flash("Phone number is required.", "error")
            return redirect("/profile/profile")
        
        if not profile_data["date_of_birth"]:
            flash("Date of birth is required.", "error")
            return redirect("/profile/profile")
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            flash("Please enter a valid email address.", "error")
            return redirect("/profile/profile")
        
        # Validate phone number (basic validation - numbers, spaces, hyphens, parentheses)
        phone_clean = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace("+", "")
        if not phone_clean.isdigit() or len(phone_clean) < 10 or len(phone_clean) > 15:
            flash("Please enter a valid phone number (10-15 digits).", "error")
            return redirect("/profile/profile")
        
        # Check for duplicate email (excluding current user)
        existing_email = user_profiles.find_one({"email": email, "username": {"$ne": username}})
        if existing_email:
            flash("This email address is already registered.", "error")
            return redirect("/profile/profile")
        
        # Validate date of birth (not in future)
        try:
            birth_date = datetime.strptime(profile_data["date_of_birth"], "%Y-%m-%d")
            if birth_date > datetime.now():
                flash("Date of birth cannot be in the future.", "error")
                return redirect("/profile/profile")
        except ValueError:
            flash("Invalid date format.", "error")
            return redirect("/profile/profile")
        
        try:
            # Update existing or insert new profile
            result = user_profiles.update_one(
                {"username": username},
                {"$set": profile_data, "$setOnInsert": {"profile_created_at": datetime.utcnow()}},
                upsert=True
            )
            
            if result.upserted_id:
                flash("Profile created successfully!", "success")
            else:
                flash("Profile updated successfully!", "success")
            
            return redirect("/profile/view_profile")
            
        except Exception as e:
            flash(f"Error saving profile: {e}", "error")
            return redirect("/profile/profile")
    
    # GET request - show form with existing data
    existing_profile = user_profiles.find_one({"username": username})
    # NEW: Handle backward compatibility - split full_name if first/last names don't exist
    if existing_profile and not existing_profile.get("first_name") and not existing_profile.get("last_name"):
        full_name = existing_profile.get("full_name", "")
        if full_name:
            name_parts = full_name.split(" ", 1)  # Split into max 2 parts
            existing_profile["first_name"] = name_parts[0] if len(name_parts) > 0 else ""
            existing_profile["last_name"] = name_parts[1] if len(name_parts) > 1 else ""
    
    return render_template("profile.html", 
                         profile=existing_profile, 
                         title="User Profile")

@profile_bp.route("/view_profile")
@login_required
def view_profile():
    """View user profile"""
    username = session["user"]
    profile = user_profiles.find_one({"username": username})
    
    if not profile:
        flash("Please complete your profile first.", "info")
        return redirect("/profile/profile")
    
    return render_template("view_profile.html", 
                         profile=profile, 
                         title="My Profile")

@profile_bp.route("/delete_profile", methods=["POST"])
@login_required
def delete_profile():
    """Delete user profile"""
    username = session["user"]
    
    try:
        result = user_profiles.delete_one({"username": username})
        if result.deleted_count > 0:
            flash("Profile deleted successfully.", "success")
        else:
            flash("No profile found to delete.", "info")
    except Exception as e:
        flash(f"Error deleting profile: {e}", "error")
    
    return redirect(url_for("profile.profile"))

@profile_bp.route("/reset_session")
def reset_session():
    """Reset session for testing different users"""
    session.clear()
    flash("Session reset. You can start fresh.", "info")
    return redirect("/profile/profile")

@profile_bp.route("/test_data")
@login_required
def test_data():
    """Add test data for demonstration"""
    username = session["user"]
    
    test_profile = {
        "username": username,
        "first_name": "John",
        "last_name": "Doe",
        "full_name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "(555) 123-4567",
        "address": "123 Main Street, Toronto, ON M5V 3A8",
        "date_of_birth": "1990-05-15",
        "marital_status": "married",
        "number_of_dependents": 2,
        "profile_created_at": datetime.utcnow(),
        "last_updated": datetime.utcnow()
    }
    
    
    try:
        user_profiles.update_one(
            {"username": username},
            {"$set": test_profile},
            upsert=True
        )
        flash("Test profile data added successfully!", "success")
    except Exception as e:
        flash(f"Error adding test data: {e}", "error")
    
    return redirect(url_for("profile.view_profile"))

@profile_bp.route("/api/profile")
@login_required
def api_profile():
    """API endpoint to get profile data as JSON"""
    username = session["user"]
    profile = user_profiles.find_one({"username": username})
    
    if profile:
        # Convert ObjectId to string for JSON serialization
        profile["_id"] = str(profile["_id"])
        return jsonify(profile), 200          # use jsonify
    else:
        return jsonify({"error": "Profile not found"}), 404
# --- Blueprint Registration Function ---
def register_profile_blueprint(app):
    """Register the profile blueprint with the main Flask app"""
    app.register_blueprint(profile_bp, url_prefix='/profile')