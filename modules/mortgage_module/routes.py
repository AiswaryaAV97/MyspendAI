from flask import Blueprint, render_template, request, jsonify, current_app
from .mortgage_service import process_mortgage, process_affordability

mortgage_bp = Blueprint("mortgage", __name__, template_folder="../../templates")

@mortgage_bp.route("/planner/mortgage", methods=["GET"])
def mortgage_page():
    active_tab = request.args.get("tab", "feature")
    return render_template("mortgage.html", active_tab=active_tab)

@mortgage_bp.route("/planner/mortgage", methods=["POST"])
def mortgage_calculate():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict(flat=True)
        if not payload:
            return jsonify({"error": "Missing request payload"}), 400

        action = payload.get("action", "calculator")
        if action == "affordability":
            result = process_affordability(payload)
        else:
            result = process_mortgage(payload)

        if isinstance(result, dict) and result.get("error"):
            return jsonify(result), 400
        return jsonify(result)
    except Exception:
        current_app.logger.exception("Error in mortgage_calculate")
        return jsonify({"error": "Internal server error"}), 500

@mortgage_bp.route("/mortgage/affordability", methods=["POST"])
def mortgage_affordability_compat():
    """Accept legacy /mortgage/affordability URL and forward to main handler"""
    return mortgage_calculate()
