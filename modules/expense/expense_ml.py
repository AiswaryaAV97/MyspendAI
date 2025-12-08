"""
expense_ml.py

AI / ML / DL engine for the SmartSpendAI Expense & Budget Analyzer.

This module provides:
- SmartCategorizer: keyword + zero-shot transformer (deep learning) with
  user-learning and confidence scores.
- AnomalyDetector: detects unusual spending based on z-score and IsolationForest.
- RecurringExpenseDetector: finds recurring expenses with ±5 day tolerance.
- SpendingForecaster: weighted monthly forecast with trend + confidence.
- SmartRecommendations: budget-based tips and savings suggestions.
- calculate_budget_status: recommended vs actual vs remaining by category.
- run_full_pipeline: single call that generates all insights for the UI.

Expected DataFrame schema for run_full_pipeline:
    - "date": pandas-compatible date/datetime column (string is fine).
    - "Category": string category per transaction.
    - "monthly": numeric value representing the *monthly equivalent* amount
      for each transaction (e.g., subscription normalized to monthly).

All public outputs are JSON-serializable (plain Python floats/ints/strings).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Optional deep learning / transformers
try:
    from transformers import pipeline

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Logging config (can be customized by the app)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Default: quiet logger, leave level/handlers to the main app if desired
    handler = logging.NullHandler()
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

CATEGORIES: List[str] = [
    "Groceries",
    "Dining",
    "Transportation",
    "Housing",
    "Utilities",
    "Shopping",
    "Health",
    "Entertainment",
    "Insurance",
    "Education",
    "Others",
]

# Recommended budget percentages (of monthly income)
RECOMMENDED_BUDGETS: Dict[str, float] = {
    "Housing": 30,
    "Transportation": 15,
    "Groceries": 12,
    "Dining": 8,
    "Utilities": 8,
    "Health": 7,
    "Insurance": 10,
    "Shopping": 5,
    "Entertainment": 5,
    "Education": 5,
    "Others": 5,
}

# ---------------------------------------------------------------------------
# Smart Categorizer (DL + rules + learning)
# ---------------------------------------------------------------------------

class SmartCategorizer:
    """
    Enhanced categorizer with:
    - Keyword rules
    - Optional zero-shot transformer (deep learning) if transformers is installed
    - Learning from user corrections
    - Confidence score per prediction
    """

    KEYWORDS: Dict[str, List[str]] = {
        "Groceries": [
            "grocery",
            "groceries",
            "supermarket",
            "freshco",
            "sobeys",
            "walmart",
            "no frills",
            "foodland",
            "costco",
            "metro",
            "loblaws",
            "real canadian",
            "food basics",
            "farm boy",
        ],
        "Dining": [
            "restaurant",
            "cafe",
            "coffee",
            "starbucks",
            "tims",
            "timmies",
            "burger",
            "mcdonald",
            "pizza",
            "kfc",
            "popeyes",
            "wendy",
            "subway",
            "chipotle",
            "harveys",
            "swiss chalet",
            "ubereats",
            "uber eats",
            "doordash",
            "skip",
            "skip the",
        ],
        "Transportation": [
            "uber",
            "lyft",
            "taxi",
            "gas",
            "fuel",
            "petro",
            "shell",
            "esso",
            "parking",
            "transit",
            "ttc",
            "bus",
            "train",
            "flight",
            "air canada",
            "westjet",
            "auto",
            "car wash",
        ],
        "Housing": [
            "rent",
            "mortgage",
            "property",
            "landlord",
            "condo",
            "apartment",
            "maintenance",
            "property tax",
            "hoa",
        ],
        "Utilities": [
            "electric",
            "electricity",
            "hydro",
            "internet",
            "wifi",
            "phone",
            "mobile",
            "cell",
            "fido",
            "rogers",
            "bell",
            "shaw",
            "telus",
            "koodo",
            "virgin",
            "water",
            "gas bill",
        ],
        "Shopping": [
            "amazon",
            "shopping",
            "retail",
            "store",
            "clothing",
            "footlocker",
            "winners",
            "best buy",
            "canadian tire",
            "home depot",
            "ikea",
            "wayfair",
            "ebay",
            "etsy",
        ],
        "Health": [
            "pharmacy",
            "shoppers drug",
            "rexall",
            "doctor",
            "hospital",
            "clinic",
            "medical",
            "gym",
            "fitness",
            "goodlife",
            "dentist",
            "dental",
            "optical",
            "vision",
            "massage",
        ],
        "Entertainment": [
            "netflix",
            "spotify",
            "apple music",
            "youtube",
            "disney",
            "hbo",
            "prime video",
            "ticket",
            "concert",
            "cineplex",
            "gaming",
            "playstation",
            "xbox",
            "nintendo",
            "steam",
        ],
        "Insurance": [
            "insurance",
            "premium",
            "manulife",
            "sunlife",
            "intact",
            "desjardins",
            "td insurance",
            "rbc insurance",
            "co-operators",
        ],
        "Education": [
            "school",
            "college",
            "university",
            "tuition",
            "course",
            "udemy",
            "coursera",
            "lynda",
            "books",
            "textbook",
        ],
    }

    def __init__(self) -> None:
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.user_corrections: Dict[str, str] = {}
        self.classifier = self._init_transformer_classifier()

    @staticmethod
    def _init_transformer_classifier():
        """Initialize zero-shot classifier if available; otherwise return None."""
        if not HAS_TRANSFORMERS:
            logger.info("Transformers not available; SmartCategorizer will use keywords only.")
            return None

        try:
            device_idx = 0 if HAS_TORCH and torch.cuda.is_available() else -1  # type: ignore
            clf = pipeline(
                "zero-shot-classification",
                model="typeform/distilbert-base-uncased-mnli",
                device=device_idx,
            )
            logger.info("SmartCategorizer: transformer model loaded successfully.")
            return clf
        except Exception as exc:
            logger.warning("Failed to load transformer pipeline, falling back to keywords. Error: %s", exc)
            return None

    @staticmethod
    def _normalize(text: Optional[str]) -> str:
        if not text:
            return ""
        return " ".join(text.lower().strip().split())

    def _keyword_match(self, text: str) -> Dict[str, Any]:
        """Keyword-based categorization with a simple confidence heuristic."""
        if not text:
            return {
                "category": "Others",
                "confidence": 0.3,
                "method": "default",
            }

        best_match: Optional[str] = None
        best_score = 0

        for cat, keywords in self.KEYWORDS.items():
            matches = sum(1 for k in keywords if k in text)
            if matches > best_score:
                best_score = matches
                best_match = cat

        if best_match:
            # Multiple keyword hits → higher confidence
            confidence = min(0.95, 0.75 + best_score * 0.05)
            return {"category": best_match, "confidence": confidence, "method": "keyword"}

        return {"category": "Others", "confidence": 0.3, "method": "default"}

    def learn_from_correction(self, text: str, correct_category: str) -> None:
        """
        Store a user correction for future use.

        Example:
            categorizer.learn_from_correction("walmart supercenter", "Groceries")
        """
        norm = self._normalize(text)
        correct_category = correct_category.strip() or "Others"

        self.user_corrections[norm] = correct_category
        self.cache[norm] = {
            "category": correct_category,
            "confidence": 1.0,
            "method": "user_corrected",
        }

    def categorize(self, text: str, amount: float = 0.0) -> Dict[str, Any]:
        """
        Categorize a transaction description.

        :param text: Raw description, e.g., "UBER *TRIP"
        :param amount: Amount of the transaction (currently not used in logic,
                       but available for future extensions).
        """
        norm = self._normalize(text)

        if not norm:
            return {"category": "Others", "confidence": 0.3, "method": "default"}

        # 1) User corrections override everything
        if norm in self.user_corrections:
            return {
                "category": self.user_corrections[norm],
                "confidence": 1.0,
                "method": "learned",
            }

        # 2) Cache
        if norm in self.cache:
            return self.cache[norm]

        # 3) Deep learning model (zero-shot) if available
        if self.classifier is not None:
            try:
                result = self.classifier(norm, CATEGORIES)
                label = str(result["labels"][0])
                score = float(result["scores"][0])

                # High confidence → accept directly
                if score >= 0.6:
                    output = {"category": label, "confidence": score, "method": "ai"}
                    self.cache[norm] = output
                    return output

                # Medium / low → compare with keyword result
                keyword_result = self._keyword_match(norm)
                if keyword_result["confidence"] > score:
                    self.cache[norm] = keyword_result
                    return keyword_result

                output = {
                    "category": label,
                    "confidence": max(score, 0.5),
                    "method": "ai+keyword",
                }
                self.cache[norm] = output
                return output

            except Exception as exc:
                logger.warning("Transformer classification failed (%s). Falling back to keywords.", exc)

        # 4) Fallback → Keyword rules only
        output = self._keyword_match(norm)
        self.cache[norm] = output
        return output

# ---------------------------------------------------------------------------
# Anomaly detector
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    is_anomaly: bool
    severity: str
    message: str
    z_score: float

class AnomalyDetector:
    """Detect unusual expenses based on historical spending."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42) -> None:
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def detect(self, history: List[float], new: float, category: str) -> AnomalyResult:
        """
        Detect if a new amount is unusual compared to the history for that category.
        """
        if len(history) < 3:
            return AnomalyResult(
                is_anomaly=False,
                severity="low",
                message="Need more data (minimum 3 past data points).",
                z_score=0.0,
            )

        history_arr = np.array(history, dtype=float)
        mean_val = float(np.mean(history_arr))
        std_val = float(np.std(history_arr))

        if std_val == 0:
            return AnomalyResult(
                is_anomaly=False,
                severity="low",
                message="Consistent spending pattern.",
                z_score=0.0,
            )

        z_score = abs((new - mean_val) / std_val)

        # IsolationForest on history + new
        arr = np.array(history + [new], dtype=float).reshape(-1, 1)
        self.model.fit(arr[:-1])
        pred = int(self.model.predict([[new]])[0])  # -1 → anomaly

        is_anomaly = pred == -1 or z_score > 2.5

        if z_score > 3:
            severity = "high"
            message = f"⚠️ Unusually high! {z_score:.1f}x above average."
        elif z_score > 2:
            severity = "medium"
            pct = (new - mean_val) / mean_val * 100 if mean_val > 0 else 0.0
            message = f"📊 Above normal by {pct:.0f}%."
        else:
            severity = "low"
            message = "✓ Normal spending pattern."

        return AnomalyResult(
            is_anomaly=is_anomaly,
            severity=severity,
            message=message,
            z_score=round(float(z_score), 2),
        )

# ---------------------------------------------------------------------------
# Recurring expenses detector
# ---------------------------------------------------------------------------

class RecurringExpenseDetector:
    """
    Identify recurring expenses per category, with ±5 day tolerance between cycles.
    """

    @staticmethod
    def detect_recurring(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if df.empty or len(df) < 2:
            return []

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        recurring: List[Dict[str, Any]] = []
        amount_tolerance = 0.25  # ±25%

        for cat in df["Category"].unique():
            cat_df = df[df["Category"] == cat]

            if len(cat_df) < 2:
                continue

            history_vals = cat_df["monthly"].astype(float).values
            base_amount = float(np.median(history_vals))

            # group by similar amounts (within tolerance)
            similar = cat_df[
                (cat_df["monthly"].astype(float) >= base_amount * (1 - amount_tolerance))
                & (cat_df["monthly"].astype(float) <= base_amount * (1 + amount_tolerance))
            ]

            if len(similar) < 2:
                continue

            dates = similar["date"].sort_values().values
            diffs = np.diff(dates).astype("timedelta64[D]").astype(int)

            if len(diffs) == 0:
                continue

            avg_gap = float(np.mean(diffs))
            std_gap = float(np.std(diffs))

            freq: Optional[str] = None
            next_date: Optional[np.datetime64] = None

            # Monthly
            if 25 <= avg_gap <= 35 and std_gap < 7:
                freq = "Monthly"
                next_date = dates[-1] + np.timedelta64(int(round(avg_gap)), "D")
            # Bi-weekly
            elif 12 <= avg_gap <= 17 and std_gap < 4:
                freq = "Bi-Weekly"
                next_date = dates[-1] + np.timedelta64(int(round(avg_gap)), "D")
            # Weekly
            elif 6 <= avg_gap <= 9 and std_gap < 2:
                freq = "Weekly"
                next_date = dates[-1] + np.timedelta64(int(round(avg_gap)), "D")
            # Annual
            elif 350 <= avg_gap <= 375:
                freq = "Annual"
                next_date = dates[-1] + np.timedelta64(365, "D")

            if freq is None or next_date is None:
                continue

            recurring.append(
                {
                    "category": str(cat),
                    "amount": float(round(base_amount, 2)),
                    "frequency": freq,
                    "occurrences": int(len(similar)),
                    "interval_days": int(round(avg_gap)),
                    "next_payment": str(next_date)[:10],
                    "reliability": "high" if std_gap < 3 else "medium",
                }
            )

        return recurring

# ---------------------------------------------------------------------------
# Spending forecaster
# ---------------------------------------------------------------------------

class SpendingForecaster:
    """
    Simple time-series style forecaster using monthly aggregates and weighted averages.
    """

    @staticmethod
    def forecast(df: pd.DataFrame, income: float) -> Dict[str, Any]:
        if df.empty:
            return {
                "forecast": 0.0,
                "confidence": 0.0,
                "trend": "unknown",
            }

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        monthly = df.groupby(df["date"].dt.to_period("M"))["monthly"].sum()

        if len(monthly) == 0:
            return {
                "forecast": 0.0,
                "confidence": 0.0,
                "trend": "unknown",
            }

        # Not enough history: simple mean
        if len(monthly) < 3:
            simple_mean = float(monthly.mean())
            return {
                "forecast": round(simple_mean, 2),
                "confidence": 0.4,
                "trend": "insufficient_data",
                "message": "Add 3+ months for more accurate forecast.",
            }

        values = monthly.astype(float).values

        # Weighted average of last 3 months (recent months weigh more)
        weights = np.exp(np.linspace(-1, 0, len(values)))
        forecast_val = float(np.average(values[-3:], weights=weights[-3:]))

        # Confidence based on coefficient of variation
        std = float(np.std(values))
        mean_val = float(np.mean(values))
        cv = std / mean_val if mean_val > 0 else 1.0
        confidence = max(0.5, min(0.95, 1.0 - cv))

        # Trend detection
        if len(values) >= 4:
            recent_avg = float(np.mean(values[-3:]))
            older_avg = float(np.mean(values[:-3]))
        else:
            recent_avg = float(np.mean(values[-2:]))
            older_avg = float(values[0])

        if older_avg > 0:
            change_pct = (recent_avg - older_avg) / older_avg * 100.0
        else:
            change_pct = 0.0

        if change_pct > 10:
            trend = "increasing"
        elif change_pct < -10:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "forecast": round(forecast_val, 2),
            "confidence": round(confidence, 2),
            "trend": trend,
            "min_estimate": round(forecast_val * 0.9, 2),
            "max_estimate": round(forecast_val * 1.1, 2),
            "change_pct": round(change_pct, 1),
        }

# ---------------------------------------------------------------------------
# Smart recommendations & budget status
# ---------------------------------------------------------------------------

class SmartRecommendations:
    """Generate human-readable spending recommendations based on income & budgets."""

    @staticmethod
    def generate(df: pd.DataFrame, income: float) -> List[Dict[str, Any]]:
        if df.empty or income <= 0:
            return []

        df = df.copy()
        df["monthly"] = df["monthly"].astype(float)
        cat_sum = df.groupby("Category")["monthly"].sum()

        recommendations: List[Dict[str, Any]] = []

        # Category-level checks against recommended budgets
        for cat, amount in cat_sum.items():
            if cat not in RECOMMENDED_BUDGETS:
                continue

            amount = float(amount)
            recommended_pct = float(RECOMMENDED_BUDGETS[cat])
            recommended_amount = income * (recommended_pct / 100.0)
            actual_pct = (amount / income * 100.0) if income > 0 else 0.0

            if actual_pct > recommended_pct * 1.2:  # > 120% of recommended
                overage = amount - recommended_amount
                recommendations.append(
                    {
                        "category": str(cat),
                        "type": "warning",
                        "message": (
                            f"💰 {cat} is {actual_pct:.1f}% of income "
                            f"(recommended: {recommended_pct:.0f}%). "
                            f"Try reducing by ${overage:.0f}/month."
                        ),
                        "priority": "high",
                    }
                )

        # Specific category behavioural tips
        if "Dining" in cat_sum and float(cat_sum["Dining"]) > income * 0.10:
            recommendations.append(
                {
                    "category": "Dining",
                    "type": "tip",
                    "message": (
                        "🍳 Dining is high. Try cooking 2–3 meals at home per week "
                        "to save around $200–300/month."
                    ),
                    "priority": "medium",
                }
            )

        if "Shopping" in cat_sum and float(cat_sum["Shopping"]) > income * 0.08:
            recommendations.append(
                {
                    "category": "Shopping",
                    "type": "tip",
                    "message": "🛒 Shopping is high. Use the 24-hour rule before non-essential purchases.",
                    "priority": "medium",
                }
            )

        # Overall savings rate
        total_spending = float(cat_sum.sum())
        savings = income - total_spending
        savings_rate = (savings / income * 100.0) if income > 0 else 0.0

        if savings_rate < 10:
            recommendations.append(
                {
                    "category": "Savings",
                    "type": "alert",
                    "message": (
                        f"⚠️ Savings rate is {savings_rate:.1f}%. "
                        "Aim for 10–20% to build your financial safety net."
                    ),
                    "priority": "high",
                }
            )
        elif savings_rate >= 20:
            recommendations.append(
                {
                    "category": "Savings",
                    "type": "success",
                    "message": (
                        f"🎉 Great job! You're saving {savings_rate:.1f}% of income. "
                        "Consider investing part of your savings for long-term growth."
                    ),
                    "priority": "low",
                }
            )

        # Sort by priority: high → medium → low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations = sorted(
            recommendations,
            key=lambda x: priority_order.get(x.get("priority", "medium"), 1),
        )

        # limit to top 8 for UI
        return recommendations[:8]

def calculate_budget_status(df: pd.DataFrame, income: float) -> List[Dict[str, Any]]:
    """
    Calculate recommended vs actual vs remaining for each category.

    Returns list of dicts:
        {
            "category": ...,
            "recommended": float,
            "actual": float,
            "remaining": float,
            "used_percentage": float,
            "status": "good" | "warning" | "over"
        }
    """
    if df.empty or income <= 0:
        return []

    df = df.copy()
    df["monthly"] = df["monthly"].astype(float)
    cat_sum = df.groupby("Category")["monthly"].sum()

    budget_status: List[Dict[str, Any]] = []

    for cat in CATEGORIES:
        recommended_pct = float(RECOMMENDED_BUDGETS.get(cat, 5.0))
        recommended_amount = income * (recommended_pct / 100.0)
        actual_amount = float(cat_sum.get(cat, 0.0))

        if actual_amount > 0:
            used_pct = (actual_amount / recommended_amount * 100.0) if recommended_amount > 0 else 0.0
            used_pct = min(100.0, used_pct)
            remaining = max(0.0, recommended_amount - actual_amount)

            if used_pct >= 100.0:
                status = "over"
            elif used_pct >= 80.0:
                status = "warning"
            else:
                status = "good"
        else:
            used_pct = 0.0
            remaining = recommended_amount
            status = "good"

        budget_status.append(
            {
                "category": cat,
                "recommended": round(recommended_amount, 2),
                "actual": round(actual_amount, 2),
                "remaining": round(remaining, 2),
                "used_percentage": round(used_pct, 1),
                "status": status,
            }
        )

    return budget_status

# ---------------------------------------------------------------------------
# Main pipeline for UI
# ---------------------------------------------------------------------------

def run_full_pipeline(df: pd.DataFrame, income: float) -> Dict[str, Any]:
    """
    Main pipeline used by the Flask UI.

    :param df: DataFrame with at least columns ["date", "Category", "monthly"].
    :param income: Monthly income (float).

    Returns a JSON-safe dictionary summarizing:
      - summary text
      - total_spent, savings, savings_rate
      - forecast dict
      - recurring_expenses list
      - recommendations list
      - category_analysis list
      - budget_status list
    """
    df = df.copy()

    if "date" not in df.columns:
        df["date"] = datetime.utcnow().date()

    if "Category" not in df.columns:
        df["Category"] = "Others"

    if "monthly" not in df.columns:
        raise ValueError("DataFrame must have a 'monthly' column for pipeline analysis.")

    df["Category"] = df["Category"].astype(str)
    df["monthly"] = df["monthly"].astype(float)

    total_spent = float(df["monthly"].sum())
    savings = float(income - total_spent)
    savings_rate = float((savings / income * 100.0) if income > 0 else 0.0)

    # AI modules
    recurring = RecurringExpenseDetector.detect_recurring(df)
    forecast = SpendingForecaster.forecast(df, income)
    recommendations = SmartRecommendations.generate(df, income)
    budget_status = calculate_budget_status(df, income)

    # Category analysis for charts & tables
    total_sum = float(df["monthly"].sum())
    category_analysis: List[Dict[str, Any]] = []

    for cat, group in df.groupby("Category"):
        cat_total = float(group["monthly"].sum())
        tx_count = int(len(group))
        avg_tx = float(cat_total / tx_count) if tx_count > 0 else 0.0
        pct = float(cat_total / total_sum * 100.0) if total_sum > 0 else 0.0

        category_analysis.append(
            {
                "category": str(cat),
                "total": round(cat_total, 2),
                "percentage": round(pct, 1),
                "transaction_count": tx_count,
                "avg_transaction": round(avg_tx, 2),
            }
        )

    category_analysis = sorted(category_analysis, key=lambda x: x["total"], reverse=True)

    summary_text = (
        f"Monthly spending: ${total_spent:.0f} from ${income:.0f} income. "
        f"Savings: ${savings:.0f} ({savings_rate:.1f}%)."
    )

    return {
        "summary": summary_text,
        "total_spent": total_spent,
        "savings": savings,
        "savings_rate": savings_rate,
        "forecast": forecast,
        "recurring_expenses": recurring,
        "recommendations": recommendations,
        "category_analysis": category_analysis,
        "budget_status": budget_status,
    }

# ---------------------------------------------------------------------------
# Global instances – convenient imports for the rest of the app
# ---------------------------------------------------------------------------

categorizer = SmartCategorizer()
anomaly_detector = AnomalyDetector()
recurring_detector = RecurringExpenseDetector()
forecaster = SpendingForecaster()
recommender = SmartRecommendations()

__all__ = [
    "SmartCategorizer",
    "AnomalyDetector",
    "RecurringExpenseDetector",
    "SpendingForecaster",
    "SmartRecommendations",
    "calculate_budget_status",
    "run_full_pipeline",
    "categorizer",
    "anomaly_detector",
    "recurring_detector",
    "forecaster",
    "recommender",
]
