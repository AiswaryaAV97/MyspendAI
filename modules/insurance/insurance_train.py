# modules/insurance/insurance_train.py

"""
Train synthetic ML models for the Insurance & Risk Advisor.

- Multi-output regression → predicts recommended coverage amounts
  (life, health, disability, property) in Canadian-like context
- Classification → predicts overall risk category (Low/Medium/High)

Outputs:
- models/insurance_coverage_model.pkl
- models/insurance_risk_model.pkl
- models/insurance_meta.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

COVERAGE_MODEL_PATH = MODEL_DIR / "insurance_coverage_model.pkl"
RISK_MODEL_PATH = MODEL_DIR / "insurance_risk_model.pkl"
META_PATH = MODEL_DIR / "insurance_meta.json"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

GENDERS = ["Male", "Female", "Prefer not to say"]
MARITAL = ["Single", "Married", "Partner", "Divorced", "Widowed"]
RISK_TOL = ["low", "medium", "high"]
PROVINCES = ["ON", "BC", "QC", "AB"]


def generate_synthetic_dataset(n_samples: int = 8000) -> pd.DataFrame:
    age = np.random.randint(20, 70, size=n_samples)
    gender = np.random.choice(GENDERS, size=n_samples)
    marital_status = np.random.choice(MARITAL, size=n_samples)
    dependents = np.random.poisson(1.0, size=n_samples)
    dependents = np.clip(dependents, 0, 4)

    income = np.random.normal(85000, 25000, size=n_samples)
    income = np.clip(income, 30000, 200000)

    assets = np.random.normal(200000, 150000, size=n_samples)
    assets = np.clip(assets, 0, 800000)

    liabilities = np.random.normal(120000, 100000, size=n_samples)
    liabilities = np.clip(liabilities, 0, 600000)

    bmi = np.random.normal(26, 4, size=n_samples)
    bmi = np.clip(bmi, 18, 40)

    bp_sys = np.random.normal(120, 15, size=n_samples)
    chol = np.random.normal(190, 35, size=n_samples)

    smoker = np.random.binomial(1, 0.18, size=n_samples)
    alcohol = np.random.binomial(1, 0.4, size=n_samples)

    exercise = np.random.choice(["low", "medium", "high"], size=n_samples, p=[0.3, 0.45, 0.25])
    risk_tolerance = np.random.choice(RISK_TOL, size=n_samples, p=[0.3, 0.5, 0.2])
    province = np.random.choice(PROVINCES, size=n_samples)

    # Base risk (0–100)
    risk = np.ones(n_samples) * 25
    risk += (age - 35) * 0.5
    risk += (bmi - 25) * 1.0
    risk += (bp_sys - 120) * 0.4
    risk += (chol - 190) * 0.2
    risk += smoker * 8
    risk += alcohol * 2
    risk += np.where(exercise == "low", 6, 0)
    risk += np.where(exercise == "high", -4, 0)
    risk += np.where(risk_tolerance == "high", +4, 0)
    risk += np.where(risk_tolerance == "low", -3, 0)
    risk += (liabilities - assets) / 40000.0 * 5.0
    risk = np.clip(risk + np.random.normal(0, 4, size=n_samples), 0, 100)

    # Rule-of-thumb baseline coverages (in CAD)
    life_rule = income * 10 + liabilities + dependents * 60000 - assets * 0.4
    life_rule = np.clip(life_rule, 50_000, 2_000_000)

    # IMPORTANT: use np.maximum, not Python max()
    health_rule = 100000 + (np.maximum(0, bmi - 25) * 2000) + smoker * 40000
    health_rule = np.clip(health_rule, 50_000, 400_000)

    disability_rule = income * 0.6 * 3  # 60% income for 3 years
    disability_rule = np.clip(disability_rule, 40_000, 400_000)

    property_rule = np.maximum(assets * 0.8, 100_000)
    property_rule = np.clip(property_rule, 100_000, 800_000)

    # "True" recommended coverage = rule + small noise, scaled by risk
    multiplier = 0.7 + (risk / 100) * 0.8  # between ~0.7 and 1.5
    life_cov = life_rule * multiplier * np.random.normal(1.0, 0.08, size=n_samples)
    health_cov = health_rule * multiplier * np.random.normal(1.0, 0.10, size=n_samples)
    disability_cov = disability_rule * multiplier * np.random.normal(1.0, 0.07, size=n_samples)
    property_cov = property_rule * (0.7 + risk / 200) * np.random.normal(1.0, 0.05, size=n_samples)

    life_cov = np.clip(life_cov, 50_000, 2_500_000)
    health_cov = np.clip(health_cov, 40_000, 500_000)
    disability_cov = np.clip(disability_cov, 30_000, 500_000)
    property_cov = np.clip(property_cov, 80_000, 1_000_000)

    # Risk category
    risk_cat = np.where(risk < 33, "Low",
                        np.where(risk < 66, "Medium", "High"))

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "dependents": dependents,
        "annual_income": income,
        "assets": assets,
        "liabilities": liabilities,
        "bmi": bmi,
        "bp_sys": bp_sys,
        "cholesterol": chol,
        "smoker": smoker,
        "alcohol": alcohol,
        "exercise": exercise,
        "risk_tolerance": risk_tolerance,
        "province": province,
        "risk_score": risk,
        "risk_category": risk_cat,
        "life_cov": life_cov,
        "health_cov": health_cov,
        "disability_cov": disability_cov,
        "property_cov": property_cov,
    })
    return df


def build_pipelines():
    numeric_features = [
        "age", "dependents", "annual_income", "assets", "liabilities",
        "bmi", "bp_sys", "cholesterol", "smoker", "alcohol"
    ]
    categorical_features = [
        "gender", "marital_status", "exercise", "risk_tolerance", "province"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    cov_reg = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=180,
            max_depth=12,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    )

    risk_clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    cov_model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", cov_reg),
    ])

    risk_model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", risk_clf),
    ])

    feature_cols = numeric_features + categorical_features
    return cov_model, risk_model, feature_cols


def main():
    print("🧪 Generating synthetic insurance dataset...")
    df = generate_synthetic_dataset(9000)

    cov_model, risk_model, feature_cols = build_pipelines()

    X = df[feature_cols]
    y_cov = df[["life_cov", "health_cov", "disability_cov", "property_cov"]]
    y_risk = df["risk_category"]

    X_train, X_val, y_cov_train, y_cov_val, y_risk_train, y_risk_val = train_test_split(
        X, y_cov, y_risk, test_size=0.2, random_state=RANDOM_SEED
    )

    print("🚀 Training coverage regression model...")
    cov_model.fit(X_train, y_cov_train)
    cov_pred = cov_model.predict(X_val)
    cov_r2 = r2_score(y_cov_val, cov_pred, multioutput="uniform_average")
    print(f"✅ Coverage model R² (avg): {cov_r2:.3f}")

    print("🚀 Training risk classifier...")
    risk_model.fit(X_train, y_risk_train)
    risk_pred = risk_model.predict(X_val)
    print("✅ Risk classifier report:")
    print(classification_report(y_risk_val, risk_pred))

    joblib.dump(cov_model, COVERAGE_MODEL_PATH)
    joblib.dump(risk_model, RISK_MODEL_PATH)
    print(f"💾 Saved coverage model → {COVERAGE_MODEL_PATH}")
    print(f"💾 Saved risk model → {RISK_MODEL_PATH}")

    meta = {
        "feature_columns": feature_cols,
        "coverage_targets": ["life_cov", "health_cov", "disability_cov", "property_cov"],
        "risk_classes": ["Low", "Medium", "High"],
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"💾 Saved metadata → {META_PATH}")
    print("🎉 Training complete. Ready to use in insurance_ml.py")


if __name__ == "__main__":
    main()
