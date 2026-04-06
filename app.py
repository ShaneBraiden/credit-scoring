"""
Credit Risk Assessment - Flask Backend

Loads the trained ML model (Logistic Regression, Random Forest, or XGBoost)
and exposes endpoints for credit risk predictions.

Dataset: Give Me Some Credit (Kaggle)
Features: 18 engineered features from credit history data
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ─── Load Model at Startup ───────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "credit_model.pkl"
)

model = None
scaler = None
feature_cols = None
model_info = {}

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = artifact["feature_cols"]
    model_info = {
        "name": artifact.get("model_name", "Unknown"),
        "roc_auc": artifact.get("metrics", {}).get("roc_auc", 0),
        "cv_auc_mean": artifact.get("cv_auc_mean", 0),
        "num_features": len(feature_cols),
        "demo_mode": artifact.get("demo_mode", False),
    }
    print(f"✓ Model loaded: {model_info['name']}")
    print(f"  ROC-AUC: {model_info['roc_auc']:.4f}")
    print(f"  Features: {model_info['num_features']}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Run 'python train_model.py' first to generate the model.")
except Exception as e:
    print(f"ERROR loading model: {e}")


# ─── Helper Functions ────────────────────────────────────────────────────────

def engineer_input_features(data: dict) -> dict:
    """
    Engineer additional features from raw input to match training data.
    
    Raw inputs from form:
        - RevolvingUtilizationOfUnsecuredLines
        - age
        - NumberOfTime30-59DaysPastDueNotWorse
        - DebtRatio
        - MonthlyIncome
        - NumberOfOpenCreditLinesAndLoans
        - NumberOfTimes90DaysLate
        - NumberRealEstateLoansOrLines
        - NumberOfTime60-89DaysPastDueNotWorse
        - NumberOfDependents
    
    Engineered features:
        - TotalTimesPastDue
        - HasDelinquency
        - HighUtilization
        - AgeGroup
        - IncomeToDebt
        - HasRealEstate
        - TotalOpenAccounts
        - IncomePerDependent
    """
    # Total times past due
    data["TotalTimesPastDue"] = (
        data.get("NumberOfTime30-59DaysPastDueNotWorse", 0) +
        data.get("NumberOfTime60-89DaysPastDueNotWorse", 0) +
        data.get("NumberOfTimes90DaysLate", 0)
    )
    
    # Has any delinquency flag
    data["HasDelinquency"] = 1 if data["TotalTimesPastDue"] > 0 else 0
    
    # High credit utilization flag
    data["HighUtilization"] = 1 if data.get("RevolvingUtilizationOfUnsecuredLines", 0) > 0.8 else 0
    
    # Age groups (0-5)
    age = data.get("age", 35)
    if age <= 25:
        data["AgeGroup"] = 0
    elif age <= 35:
        data["AgeGroup"] = 1
    elif age <= 45:
        data["AgeGroup"] = 2
    elif age <= 55:
        data["AgeGroup"] = 3
    elif age <= 65:
        data["AgeGroup"] = 4
    else:
        data["AgeGroup"] = 5
    
    # Income to debt ratio
    debt_ratio = data.get("DebtRatio", 0.5)
    monthly_income = data.get("MonthlyIncome", 5000)
    if debt_ratio > 0:
        data["IncomeToDebt"] = min(1 / debt_ratio, 100)
    else:
        data["IncomeToDebt"] = min(monthly_income, 100)
    
    # Has real estate flag
    data["HasRealEstate"] = 1 if data.get("NumberRealEstateLoansOrLines", 0) > 0 else 0
    
    # Total open accounts
    data["TotalOpenAccounts"] = (
        data.get("NumberOfOpenCreditLinesAndLoans", 0) +
        data.get("NumberRealEstateLoansOrLines", 0)
    )
    
    # Income per dependent
    dependents = data.get("NumberOfDependents", 0)
    if dependents > 0:
        data["IncomePerDependent"] = monthly_income / (dependents + 1)
    else:
        data["IncomePerDependent"] = monthly_income
    
    return data


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend UI."""
    return render_template("index.html")


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Return information about the loaded model."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_name": model_info.get("name", "Unknown"),
        "roc_auc": round(model_info.get("roc_auc", 0), 4),
        "cv_auc_mean": round(model_info.get("cv_auc_mean", 0), 4),
        "num_features": model_info.get("num_features", 0),
        "demo_mode": model_info.get("demo_mode", False),
        "feature_columns": feature_cols,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON with credit features, return risk prediction.

    Expected JSON keys (Give Me Some Credit dataset features):
        - RevolvingUtilizationOfUnsecuredLines
        - age
        - NumberOfTime30-59DaysPastDueNotWorse
        - DebtRatio
        - MonthlyIncome
        - NumberOfOpenCreditLinesAndLoans
        - NumberOfTimes90DaysLate
        - NumberRealEstateLoansOrLines
        - NumberOfTime60-89DaysPastDueNotWorse
        - NumberOfDependents

    Response:
        {
            "decision": "Approved" | "Rejected",
            "risk_percentage": 12.5,
            "risk_level": "Low Risk" | "High Risk",
            "model_name": "XGBoost",
            "confidence": "high" | "medium" | "low"
        }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run 'python train_model.py' first."}), 500

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided."}), 400

    # Required raw features from the form
    raw_features = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]
    
    # Check for missing raw features
    missing = [f for f in raw_features if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        # Convert to floats
        for key in raw_features:
            data[key] = float(data[key])
        
        # Engineer additional features
        data = engineer_input_features(data)
        
        # Extract features in the exact order the model expects
        features = [float(data[col]) for col in feature_cols]
        
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data type for input fields: {e}"}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing engineered feature: {e}"}), 400

    # Scale and predict
    features_scaled = scaler.transform(np.array([features]))
    prediction = model.predict(features_scaled)[0]
    risk_prob = model.predict_proba(features_scaled)[0][1]  # Probability of default

    risk_percentage = round(risk_prob * 100, 1)
    
    # Determine confidence level based on probability distance from threshold
    prob_distance = abs(risk_prob - 0.5)
    if prob_distance > 0.3:
        confidence = "high"
    elif prob_distance > 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    result = {
        "decision": "Rejected" if prediction == 1 else "Approved",
        "risk_percentage": risk_percentage,
        "risk_level": "High Risk" if prediction == 1 else "Low Risk",
        "model_name": model_info.get("name", "Unknown"),
        "confidence": confidence,
    }

    return jsonify(result)


# ─── Run Server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
