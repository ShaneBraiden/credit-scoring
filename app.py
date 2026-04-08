"""
Indian Credit Risk Assessment - Flask Backend

Loads the trained ML model and exposes endpoints for loan approval predictions.
Supports comprehensive features for Indian credit assessment.

Features: CIBIL Score, Income (INR), Employment, Banking Behavior, Loan Details
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
label_encoders = {}
model_info = {}

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = artifact["feature_cols"]
    label_encoders = artifact.get("label_encoders", {})
    model_info = {
        "name": artifact.get("model_name", "Unknown"),
        "roc_auc": artifact.get("metrics", {}).get("roc_auc", 0),
        "accuracy": artifact.get("metrics", {}).get("accuracy", 0),
        "cv_auc_mean": artifact.get("cv_auc_mean", 0),
        "num_features": len(feature_cols),
        "demo_mode": artifact.get("demo_mode", False),
    }
    print(f"Model loaded: {model_info['name']}")
    print(f"  ROC-AUC: {model_info['roc_auc']:.4f}")
    print(f"  Accuracy: {model_info['accuracy']:.4f}")
    print(f"  Features: {model_info['num_features']}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Run 'python combine_datasets.py' then 'python train_model.py' first.")
except Exception as e:
    print(f"ERROR loading model: {e}")


# ─── Indian States List ──────────────────────────────────────────────────────
INDIAN_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
    'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
    'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    'Delhi', 'Chandigarh'
]

# Education levels
EDUCATION_LEVELS = ['Below 10th', '10th Pass', '12th Pass', 'Graduate', 'Post Graduate', 'Professional Degree']

# Employment types
EMPLOYMENT_TYPES = ['Salaried', 'Self-Employed', 'Business Owner', 'Freelancer', 'Government Employee', 'Retired']

# Loan purposes
LOAN_PURPOSES = ['Home Loan', 'Personal Loan', 'Education Loan', 'Vehicle Loan', 'Business Loan', 'Medical Emergency', 'Wedding', 'Debt Consolidation']


def engineer_input_features(data: dict) -> dict:
    """
    Engineer additional features from raw input to match training data.
    """
    # 1. Income to loan ratio
    annual_income = data.get('annual_income', 500000)
    loan_amount = data.get('loan_amount', 100000)
    data['income_to_loan_ratio'] = min(annual_income / (loan_amount + 1), 100)
    
    # 2. CIBIL score buckets
    cibil = data.get('cibil_score', 700)
    if cibil <= 550:
        data['cibil_bucket'] = 0
    elif cibil <= 650:
        data['cibil_bucket'] = 1
    elif cibil <= 750:
        data['cibil_bucket'] = 2
    else:
        data['cibil_bucket'] = 3
    
    # 3. Good CIBIL flag
    data['good_cibil'] = 1 if cibil >= 700 else 0
    
    # 4. Debt to income ratio
    existing_debt = data.get('existing_debt', 0)
    data['debt_to_income'] = min(existing_debt / (annual_income + 1), 10)
    
    # 5. High debt flag
    data['high_debt'] = 1 if data['debt_to_income'] > 0.5 else 0
    
    # 6. EMI to income ratio
    loan_tenure = data.get('loan_tenure', 60)
    monthly_income = data.get('monthly_income', annual_income / 12)
    estimated_emi = loan_amount / loan_tenure
    data['emi_to_income'] = min(estimated_emi / (monthly_income + 1), 1)
    
    # 7. Employment stability
    emp_years = data.get('employment_years', 2)
    data['stable_employment'] = 1 if emp_years >= 2 else 0
    data['experienced'] = 1 if emp_years >= 5 else 0
    
    # 8. Credit behavior
    num_past_loans = data.get('num_past_loans', 0)
    loans_repaid = data.get('loans_repaid', 0)
    data['repayment_ratio'] = loans_repaid / (num_past_loans + 1)
    data['has_missed_payments'] = 1 if data.get('missed_payments', 0) > 0 else 0
    
    # 9. Banking health
    avg_balance = data.get('avg_bank_balance', monthly_income * 0.5)
    data['savings_ratio'] = min(avg_balance / (monthly_income + 1), 10)
    
    # 10. Age groups
    age = data.get('age', 30)
    if age <= 25:
        data['age_group'] = 0
    elif age <= 35:
        data['age_group'] = 1
    elif age <= 45:
        data['age_group'] = 2
    elif age <= 55:
        data['age_group'] = 3
    else:
        data['age_group'] = 4
    
    # 11. High credit utilization
    credit_util = data.get('credit_utilization', 30)
    data['high_utilization'] = 1 if credit_util > 70 else 0
    
    return data


def encode_categorical(value: str, category: str) -> int:
    """Encode categorical values using saved label encoders or defaults."""
    if category in label_encoders:
        try:
            return int(label_encoders[category].transform([value])[0])
        except (ValueError, KeyError):
            # Value not seen during training, return middle value
            return len(label_encoders[category].classes_) // 2
    
    # Fallback encoding
    defaults = {
        'education': {'Below 10th': 0, '10th Pass': 1, '12th Pass': 2, 'Graduate': 3, 'Post Graduate': 4, 'Professional Degree': 5},
        'employment_type': {'Salaried': 0, 'Self-Employed': 1, 'Business Owner': 2, 'Freelancer': 3, 'Government Employee': 4, 'Retired': 5},
        'state': {s: i for i, s in enumerate(INDIAN_STATES)},
        'loan_purpose': {p: i for i, p in enumerate(LOAN_PURPOSES)},
    }
    return defaults.get(category, {}).get(value, 0)


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
        "accuracy": round(model_info.get("accuracy", 0), 4),
        "cv_auc_mean": round(model_info.get("cv_auc_mean", 0), 4),
        "num_features": model_info.get("num_features", 0),
        "demo_mode": model_info.get("demo_mode", False),
        "feature_columns": feature_cols,
    })


@app.route("/api/options", methods=["GET"])
def get_options():
    """Return dropdown options for the frontend."""
    return jsonify({
        "states": INDIAN_STATES,
        "education_levels": EDUCATION_LEVELS,
        "employment_types": EMPLOYMENT_TYPES,
        "loan_purposes": LOAN_PURPOSES,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON with loan application data, return approval prediction.

    Expected JSON keys (all amounts in INR):
        Personal & Financial:
            - age: Applicant age (21-65)
            - annual_income: Annual income in INR
            - employment_years: Years of employment
            - education: Education level
            - employment_type: Type of employment
            - state: Indian state
            - dependents: Number of dependents
        
        Credit History:
            - cibil_score: CIBIL score (300-900)
            - num_past_loans: Number of past loans
            - loans_repaid: Loans repaid successfully
            - missed_payments: Missed payments in last 12 months
            - credit_utilization: Credit card usage %
            - existing_debt: Existing debt in INR
        
        Banking Behavior:
            - avg_bank_balance: Average monthly balance in INR
            - has_savings: Has savings account (0/1)
        
        Loan Details:
            - loan_amount: Requested loan amount in INR
            - loan_purpose: Purpose of loan
            - loan_tenure: Loan duration in months

    Response:
        {
            "decision": "Approved" | "Rejected",
            "approval_probability": 75.5,
            "risk_level": "Low Risk" | "Medium Risk" | "High Risk",
            "cibil_category": "Excellent" | "Good" | "Fair" | "Poor",
            "model_name": "XGBoost",
            "confidence": "high" | "medium" | "low",
            "monthly_emi_estimate": 15000
        }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run training first."}), 500

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided."}), 400

    # Required fields
    required_fields = ['age', 'annual_income', 'cibil_score', 'loan_amount']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        # Convert numeric fields
        data['age'] = int(data['age'])
        data['annual_income'] = float(data['annual_income'])
        data['monthly_income'] = data['annual_income'] / 12
        data['employment_years'] = int(data.get('employment_years', 2))
        data['dependents'] = int(data.get('dependents', 0))
        
        data['cibil_score'] = int(data['cibil_score'])
        data['num_past_loans'] = int(data.get('num_past_loans', 0))
        data['loans_repaid'] = int(data.get('loans_repaid', 0))
        data['missed_payments'] = int(data.get('missed_payments', 0))
        data['has_credit_card'] = int(data.get('has_credit_card', 1))
        data['credit_utilization'] = float(data.get('credit_utilization', 30))
        data['existing_debt'] = float(data.get('existing_debt', 0))
        data['credit_inquiries'] = int(data.get('credit_inquiries', 1))
        
        data['avg_bank_balance'] = float(data.get('avg_bank_balance', data['monthly_income'] * 0.5))
        data['monthly_transactions'] = int(data.get('monthly_transactions', 20))
        data['spending_ratio'] = float(data.get('spending_ratio', 0.6))
        data['has_savings'] = int(data.get('has_savings', 1))
        data['account_age'] = int(data.get('account_age', 3))
        
        data['loan_amount'] = float(data['loan_amount'])
        data['loan_tenure'] = int(data.get('loan_tenure', 60))
        
        # Encode categorical fields
        data['education_encoded'] = encode_categorical(data.get('education', 'Graduate'), 'education')
        data['employment_type_encoded'] = encode_categorical(data.get('employment_type', 'Salaried'), 'employment_type')
        data['state_encoded'] = encode_categorical(data.get('state', 'Maharashtra'), 'state')
        data['loan_purpose_encoded'] = encode_categorical(data.get('loan_purpose', 'Personal Loan'), 'loan_purpose')
        
        # Engineer additional features
        data = engineer_input_features(data)
        
        # Extract features in the exact order the model expects
        features = []
        for col in feature_cols:
            if col in data:
                features.append(float(data[col]))
            else:
                features.append(0.0)  # Default value
        
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data type: {e}"}), 400

    # Scale and predict
    features_scaled = scaler.transform(np.array([features]))
    prediction = model.predict(features_scaled)[0]
    approval_prob = model.predict_proba(features_scaled)[0][1]

    approval_percentage = round(approval_prob * 100, 1)
    
    # CIBIL category
    cibil = data['cibil_score']
    if cibil >= 750:
        cibil_category = "Excellent"
    elif cibil >= 700:
        cibil_category = "Good"
    elif cibil >= 650:
        cibil_category = "Fair"
    else:
        cibil_category = "Poor"
    
    # Risk level based on multiple factors
    debt_to_income = data.get('debt_to_income', 0)
    
    if cibil >= 720 and debt_to_income <= 0.3 and data.get('missed_payments', 0) == 0:
        risk_level = "Low Risk"
    elif cibil >= 650 and debt_to_income <= 0.5:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
    
    # Confidence based on probability distance from threshold
    prob_distance = abs(approval_prob - 0.5)
    if prob_distance > 0.3:
        confidence = "high"
    elif prob_distance > 0.15:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Estimate EMI (simple calculation)
    annual_rate = 0.12  # 12% assumed interest
    monthly_rate = annual_rate / 12
    tenure = data['loan_tenure']
    loan_amt = data['loan_amount']
    
    if monthly_rate > 0:
        emi = loan_amt * monthly_rate * ((1 + monthly_rate) ** tenure) / (((1 + monthly_rate) ** tenure) - 1)
    else:
        emi = loan_amt / tenure
    
    result = {
        "decision": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": approval_percentage,
        "risk_level": risk_level,
        "cibil_category": cibil_category,
        "model_name": model_info.get("name", "Unknown"),
        "confidence": confidence,
        "monthly_emi_estimate": round(emi),
    }

    return jsonify(result)


# ─── Run Server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
