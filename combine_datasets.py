"""
Combine/Generate Indian Credit Risk Dataset

Creates a comprehensive Indian credit risk dataset with features covering:
- Personal & financial info
- Credit history
- Banking behavior
- Loan details

Run: python combine_datasets.py
"""

import pandas as pd
import numpy as np
import os

# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_FILE = os.path.join(DATASET_DIR, "indian_credit_data.csv")

# Sample size
SAMPLE_SIZE = 50000  # 50K records

# Indian States
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

# Loan purposes
LOAN_PURPOSES = ['Home Loan', 'Personal Loan', 'Education Loan', 'Vehicle Loan', 'Business Loan', 'Medical Emergency', 'Wedding', 'Debt Consolidation']

# Employment types
EMPLOYMENT_TYPES = ['Salaried', 'Self-Employed', 'Business Owner', 'Freelancer', 'Government Employee', 'Retired']


def generate_indian_credit_dataset(n_samples: int) -> pd.DataFrame:
    """Generate synthetic Indian credit risk dataset."""
    
    np.random.seed(42)
    
    print(f"Generating {n_samples:,} synthetic records...")
    
    # ─── 1. Personal & Financial Info ─────────────────────────────────────────
    age = np.random.randint(21, 65, n_samples)
    
    # Annual income in INR (based on age and randomness)
    base_income = 200000 + (age - 21) * 15000 + np.random.randint(-100000, 300000, n_samples)
    annual_income = np.clip(base_income, 150000, 5000000)  # 1.5L to 50L
    
    # Monthly income
    monthly_income = annual_income / 12
    
    # Employment length (years)
    max_employment = np.minimum(age - 21, 40)
    employment_years = np.array([np.random.randint(0, max(1, m)) for m in max_employment])
    
    # Education
    education = np.random.choice(EDUCATION_LEVELS, n_samples, p=[0.05, 0.10, 0.15, 0.40, 0.20, 0.10])
    
    # Employment type
    employment_type = np.random.choice(EMPLOYMENT_TYPES, n_samples, p=[0.45, 0.15, 0.15, 0.10, 0.10, 0.05])
    
    # State
    state = np.random.choice(INDIAN_STATES, n_samples)
    
    # Number of dependents
    dependents = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.15, 0.20, 0.30, 0.20, 0.10, 0.05])
    
    # ─── 2. Credit History ────────────────────────────────────────────────────
    # CIBIL Score (300-900)
    cibil_score = np.random.randint(300, 900, n_samples)
    
    # Number of past loans
    num_past_loans = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.20, 0.30, 0.25, 0.15, 0.07, 0.03])
    
    # Number of loans repaid successfully
    loans_repaid = np.array([np.random.randint(0, max(1, l+1)) for l in num_past_loans])
    
    # Number of missed payments (EMI defaults) in last 12 months
    missed_payments = np.random.choice([0, 0, 0, 0, 1, 1, 2, 3], n_samples)
    
    # Credit card usage ratio (0-100%)
    has_credit_card = np.random.choice([0, 1], n_samples, p=[0.40, 0.60])
    credit_utilization = np.where(has_credit_card == 1, np.random.uniform(0, 100, n_samples), 0)
    
    # Existing debt in INR
    existing_debt = np.random.randint(0, 2000000, n_samples)
    
    # Number of credit inquiries in last 6 months
    credit_inquiries = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.30, 0.30, 0.20, 0.10, 0.07, 0.03])
    
    # ─── 3. Banking Behavior ──────────────────────────────────────────────────
    # Average monthly bank balance in INR
    avg_bank_balance = monthly_income * np.random.uniform(0.1, 2.0, n_samples)
    avg_bank_balance = np.clip(avg_bank_balance, 5000, 5000000)
    
    # Number of bank transactions per month
    monthly_transactions = np.random.randint(5, 100, n_samples)
    
    # Monthly spending (as % of income)
    spending_ratio = np.random.uniform(0.3, 0.95, n_samples)
    
    # Has savings account
    has_savings = np.random.choice([0, 1], n_samples, p=[0.10, 0.90])
    
    # Account age (years)
    account_age = np.array([np.random.randint(1, max(2, a - 20)) for a in age])
    
    # ─── 4. Loan Details ──────────────────────────────────────────────────────
    # Loan amount requested (INR)
    loan_amount = np.random.randint(50000, 5000000, n_samples)
    
    # Loan purpose
    loan_purpose = np.random.choice(LOAN_PURPOSES, n_samples)
    
    # Loan tenure (months)
    loan_tenure = np.random.choice([12, 24, 36, 48, 60, 72, 84, 120, 180, 240], n_samples)
    
    # ─── Calculate Risk Score and Target ──────────────────────────────────────
    # Higher score = more likely to be approved
    risk_score = (
        (cibil_score - 300) / 600 * 30 +  # CIBIL score (max 30 points)
        np.minimum(employment_years / 10, 1) * 15 +  # Employment stability (max 15)
        (1 - missed_payments / 3) * 15 +  # Payment history (max 15)
        np.minimum(annual_income / 2000000, 1) * 10 +  # Income level (max 10)
        (loans_repaid / np.maximum(num_past_loans, 1)) * 10 +  # Loan repayment ratio (max 10)
        (1 - credit_utilization / 100) * 5 +  # Credit utilization (max 5)
        (1 - existing_debt / 2000000) * 5 +  # Existing debt (max 5)
        (avg_bank_balance / monthly_income) * 5 +  # Bank balance ratio (max ~5)
        np.random.uniform(-10, 10, n_samples)  # Random factor
    )
    
    # Normalize to 0-100
    risk_score = np.clip(risk_score, 0, 100)
    
    # Loan approval: 1 = Approved, 0 = Rejected
    # Higher risk score = more likely approved
    approval_threshold = 45 + np.random.uniform(-10, 10, n_samples)
    loan_approved = (risk_score > approval_threshold).astype(int)
    
    # ─── Create DataFrame ─────────────────────────────────────────────────────
    df = pd.DataFrame({
        # Personal & Financial
        'age': age,
        'annual_income': annual_income.astype(int),
        'monthly_income': monthly_income.astype(int),
        'employment_years': employment_years,
        'education': education,
        'employment_type': employment_type,
        'state': state,
        'dependents': dependents,
        
        # Credit History
        'cibil_score': cibil_score,
        'num_past_loans': num_past_loans,
        'loans_repaid': loans_repaid,
        'missed_payments': missed_payments,
        'has_credit_card': has_credit_card,
        'credit_utilization': credit_utilization.round(1),
        'existing_debt': existing_debt,
        'credit_inquiries': credit_inquiries,
        
        # Banking Behavior
        'avg_bank_balance': avg_bank_balance.astype(int),
        'monthly_transactions': monthly_transactions,
        'spending_ratio': spending_ratio.round(2),
        'has_savings': has_savings,
        'account_age': account_age,
        
        # Loan Details
        'loan_amount': loan_amount,
        'loan_purpose': loan_purpose,
        'loan_tenure': loan_tenure,
        
        # Target
        'loan_approved': loan_approved
    })
    
    return df


def main():
    print("="*60)
    print("GENERATING INDIAN CREDIT RISK DATASET")
    print("="*60)
    
    # Generate dataset
    df = generate_indian_credit_dataset(SAMPLE_SIZE)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget distribution:")
    print(df['loan_approved'].value_counts())
    print(f"  Approved (1): {(df['loan_approved']==1).sum():,}")
    print(f"  Rejected (0): {(df['loan_approved']==0).sum():,}")
    
    # Save dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved dataset to: {OUTPUT_FILE}")
    
    # Preview
    print("\nSample records:")
    print(df.head(5).to_string())
    
    print("\nColumn summary:")
    print(df.describe().round(2).to_string())
    
    print("\n" + "="*60)
    print("DONE! Now run: python train_model.py")
    print("="*60)


if __name__ == "__main__":
    main()
