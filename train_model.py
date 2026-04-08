"""
Indian Credit Risk Assessment - Model Training Script

Trains multiple ML models on the Indian Credit Risk dataset:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (best performer)

Predicts whether a loan application will be APPROVED or REJECTED.

Features include:
- Personal & Financial info (age, income, education, employment)
- Credit History (CIBIL score, past loans, missed payments)
- Banking Behavior (account balance, transactions, spending)
- Loan Details (amount, purpose, tenure)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "indian_credit_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "credit_model.pkl")
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def check_xgboost_available():
    """Check if XGBoost is available."""
    try:
        import xgboost as xgb
        return True
    except ImportError:
        return False


def load_and_prepare_data(filepath):
    """Load and prepare the Indian Credit Risk dataset."""
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nTarget distribution:")
    print(df['loan_approved'].value_counts())
    print(f"  Approved (1): {(df['loan_approved']==1).sum():,}")
    print(f"  Rejected (0): {(df['loan_approved']==0).sum():,}")
    
    # Encode categorical columns
    label_encoders = {}
    categorical_cols = ['education', 'employment_type', 'state', 'loan_purpose']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
            label_encoders[col] = le
            df = df.drop(columns=[col])
    
    # Ensure no missing values
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
    
    return df, label_encoders


def engineer_features(df):
    """Create additional features for the model."""
    
    df = df.copy()
    
    # 1. Income to loan ratio
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['income_to_loan_ratio'] = df['income_to_loan_ratio'].clip(0, 100)
    
    # 2. CIBIL score buckets
    df['cibil_bucket'] = pd.cut(
        df['cibil_score'],
        bins=[0, 550, 650, 750, 900],
        labels=[0, 1, 2, 3]
    ).cat.codes.replace(-1, 1)
    
    # 3. Good CIBIL flag
    df['good_cibil'] = (df['cibil_score'] >= 700).astype(int)
    
    # 4. Debt to income ratio
    df['debt_to_income'] = df['existing_debt'] / (df['annual_income'] + 1)
    df['debt_to_income'] = df['debt_to_income'].clip(0, 10)
    
    # 5. High debt flag
    df['high_debt'] = (df['debt_to_income'] > 0.5).astype(int)
    
    # 6. EMI affordability (estimated monthly payment / monthly income)
    estimated_emi = df['loan_amount'] / df['loan_tenure']
    df['emi_to_income'] = estimated_emi / (df['monthly_income'] + 1)
    df['emi_to_income'] = df['emi_to_income'].clip(0, 1)
    
    # 7. Employment stability
    df['stable_employment'] = (df['employment_years'] >= 2).astype(int)
    df['experienced'] = (df['employment_years'] >= 5).astype(int)
    
    # 8. Credit behavior
    df['repayment_ratio'] = df['loans_repaid'] / (df['num_past_loans'] + 1)
    df['has_missed_payments'] = (df['missed_payments'] > 0).astype(int)
    
    # 9. Banking health
    df['savings_ratio'] = df['avg_bank_balance'] / (df['monthly_income'] + 1)
    df['savings_ratio'] = df['savings_ratio'].clip(0, 10)
    
    # 10. Age groups
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=[0, 1, 2, 3, 4]
    ).cat.codes.replace(-1, 2)
    
    # 11. High credit utilization
    df['high_utilization'] = (df['credit_utilization'] > 70).astype(int)
    
    print(f"Engineered features added. Total features: {df.shape[1]}")
    
    return df


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics."""
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
    
    return metrics


def train_models(X_train, X_test, y_train, y_test, feature_cols):
    """Train and evaluate multiple models."""
    
    results = []
    models = {}
    
    # 1. Logistic Regression
    print("\n" + "-"*60)
    print("Training Logistic Regression...")
    print("-"*60)
    
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        solver="lbfgs"
    )
    lr_model.fit(X_train, y_train)
    
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results.append(lr_metrics)
    models["Logistic Regression"] = lr_model
    
    # 2. Random Forest
    print("\n" + "-"*60)
    print("Training Random Forest...")
    print("-"*60)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results.append(rf_metrics)
    models["Random Forest"] = rf_model
    
    # 3. XGBoost (if available) - WITH GPU SUPPORT
    if check_xgboost_available():
        import xgboost as xgb
        
        print("\n" + "-"*60)
        print("Training XGBoost with GPU (RTX 4050)...")
        print("-"*60)
        
        # Check GPU availability
        gpu_available = False
        try:
            # Test if CUDA is available for XGBoost
            test_model = xgb.XGBClassifier(device='cuda', n_estimators=1)
            test_model.fit([[1,2],[3,4]], [0,1])
            gpu_available = True
            print("GPU (CUDA) detected - using GPU acceleration!")
        except Exception as e:
            print(f"GPU not available, using CPU: {str(e)[:50]}...")
        
        xgb_model = xgb.XGBClassifier(
            device='cuda' if gpu_available else 'cpu',
            tree_method='hist',
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            eval_metric="auc"
        )
        xgb_model.fit(X_train, y_train, verbose=True)
        
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        results.append(xgb_metrics)
        models["XGBoost"] = xgb_model
    else:
        print("\nNote: XGBoost not installed. Run: pip install xgboost")
    
    return results, models


def main():
    """Main training pipeline."""
    
    print("="*60)
    print("INDIAN CREDIT RISK ASSESSMENT - MODEL TRAINING")
    print("="*60)
    print("Dataset: Indian Credit Risk Data")
    
    # ─── Check if dataset exists ──────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Dataset not found at {DATA_PATH}")
        print("\nPlease run 'python combine_datasets.py' first to generate the dataset.")
        print("\nOr run with --demo flag for demo mode:")
        print("  python train_model.py --demo")
        
        if "--demo" in sys.argv:
            run_demo_mode()
            return
        
        sys.exit(1)
    
    # ─── Load and preprocess data ─────────────────────────────────────────────
    print("\nLoading dataset...")
    df, label_encoders = load_and_prepare_data(DATA_PATH)
    
    print("\nEngineering features...")
    df = engineer_features(df)
    
    # ─── Prepare features and target ──────────────────────────────────────────
    print("\nPreparing train/test split...")
    
    target_col = 'loan_approved'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Features: {len(feature_cols)}")
    
    # ─── Train models ─────────────────────────────────────────────────────────
    results, models = train_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_cols)
    
    # ─── Compare and select best model ────────────────────────────────────────
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    results_df = pd.DataFrame(results).set_index('model_name')
    print(results_df.round(4).to_string())
    
    best_model_name = results_df['roc_auc'].idxmax()
    best_model = models[best_model_name]
    best_idx = [r['model_name'] for r in results].index(best_model_name)
    
    print(f"\nBest model: {best_model_name} (ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f})")
    
    # ─── Cross-validation ─────────────────────────────────────────────────────
    print(f"\nCross-validating {best_model_name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # ─── Save model artifact ──────────────────────────────────────────────────
    model_artifact = {
        "model": best_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "label_encoders": label_encoders,
        "model_name": best_model_name,
        "metrics": results[best_idx],
        "cv_auc_mean": cv_scores.mean(),
        "cv_auc_std": cv_scores.std(),
    }
    
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel: {best_model_name}")
    print(f"ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f}")
    print(f"Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nRun 'python app.py' to start the web server.")


def run_demo_mode():
    """Run with synthetic data for demo purposes."""
    
    print("\n" + "-"*60)
    print("RUNNING IN DEMO MODE (synthetic data)")
    print("-"*60)
    
    # Import the generator from combine_datasets
    from combine_datasets import generate_indian_credit_dataset
    
    df = generate_indian_credit_dataset(10000)
    
    # Encode categorical columns
    categorical_cols = ['education', 'employment_type', 'state', 'loan_purpose']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
            df = df.drop(columns=[col])
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare data
    target_col = 'loan_approved'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    metrics = evaluate_model(model, X_test_scaled, y_test, "Logistic Regression (Demo)")
    
    # Save model
    model_artifact = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "label_encoders": label_encoders,
        "model_name": "Logistic Regression (Demo)",
        "metrics": metrics,
        "demo_mode": True,
    }
    
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"\nDemo model saved to {MODEL_PATH}")
    print("\nNote: This is demo mode with synthetic data.")
    print("For production, run 'python combine_datasets.py' first.")


if __name__ == "__main__":
    main()
