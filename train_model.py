"""
Credit Risk Assessment - Model Training Script

Trains multiple ML models on the Give Me Some Credit dataset:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (best performer)

Handles missing values, engineers features, addresses class imbalance,
and saves the best model for the Flask backend.

Dataset: Give Me Some Credit (Kaggle)
https://www.kaggle.com/competitions/GiveMeSomeCredit
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import (
    load_and_preprocess_data,
    handle_missing_values,
    engineer_features,
    prepare_features_and_target,
)
from utils.evaluation import (
    evaluate_model,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    compare_models,
)

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "cs-training.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "credit_model.pkl")
RANDOM_STATE = 42

# Set random seed for reproducibility
np.random.seed(RANDOM_STATE)


def check_xgboost_available():
    """Check if XGBoost is available."""
    try:
        import xgboost as xgb
        return True
    except ImportError:
        return False


def train_models(X_train, X_test, y_train, y_test, feature_cols):
    """Train and evaluate multiple models."""
    
    results = []
    models = {}
    
    # Calculate class weight for imbalanced data
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nClass imbalance ratio: {pos_weight:.2f}:1 (negative:positive)")
    
    # ─── 1. Logistic Regression ───────────────────────────────────────────────
    print("\n" + "─"*60)
    print("Training Logistic Regression...")
    print("─"*60)
    
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",  # Handle imbalance
        solver="lbfgs"
    )
    lr_model.fit(X_train, y_train)
    
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results.append(lr_metrics)
    models["Logistic Regression"] = lr_model
    
    # ─── 2. Random Forest ─────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("Training Random Forest...")
    print("─"*60)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
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
    
    # ─── 3. XGBoost (if available) ────────────────────────────────────────────
    if check_xgboost_available():
        import xgboost as xgb
        
        print("\n" + "─"*60)
        print("Training XGBoost...")
        print("─"*60)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,  # Handle imbalance
            random_state=RANDOM_STATE,
            eval_metric="auc",
            use_label_encoder=False
        )
        xgb_model.fit(X_train, y_train)
        
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        results.append(xgb_metrics)
        models["XGBoost"] = xgb_model
    else:
        print("\n⚠ XGBoost not installed. Run: pip install xgboost")
    
    return results, models


def main():
    """Main training pipeline."""
    
    print("="*60)
    print("CREDIT RISK ASSESSMENT - MODEL TRAINING")
    print("="*60)
    
    # ─── Check if dataset exists ──────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ ERROR: Dataset not found at {DATA_PATH}")
        print("\nPlease download the 'Give Me Some Credit' dataset:")
        print("1. Go to: https://www.kaggle.com/competitions/GiveMeSomeCredit/data")
        print("2. Download 'cs-training.csv'")
        print("3. Place it in the 'dataset/' folder")
        print("\nAlternatively, run with --demo flag for demo mode with synthetic data:")
        print("  python train_model.py --demo")
        
        # Check for --demo flag
        if "--demo" in sys.argv:
            print("\n" + "─"*60)
            print("RUNNING IN DEMO MODE (synthetic data)")
            print("─"*60)
            run_demo_mode()
            return
        
        sys.exit(1)
    
    # ─── Load and preprocess data ─────────────────────────────────────────────
    print("\n📊 Loading dataset...")
    df = load_and_preprocess_data(DATA_PATH)
    
    print("\n🔧 Handling missing values...")
    df = handle_missing_values(df)
    
    print("\n⚙️ Engineering features...")
    df = engineer_features(df)
    
    # ─── Prepare features and split ───────────────────────────────────────────
    print("\n📦 Preparing train/test split...")
    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_features_and_target(df)
    
    # ─── Train models ─────────────────────────────────────────────────────────
    results, models = train_models(X_train, X_test, y_train, y_test, feature_cols)
    
    # ─── Compare models ───────────────────────────────────────────────────────
    comparison_df = compare_models(results)
    
    # ─── Select best model ────────────────────────────────────────────────────
    best_model_name = comparison_df["roc_auc"].idxmax()
    best_model = models[best_model_name]
    
    print(f"\n✅ Best model selected: {best_model_name}")
    
    # ─── Cross-validation for best model ──────────────────────────────────────
    print(f"\n📈 Cross-validating {best_model_name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # ─── Save model artifact ──────────────────────────────────────────────────
    model_artifact = {
        "model": best_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "model_name": best_model_name,
        "metrics": results[comparison_df.index.get_loc(best_model_name)],
        "cv_auc_mean": cv_scores.mean(),
        "cv_auc_std": cv_scores.std(),
    }
    
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")
    
    # ─── Generate visualizations ──────────────────────────────────────────────
    print("\n📊 Generating visualizations...")
    
    # Prepare data for ROC curve
    roc_data = []
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_data.append((name, y_test, y_prob))
    
    plot_roc_curve(roc_data, save_path="roc_curves.png")
    
    # Feature importance for best model
    plot_feature_importance(best_model, feature_cols, save_path="feature_importance.png")
    
    # Confusion matrix for best model
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, best_model_name, save_path="confusion_matrix.png")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel: {best_model_name}")
    print(f"ROC-AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nRun 'python app.py' to start the web server.")


def run_demo_mode():
    """Run with synthetic data for demo purposes."""
    
    np.random.seed(RANDOM_STATE)
    n_samples = 5000
    
    # Generate synthetic data matching Give Me Some Credit schema
    df = pd.DataFrame({
        "SeriousDlqin2yrs": np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], n_samples),
        "RevolvingUtilizationOfUnsecuredLines": np.random.uniform(0, 1.5, n_samples),
        "age": np.random.randint(21, 75, n_samples),
        "NumberOfTime30-59DaysPastDueNotWorse": np.random.choice([0, 0, 0, 0, 1, 2], n_samples),
        "DebtRatio": np.random.uniform(0, 2, n_samples),
        "MonthlyIncome": np.random.randint(1000, 20000, n_samples),
        "NumberOfOpenCreditLinesAndLoans": np.random.randint(0, 20, n_samples),
        "NumberOfTimes90DaysLate": np.random.choice([0, 0, 0, 0, 0, 1], n_samples),
        "NumberRealEstateLoansOrLines": np.random.choice([0, 0, 1, 1, 2], n_samples),
        "NumberOfTime60-89DaysPastDueNotWorse": np.random.choice([0, 0, 0, 0, 1], n_samples),
        "NumberOfDependents": np.random.choice([0, 0, 0, 1, 1, 2, 3], n_samples),
    })
    
    print(f"Generated synthetic dataset: {len(df)} records")
    
    # Apply same preprocessing pipeline
    df = engineer_features(df)
    
    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_features_and_target(df)
    
    # Train simple model for demo
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test, "Logistic Regression (Demo)")
    
    # Save model
    model_artifact = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "model_name": "Logistic Regression (Demo)",
        "metrics": metrics,
        "demo_mode": True,
    }
    
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"\n💾 Demo model saved to {MODEL_PATH}")
    print("\n⚠️ Note: This is demo mode with synthetic data.")
    print("For production, download the real dataset.")


if __name__ == "__main__":
    main()
