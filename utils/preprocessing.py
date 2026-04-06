"""
Data preprocessing and feature engineering utilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load the Give Me Some Credit dataset and perform initial preprocessing.
    
    Args:
        filepath: Path to the CSV file
        sample_size: Optional sample size for faster iteration
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Drop the unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Remove extreme outliers in RevolvingUtilizationOfUnsecuredLines
    # Values > 1 are unusual (using more than credit limit)
    df = df[df["RevolvingUtilizationOfUnsecuredLines"] <= 10]
    
    # Remove extreme age outliers (age 0 or >100)
    df = df[(df["age"] > 0) & (df["age"] <= 100)]
    
    # Cap extreme values in past due columns
    past_due_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate"
    ]
    for col in past_due_cols:
        # Values of 96 and 98 are special codes, cap at 20
        df[col] = df[col].clip(upper=20)
    
    print(f"Dataset loaded: {len(df)} records, {df.shape[1]} columns")
    print(f"Target distribution:\n{df['SeriousDlqin2yrs'].value_counts(normalize=True).round(3)}")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    MonthlyIncome: Fill with median grouped by age buckets
    NumberOfDependents: Fill with mode (0)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Create age buckets for grouped imputation
    df["age_bucket"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 100], labels=False)
    
    # Fill MonthlyIncome with median by age bucket
    if df["MonthlyIncome"].isna().sum() > 0:
        median_income_by_age = df.groupby("age_bucket")["MonthlyIncome"].transform("median")
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(median_income_by_age)
        # Fill any remaining with overall median
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    
    # Fill NumberOfDependents with 0 (mode)
    if df["NumberOfDependents"].isna().sum() > 0:
        df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)
    
    # Drop the temporary age_bucket column
    df = df.drop(columns=["age_bucket"])
    
    missing_report = df.isna().sum()
    if missing_report.sum() > 0:
        print(f"Remaining missing values:\n{missing_report[missing_report > 0]}")
    else:
        print("All missing values handled.")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features to improve model performance.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # 1. Total times past due (sum of all delinquency counts)
    df["TotalTimesPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    )
    
    # 2. Has any delinquency flag
    df["HasDelinquency"] = (df["TotalTimesPastDue"] > 0).astype(int)
    
    # 3. High credit utilization flag (using more than 80% of credit)
    df["HighUtilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.8).astype(int)
    
    # 4. Age groups
    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)
    
    # 5. Income to debt ratio (inverse of DebtRatio where safe)
    df["IncomeToDebt"] = np.where(
        df["DebtRatio"] > 0,
        1 / df["DebtRatio"],
        df["MonthlyIncome"]  # If no debt, use income as proxy
    )
    df["IncomeToDebt"] = df["IncomeToDebt"].clip(upper=100)  # Cap extreme values
    
    # 6. Has real estate flag
    df["HasRealEstate"] = (df["NumberRealEstateLoansOrLines"] > 0).astype(int)
    
    # 7. Total open accounts
    df["TotalOpenAccounts"] = (
        df["NumberOfOpenCreditLinesAndLoans"] +
        df["NumberRealEstateLoansOrLines"]
    )
    
    # 8. Monthly income per dependent
    df["IncomePerDependent"] = np.where(
        df["NumberOfDependents"] > 0,
        df["MonthlyIncome"] / (df["NumberOfDependents"] + 1),
        df["MonthlyIncome"]
    )
    
    print(f"Engineered features added. Total features: {df.shape[1]}")
    
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "SeriousDlqin2yrs",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Prepare features and target for model training.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Fraction for test split
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test, feature_columns, scaler
    """
    # Define feature columns (exclude target and any ID columns)
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames for interpretability
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(feature_cols)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler
