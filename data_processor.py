import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepare_credit_data(df_governed, raw_df):
    """
    Executes advanced feature engineering, standardizes target binary limits,
    safeguards the testing pool from information leakage, and applies SMOTE.
    """
    # 1. Advanced Feature Engineering & Proxy Computations
    df_pipeline = df_governed.copy()
    
    # Generate corporate risk indicators
    df_pipeline['debt_burden_score'] = df_pipeline['installment_income_pct'] * df_pipeline['duration_months']
    df_pipeline['age_to_tenure_ratio'] = df_pipeline['age_years'] / df_pipeline['duration_months']
    
    # 2. Target Variable Realignment to Machine Learning Binary Norms
    # Remap old-school labels: 1 (Safe) becomes 0, 2 (Default) becomes 1
    df_pipeline['target'] = df_pipeline['default_status'].map({1: 0, 2: 1})
    
    # 3. Separate Feature Matrix (X) from the Newly Encoded Label (y)
    X = df_pipeline.drop(columns=['default_status', 'target', 'industry_sector', 'loan_purpose_code'])
    y = df_pipeline['target']
    
    # 4. Stratified Split Configuration (Locking away 20% for validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )
    
    # 5. Internal Training Set Balancing via SMOTE (Leakage Protection Barrier)
    smote_generator = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote_generator.fit_resample(X_train, y_train)
    
    # Compile Pipeline Integrity Diagnostics
    print("==================================================================")
    print("             CREDIT RISK PIPELINE INTEGRITY REPORT               ")
    print("==================================================================")
    print(f"• Engineered Input Features Created   : {X_train_balanced.shape[1]} columns")
    print(f"• Untouched Validation Portfolio Size : {X_test.shape[0]} accounts")
    print(f"• Pre-SMOTE Raw Training Set Size     : {X_train.shape[0]} accounts")
    print(f"• Post-SMOTE Balanced Training Size   : {X_train_balanced.shape[0]} accounts")
    print("\n--- Training Set Class Distribution After Synthetic Balance ---")
    print(f"  - Healthy Portfolio Profiles (Class 0)   : {y_train_balanced.value_counts()[0]} rows")
    print(f"  - Synthesized Default Profiles (Class 1) : {y_train_balanced.value_counts()[1]} rows")
    print("==================================================================")
    
    return X_train_balanced, X_test, y_train_balanced, y_test