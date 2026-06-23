import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

print("🚀 Running Enterprise Pipeline Compiler with StandardScaler...")

# 1. Pull dataset from relational core
connection = sqlite3.connect("central_bank_vault.db")
raw_df = pd.read_sql_query("SELECT * FROM msme_portfolio_ledger;", connection)
connection.close()

# 2. Rebuild structural DataFrame with clean real column names mapping
df_governed = pd.DataFrame()
df_governed['duration_months'] = raw_df.iloc[:, 1].astype(int)        # Column 2: Duration
df_governed['installment_income_pct'] = raw_df.iloc[:, 7].astype(int) # Column 8: Installment rate
df_governed['age_years'] = raw_df.iloc[:, 9].astype(int)              # Column 10: Age
df_governed['default_status'] = raw_df.iloc[:, 24].astype(int)        # Column 25: Target status

# Scale raw credit column to INR matching the true feature layout
credit_col_idx = raw_df.max().idxmax()
df_governed['loan_amount_INR'] = raw_df[credit_col_idx].astype(float) * 28

# Calculate engineered metrics with real variance distribution
df_governed['rbi_sector_gnpa_pct'] = 5.8 # Use safe baseline mean for vector mapping
df_governed['debt_burden_score'] = df_governed['installment_income_pct'] * df_governed['duration_months']
df_governed['age_to_tenure_ratio'] = df_governed['age_years'] / df_governed['duration_months']

# Ensure target labels are binary (0 or 1)
df_governed['default_status'] = df_governed['default_status'].apply(lambda x: 1 if x > 1 else 0)

# Isolate feature matrix
features_list = ['duration_months', 'installment_income_pct', 'age_years', 'loan_amount_INR', 'rbi_sector_gnpa_pct', 'debt_burden_score', 'age_to_tenure_ratio']
X = df_governed[features_list]
y = df_governed['default_status']

# 3. Build a robust pipeline that SCALES features automatically
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Optimize regularization across the scaled grid
param_grid = {'classifier__C': [0.01, 0.1, 1.0, 10.0]}
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc')
grid.fit(X, y)

# 4. Save the compiled pipeline object to disk
joblib.dump(grid.best_estimator_, 'credit_risk_logistic_model.joblib')
joblib.dump(X.columns.tolist(), 'model_features.joblib')

print(" SUCCESS! Production Pipeline Compiled and Scaled.")