import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

print("🚀 Running Terminal Hotfix Compiler...")

# 1. Connect to database and extract identical features
connection = sqlite3.connect("central_bank_vault.db")
sql_query = """
    SELECT 
        "1" AS duration_months,
        "7" AS installment_income_pct,
        "9" AS age_years,
        "3" AS loan_purpose_code,
        "24" AS default_status
    FROM 
        msme_portfolio_ledger;
"""
sql_df = pd.read_sql_query(sql_query, connection)
raw_df = pd.read_sql_query("SELECT * FROM msme_portfolio_ledger;", connection)
connection.close()

# 2. Rebuild structural DataFrame without credit_amount_DM
df_governed = pd.DataFrame()
df_governed['duration_months'] = sql_df['duration_months'].astype(int)
df_governed['age_years'] = sql_df['age_years'].astype(int)
df_governed['installment_income_pct'] = sql_df['installment_income_pct'].astype(int)
df_governed['default_status'] = sql_df['default_status'].astype(int)

credit_col_idx = raw_df.max().idxmax()
df_governed['loan_amount_INR'] = raw_df[credit_col_idx].astype(float) * 28

# Map sectors and features
def map_rbi_sector_metrics(code):
    if 1 <= code <= 5: return 12.4
    elif 6 <= code <= 10: return 9.2
    elif 11 <= code <= 15: return 5.8
    elif 16 <= code <= 20: return 3.1
    else: return 8.4

df_governed['rbi_sector_gnpa_pct'] = [map_rbi_sector_metrics(int(c)) for c in sql_df['loan_purpose_code']]
df_governed['debt_burden_score'] = df_governed['installment_income_pct'] * df_governed['duration_months']
df_governed['age_to_tenure_ratio'] = df_governed['age_years'] / df_governed['duration_months']

# 3. Isolate target array vectors
X = df_governed[['duration_months', 'installment_income_pct', 'age_years', 'loan_amount_INR', 'rbi_sector_gnpa_pct', 'debt_burden_score', 'age_to_tenure_ratio']]
y = df_governed['default_status']

# 4. Train the optimized model architecture 
print("⚙️ Tuning model parameters and locking features...")
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {'C': [0.1, 1.0, 10.0]}, cv=3, scoring='roc_auc')
lr_grid.fit(X, y)

# 5. Overwrite the files with the absolute true arrays
joblib.dump(lr_grid.best_estimator_, 'credit_risk_logistic_model.joblib')
joblib.dump(X.columns.tolist(), 'model_features.joblib')

print("SUCCESS! 'credit_risk_logistic_model.joblib' and 'model_features.joblib' have been forced onto your disk!")
print("Expected Columns Layout:", X.columns.tolist())