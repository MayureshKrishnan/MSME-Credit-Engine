import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

print("🚀 Compiling High-Variance Credit Risk Engine...")

# 1. Synthesize a clean, high-variance dataset matching the UI parameters perfectly
np.random.seed(42)
n_samples = 2000

duration = np.random.randint(6, 72, n_samples)
installment = np.random.randint(1, 11, n_samples)
age = np.random.randint(18, 70, n_samples)
loan_amount = np.random.randint(10000, 1000000, n_samples)
gnpa = np.random.choice([3.1, 5.8, 8.4, 9.2, 12.4], n_samples)

debt_burden = installment * duration
age_to_tenure = age / duration

# 2. Define clear mathematical credit rules so sliders respond logically
# Higher debt, longer duration, younger age, and high GNPA increase default probability
log_odds = (
    0.05 * (loan_amount / 100000) + 
    0.04 * duration + 
    0.2 * installment + 
    0.15 * gnpa - 
    0.06 * age - 2.0
)
prob = 1 / (1 + np.exp(-log_odds))
default_status = (prob > np.random.uniform(0, 1, n_samples)).astype(int)

# Assemble structural DataFrame
X = pd.DataFrame({
    'duration_months': duration,
    'installment_income_pct': installment,
    'age_years': age,
    'loan_amount_INR': loan_amount,
    'rbi_sector_gnpa_pct': gnpa,
    'debt_burden_score': debt_burden,
    'age_to_tenure_ratio': age_to_tenure
})
y = default_status

# 3. Create a stable, scaled production pipeline
production_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])
production_pipeline.fit(X, y)

# 4. Export exact layouts to files
joblib.dump(production_pipeline, 'credit_risk_logistic_model.joblib')
joblib.dump(X.columns.tolist(), 'model_features.joblib')

print(" SUCCESS: High-variance pipeline successfully generated!")