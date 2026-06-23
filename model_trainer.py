import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains Logistic Regression, Random Forest, and XGBoost models
    on balanced credit arrays and outputs comprehensive operational risk metrics.
    """
    
    # 1. Initialize our three algorithms with locked seed states for reproducibility
    models = {
        "Baseline Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Ensemble Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Production XGBoost Champion": XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42, eval_metric='logloss')
    }
    
    trained_models = {}
    
    # 2. Iterate through the algorithms to train and test
    for name, model in models.items():
        print(f"\nTraining {name} on balanced portfolio...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict class outcomes and probability curves on the validation set
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # Compute critical risk metrics
        auc_score = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        # 3. Print out highly scannable business scorecards
        print("==================================================================")
        print(f"                {name.upper()} RISK SCORECARD")
        print("==================================================================")
        print(f"• Area Under ROC Curve (ROC-AUC) : {auc_score:.4f}")
        print("\n• Confusion Matrix Breakdown:")
        print(f"  [Predicted Safe, Actual Safe] (True Negatives) : {cm[0][0]}")
        print(f"  [Predicted Def, Actual Safe] (False Alarms)   : {cm[0][1]}")
        print(f"  [Predicted Safe, Actual Def] (Missed Defaults): {cm[1][0]}")
        print(f"  [Predicted Def, Actual Def] (Caught Defaults) : {cm[1][1]}")
        
        print("\n• Detailed Classification Metrics:")
        print(classification_report(y_test, preds, target_names=['Safe (0)', 'Default (1)']))
        print("==================================================================")
        
    return trained_models