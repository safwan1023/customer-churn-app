import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from load_data import load_raw_data
from preprocess import clean_data
from build_features import split_data, build_preprocessor

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss",
                              random_state=42, scale_pos_weight=2.77),
}


def run_comparison():
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(cleaned_df)

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in MODELS.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", model)])
        cv_f1_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "CV_F1_mean": round(cv_f1_scores.mean(), 4),
            "CV_F1_std": round(cv_f1_scores.std(), 4),
            "Test_Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Test_Precision": round(precision_score(y_test, y_pred), 4),
            "Test_Recall": round(recall_score(y_test, y_pred), 4),
            "Test_F1": round(f1_score(y_test, y_pred), 4),
            "Test_ROC_AUC": round(roc_auc_score(y_test, y_proba), 4),
        })

    return pd.DataFrame(results).sort_values("Test_F1", ascending=False)


if __name__ == "__main__":
    df = run_comparison()
    print("=" * 100)
    print("MODEL COMPARISON (5-fold Stratified Cross-Validation + Held-out Test Set)")
    print("=" * 100)
    print(df.to_string(index=False))