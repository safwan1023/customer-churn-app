import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))

import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score

from load_data import load_raw_data
from preprocess import clean_data
from build_features import split_data, build_preprocessor

PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [6, 8, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}


def build_pipeline():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
    ])


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(cleaned_df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = build_pipeline()

    search = RandomizedSearchCV(
        pipe, param_distributions=PARAM_GRID, n_iter=15,
        scoring="f1", cv=cv, random_state=42, n_jobs=-1,
    )

    print("=" * 70)
    print("RANDOMIZED SEARCH CV (15 combinations, 5-fold CV each)")
    print("=" * 70)
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"Time taken: {time.time() - t0:.1f}s")
    print(f"Best CV F1 score: {round(search.best_score_, 4)}")
    print(f"Best params: {search.best_params_}")

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n--- Tuned model performance on held-out TEST set ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/churn_pipeline.joblib")
    print("\nSaved tuned pipeline to models/churn_pipeline.joblib")