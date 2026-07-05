import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))
from load_data import load_raw_data
from preprocess import clean_data
from build_features import split_data, build_preprocessor


def train_logistic_regression():
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(cleaned_df)

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_transformed, y_train)

    y_pred = model.predict(X_test_transformed)

    print("=" * 60)
    print("LOGISTIC REGRESSION - BASELINE RESULTS")
    print("=" * 60)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return model, preprocessor


if __name__ == "__main__":
    train_logistic_regression()