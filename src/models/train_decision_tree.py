import sys
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))
from load_data import load_raw_data
from preprocess import clean_data
from build_features import split_data, build_preprocessor


def train_decision_tree():
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(cleaned_df)

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # max_depth=5 is a deliberate constraint: an unconstrained tree
    # (max_depth=None) would likely overfit badly on this dataset,
    # memorizing training noise instead of learning general patterns.
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train_transformed, y_train)

    y_pred = model.predict(X_test_transformed)

    print("=" * 60)
    print("DECISION TREE (max_depth=5) - RESULTS")
    print("=" * 60)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Feature importance - a built-in tree attribute showing which
    # features drove the most impurity reduction across all splits.
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]

    print("\nTop 5 most important features:")
    for name, score in top_features:
        print(f"  {name}: {round(score, 4)}")

    return model, preprocessor


if __name__ == "__main__":
    train_decision_tree()