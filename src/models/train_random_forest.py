import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))
from load_data import load_raw_data
from preprocess import clean_data
from build_features import split_data, build_preprocessor


def train_random_forest():
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(cleaned_df)

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # n_estimators=200: number of trees in the forest - the "vote count"
    # max_depth=8: each individual tree still gets some depth constraint,
    #   preventing any single tree from being wildly overfit before voting
    # class_weight="balanced": tells the model to weight the minority
    #   (Churn) class more heavily during training, directly countering
    #   the 73.5/26.5 imbalance we've known about since Phase 4
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_transformed, y_train)

    y_pred = model.predict(X_test_transformed)

    print("=" * 60)
    print("RANDOM FOREST - RESULTS")
    print("=" * 60)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]

    print("\nTop 5 most important features:")
    for name, score in top_features:
        print(f"  {name}: {round(score, 4)}")

    return model, preprocessor


if __name__ == "__main__":
    train_random_forest()