import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from load_data import load_raw_data
from preprocess import clean_data

TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_group",
]


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Splits BEFORE any scaling/encoding is fit, to prevent data leakage.
    stratify=y ensures both train and test sets keep the same ~26.5% churn
    ratio - critical on imbalanced data, otherwise a random split could
    accidentally produce a test set with a very different churn rate."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Bundles scaling and encoding into a single reusable object.
    ColumnTransformer applies different transformations to different
    column groups in one step, and - critically - it will be fit ONLY
    on training data (see main block below)."""
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return preprocessor


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)

    X_train, X_test, y_train, y_test = split_data(cleaned_df)

    print("=" * 60)
    print("TRAIN/TEST SPLIT SHAPES")
    print("=" * 60)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train churn rate:", y_train.mean().round(4))
    print("y_test churn rate:", y_test.mean().round(4))

    preprocessor = build_preprocessor()

    # fit_transform on TRAIN (learns the scaling mean/std and the
    # one-hot categories from training data only)
    X_train_transformed = preprocessor.fit_transform(X_train)

    # transform ONLY (reuses what was learned from train - never refit)
    X_test_transformed = preprocessor.transform(X_test)

    print("\n" + "=" * 60)
    print("AFTER ENCODING + SCALING")
    print("=" * 60)
    print("X_train_transformed shape:", X_train_transformed.shape)
    print("X_test_transformed shape:", X_test_transformed.shape)
    print("\nFeature names after one-hot encoding:")
    print(preprocessor.get_feature_names_out())