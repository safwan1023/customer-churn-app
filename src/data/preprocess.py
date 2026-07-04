import pandas as pd
from load_data import load_raw_data

TARGET = "Churn"


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """The 11 blank-string rows are all tenure=0 (brand-new customers who
    haven't been billed yet). Correct value is 0.0 - not dropped, not
    mean-imputed, since these customers genuinely have zero billing history."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """customerID is a unique identifier with zero predictive signal."""
    return df.drop(columns=["customerID"])


def normalize_no_service_strings(df: pd.DataFrame) -> pd.DataFrame:
    """'No internet service' / 'No phone service' are functionally 'No'
    for modeling purposes - the base service columns already capture
    that fact, so this just removes redundant categories."""
    df = df.copy()
    cols_to_fix = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
    ]
    for col in cols_to_fix:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Converts Churn from Yes/No text into 1/0 integers, required by
    every scikit-learn model."""
    df = df.copy()
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """New feature, directly motivated by EDA: tenure was bimodal (U-shaped),
    so bucketing it into groups may help models capture the 'early churn
    cliff' more explicitly than the raw continuous number does."""
    df = df.copy()
    bins = [-1, 12, 24, 48, 72]
    labels = ["0-12mo", "13-24mo", "25-48mo", "49-72mo"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full cleaning pipeline in the correct order."""
    df = fix_total_charges(df)
    df = normalize_no_service_strings(df)
    df = add_tenure_group(df)
    df = encode_target(df)
    df = drop_id_column(df)
    return df


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)

    print("=" * 60)
    print("CLEANED DATA SHAPE:", cleaned_df.shape)
    print("=" * 60)
    print("\n--- DTYPES ---")
    print(cleaned_df.dtypes)
    print("\n--- CHURN RATE (sanity check, should still be ~0.265) ---")
    print(cleaned_df["Churn"].mean())
    print("\n--- NEW tenure_group COLUMN ---")
    print(cleaned_df["tenure_group"].value_counts())
    print("\n--- ANY NULLS LEFT? ---")
    print(cleaned_df.isna().sum().sum())

    cleaned_df.to_csv("data/processed/churn_clean.csv", index=False)
    print("\nSaved cleaned data to data/processed/churn_clean.csv")