import pandas as pd
from load_data import load_raw_data
from preprocess import clean_data

RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges"]
CLEAN_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def compare_correlations():
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)

    print("=" * 60)
    print("CORRELATION - BEFORE CLEANING (TotalCharges excluded, was a string)")
    print("=" * 60)
    print(raw_df[RAW_NUMERIC].corr())

    print("\n" + "=" * 60)
    print("CORRELATION - AFTER CLEANING (TotalCharges included, now numeric)")
    print("=" * 60)
    print(cleaned_df[CLEAN_NUMERIC].corr())

    print("\n" + "=" * 60)
    print("CORRELATION WITH TARGET (Churn) - only possible after cleaning")
    print("=" * 60)
    corr_with_target = cleaned_df[CLEAN_NUMERIC + ["Churn"]].corr()["Churn"].sort_values(ascending=False)
    print(corr_with_target)


if __name__ == "__main__":
    compare_correlations()