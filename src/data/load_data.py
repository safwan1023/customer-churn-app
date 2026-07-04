import pandas as pd

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn dataset from disk."""
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_raw_data()

    print("=" * 60)
    print("SHAPE (rows, columns):", df.shape)
    print("=" * 60)

    print("\n--- INFO ---")
    print(df.info())

    print("\n--- DESCRIBE (numeric columns) ---")
    print(df.describe())

    print("\n--- FIRST 5 ROWS ---")
    print(df.head())