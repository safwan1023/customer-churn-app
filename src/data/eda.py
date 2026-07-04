import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_raw_data

FIGURES_DIR = "reports/figures"


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Standard null check. Won't catch hidden blanks like TotalCharges,
    which is exactly why we do a second, targeted check below."""
    return df.isna().sum()


def check_hidden_numeric_nulls(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Catches values that LOOK like numbers but were stored as strings
    with something non-numeric (blank spaces, stray characters) hiding
    inside. This is how we surface the TotalCharges bug properly."""
    coerced = pd.to_numeric(df[column], errors="coerce")
    bad_rows = df[coerced.isna()]
    return bad_rows


def check_duplicates(df: pd.DataFrame, subset: str = "customerID") -> int:
    """Duplicate customerIDs would mean the same customer appears twice —
    a serious data integrity issue if found."""
    return df.duplicated(subset=subset).sum()


def check_target_balance(df: pd.DataFrame) -> pd.Series:
    """Proportions (not raw counts) of each class in the target column."""
    return df["Churn"].value_counts(normalize=True)


def plot_target_distribution(df: pd.DataFrame):
    """Saves a bar chart of Churn Yes/No counts to reports/figures/."""
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
    plt.title("Churn Distribution (Target Variable)")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/target_distribution.png")
    plt.close()


def plot_numeric_boxplots(df: pd.DataFrame):
    """Boxplots reveal outliers: points beyond the whiskers are flagged
    as statistically unusual values (using the IQR method under the hood)."""
    numeric_cols = ["tenure", "MonthlyCharges"]
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(10, 4))
    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(data=df, y=col, ax=ax, color="skyblue")
        ax.set_title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/numeric_boxplots.png")
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame):
    """Histograms with a smoothed density curve (KDE) show the SHAPE of
    each numeric feature's distribution — skewed, bimodal, normal, etc."""
    numeric_cols = ["tenure", "MonthlyCharges"]
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(10, 4))
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(data=df, x=col, kde=True, ax=ax, color="teal")
        ax.set_title(f"Distribution: {col}")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/numeric_distributions.png")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation only works on truly numeric columns. TotalCharges is
    excluded here because it's still stored as a string — we'll redo
    this heatmap in Phase 5 once it's cleaned."""
    numeric_df = df[["SeniorCitizen", "tenure", "MonthlyCharges"]]
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png")
    plt.close()


def plot_categorical_vs_churn(df: pd.DataFrame):
    """For each categorical feature, shows how Churn splits within each
    category — this is where business insight actually lives, more than
    the correlation heatmap does for categorical data."""
    cols_to_plot = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, col in zip(axes, cols_to_plot):
        sns.countplot(data=df, x=col, hue="Churn", palette="Set2", ax=ax)
        ax.set_title(f"{col} vs Churn")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/categorical_vs_churn.png")
    plt.close()


if __name__ == "__main__":
    df = load_raw_data()

    print("=" * 60)
    print("MISSING VALUES (naive .isna() check)")
    print("=" * 60)
    print(check_missing_values(df))

    print("\n" + "=" * 60)
    print("HIDDEN NUMERIC NULLS in TotalCharges")
    print("=" * 60)
    bad = check_hidden_numeric_nulls(df, "TotalCharges")
    print(f"Rows found: {len(bad)}")
    print(bad[["customerID", "tenure", "TotalCharges", "Churn"]])

    print("\n" + "=" * 60)
    print("DUPLICATE customerIDs")
    print("=" * 60)
    print(f"Duplicate count: {check_duplicates(df)}")

    print("\n" + "=" * 60)
    print("TARGET VARIABLE BALANCE")
    print("=" * 60)
    print(check_target_balance(df))

    plot_target_distribution(df)
    print(f"\nSaved plot to {FIGURES_DIR}/target_distribution.png")

    plot_numeric_boxplots(df)
    plot_numeric_distributions(df)
    print(f"Saved boxplots and distributions to {FIGURES_DIR}/")

    plot_correlation_heatmap(df)
    plot_categorical_vs_churn(df)
    print(f"Saved correlation heatmap and categorical plots to {FIGURES_DIR}/")