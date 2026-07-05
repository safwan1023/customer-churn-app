import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))

import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_raw_data
from preprocess import clean_data
from build_features import split_data

MODEL_PATH = "models/churn_pipeline.joblib"
FIGURES_DIR = "reports/figures"


def generate_shap_summary():
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(cleaned_df)

    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values_churn = shap_values[:, :, 1]
    else:
        shap_values_churn = shap_values

    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values_churn, X_test_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_summary.png", bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {FIGURES_DIR}/shap_summary.png")

    mean_abs_shap = np.abs(shap_values_churn).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    print("\nTop 10 features by mean |SHAP value|:")
    for name, val in ranking[:10]:
        print(f"  {name}: {round(float(val), 4)}")


if __name__ == "__main__":
    generate_shap_summary()