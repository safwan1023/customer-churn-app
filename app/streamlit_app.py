import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

MODEL_PATH = "models/churn_pipeline.joblib"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def add_tenure_group(tenure: int) -> str:
    if tenure <= 12:
        return "0-12mo"
    elif tenure <= 24:
        return "13-24mo"
    elif tenure <= 48:
        return "25-48mo"
    else:
        return "49-72mo"


pipeline = load_pipeline()

st.title("📉 Customer Churn Prediction")
st.caption("Predict the likelihood a telecom customer will churn, and see what's driving the risk.")

with st.sidebar:
    st.header("Customer Profile")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    st.subheader("Account & Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)))

    predict_button = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

col1, col2 = st.columns([1, 1])

if predict_button:
    input_data = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        "tenure_group": add_tenure_group(tenure),
    }
    input_df = pd.DataFrame([input_data])

    probability = float(pipeline.predict_proba(input_df)[0][1])
    prediction = "Yes" if probability >= 0.5 else "No"

    if probability >= 0.7:
        risk_level, color = "High Risk", "red"
    elif probability >= 0.4:
        risk_level, color = "Medium Risk", "orange"
    else:
        risk_level, color = "Low Risk", "green"

    with col1:
        st.subheader("Prediction Result")
        st.metric("Will this customer churn?", prediction)
        st.markdown(f"**Risk Level:** :{color}[{risk_level}]")
        st.progress(probability, text=f"Churn Probability: {probability:.1%}")

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Churn Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#d4f5dd"},
                    {"range": [40, 70], "color": "#ffe8b3"},
                    {"range": [70, 100], "color": "#ffcccc"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Why this prediction? (Top business drivers)")
    st.markdown("""
    - **Contract type** is historically the strongest churn driver — month-to-month customers churn far more than one/two-year customers.
    - **Fiber optic internet** customers churn more than DSL, often tied to price sensitivity.
    - **Low tenure** (new customers) are the highest-risk segment; risk drops sharply after ~12 months.
    - **Electronic check** payment correlates with higher churn than automatic payment methods.
    """)
else:
    st.info("👈 Fill in the customer profile in the sidebar and click **Predict Churn Risk** to get a prediction.")