import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Churn Analytics | Customer Retention",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "churn_model.pkl"


# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown(
    """
    <style>
        :root {
            --primary: #4F46E5;
            --background: #F7F8FC;
            --text: #111827;
            --muted: #6B7280;
        }

        .stApp {
            background: var(--background);
        }

        [data-testid="stHeader"] {
            background: rgba(247, 248, 252, 0.92);
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        [data-testid="stSidebar"] * {
            color: #F9FAFB;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
        }

        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #111827 0%, #312E81 100%);
            color: white;
            margin-bottom: 1.25rem;
        }

        .hero h1 {
            color: white;
            margin: 0 0 0.4rem 0;
            font-size: 2rem;
        }

        .hero p {
            color: #D1D5DB;
            margin: 0;
            font-size: 1rem;
        }

        .section-note {
            color: #6B7280;
            font-size: 0.92rem;
            margin-top: -0.45rem;
            margin-bottom: 1rem;
        }

        .result-card {
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            border: 1px solid #E5E7EB;
            background: white;
            margin-top: 1rem;
        }

        .result-label {
            font-size: 0.85rem;
            color: #6B7280;
            margin-bottom: 0.25rem;
        }

        .result-value {
            font-size: 1.55rem;
            font-weight: 750;
            margin-bottom: 0.35rem;
        }

        .small-muted {
            color: #6B7280;
            font-size: 0.85rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #E5E7EB;
            padding: 1rem 1.1rem;
            border-radius: 14px;
        }

        div[data-testid="stMetricLabel"] {
            color: #6B7280;
        }

        div[data-testid="stForm"] {
            background: white;
            padding: 1.25rem;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 10px;
            font-weight: 650;
            min-height: 2.8rem;
        }

        hr {
            border-color: #E5E7EB;
        }

        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================
@st.cache_data
def load_data(path_string):
    return pd.read_csv(path_string)


@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: `{DATA_PATH.name}`")
    st.info("Place the CSV file in the same folder as `app.py`.")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Saved model not found: `{MODEL_PATH.name}`")
    st.info("Place `churn_model.pkl` in the same folder as `app.py`.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error("The app could not load the dataset or saved model.")
    st.exception(error)
    st.stop()

required_columns = {
    "Churn", "Contract", "MonthlyCharges", "tenure", "TotalCharges",
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling", "PaymentMethod",
}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    st.error(f"The dataset is missing required columns: {', '.join(sorted(missing_columns))}")
    st.stop()

# Make numeric fields numeric for dashboard calculations.
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("## 📊 Churn Analytics")
    st.caption("Customer retention intelligence")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Dashboard", "Predict Churn"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**About this project**")
    st.caption(
        "An interactive analytics app using a saved machine-learning pipeline "
        "to explore customer churn and estimate churn risk."
    )
    st.caption("Built with Python · Pandas · Scikit-learn · Streamlit")


# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <h1>Customer Churn Analytics</h1>
            <p>Explore retention patterns, understand customer behaviour, and identify areas where churn is highest.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_customers = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    retained = int((df["Churn"] == "No").sum())
    churn_rate = (churned / total_customers * 100) if total_customers else 0
    retention_rate = (retained / total_customers * 100) if total_customers else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total customers", f"{total_customers:,}")
    k2.metric("Churned customers", f"{churned:,}")
    k3.metric("Retained customers", f"{retained:,}")
    k4.metric("Churn rate", f"{churn_rate:.1f}%")

    st.markdown("")
    st.subheader("Retention snapshot")
    st.markdown(
        f'<div class="section-note">Of the customers in this dataset, {retention_rate:.1f}% stayed and {churn_rate:.1f}% churned.</div>',
        unsafe_allow_html=True,
    )
    progress_col, insight_col = st.columns([1.5, 1])

    with progress_col:
        st.markdown("**Customer outcome split**")
        split = pd.DataFrame(
            {
                "Customers": [retained, churned],
            },
            index=["Stayed", "Churned"],
        )
        st.bar_chart(split, color=["#4F46E5"], height=230)

    with insight_col:
        st.markdown("**Quick observations**")
        st.info(
            f"**{churned:,} customers** in the dataset have churned. "
            f"That is approximately **1 in {max(1, round(100 / churn_rate))} customers**."
            if churn_rate > 0
            else "No churned customers were found in this dataset."
        )
        if "Contract" in df.columns:
            contract_rates = (
                df.assign(_Churned=df["Churn"].eq("Yes"))
                .groupby("Contract", dropna=False)["_Churned"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
            )
            if not contract_rates.empty:
                highest_contract = str(contract_rates.index[0])
                highest_rate = float(contract_rates.iloc[0])
                st.warning(
                    f"**{highest_contract}** customers have the highest churn rate "
                    f"among contract groups in this dataset ({highest_rate:.1f}%)."
                )

    st.divider()
    st.subheader("Explore churn patterns")
    left, right = st.columns(2)

    with left:
        st.markdown("**Churn rate by contract**")
        contract_rate = (
            df.assign(_Churned=df["Churn"].eq("Yes"))
            .groupby("Contract")["_Churned"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .rename("Churn rate (%)")
        )
        st.bar_chart(contract_rate, color="#EF4444", height=300)
        st.caption("The percentage of customers who churned within each contract type.")

    with right:
        st.markdown("**Average monthly charges by outcome**")
        avg_charges = df.groupby("Churn")["MonthlyCharges"].mean()
        avg_charges.index = avg_charges.index.map({"No": "Stayed", "Yes": "Churned"})
        avg_charges = avg_charges.rename("Average monthly charges ($)")
        st.bar_chart(avg_charges, color="#4F46E5", height=300)
        st.caption("Compares average monthly charges for customers who stayed and churned.")

    st.divider()
    with st.expander("Explore dataset and data quality"):
        st.markdown("**Dataset preview**")
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        m1, m2 = st.columns(2)
        m1.metric("Rows", f"{df.shape[0]:,}")
        m2.metric("Columns", f"{df.shape[1]:,}")
        missing_values = df.isna().sum().rename("Missing values").to_frame()
        missing_values = missing_values[missing_values["Missing values"] > 0]
        st.markdown("**Missing values**")
        if missing_values.empty:
            st.success("No missing values detected in the loaded dataset.")
        else:
            st.dataframe(missing_values, use_container_width=True)


# =========================================================
# PREDICT CHURN
# =========================================================
elif page == "Predict Churn":
    st.markdown(
        """
        <div class="hero">
            <h1>Customer Churn Prediction</h1>
            <p>Enter a customer's subscription and billing details to estimate their churn risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "This estimate comes from a trained machine-learning model. "
        "It supports analysis and should not be treated as a guarantee of future behaviour."
    )

    with st.form("customer_form"):
        st.subheader("1 · Customer profile")
        profile1, profile2, profile3 = st.columns(3)
        with profile1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda value: "Yes" if value == 1 else "No",
            )
        with profile2:
            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        with profile3:
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=100, value=12, step=1
            )

        st.divider()
        st.subheader("2 · Services")
        service1, service2, service3 = st.columns(3)
        with service1:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
        with service2:
            online_security = st.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
        with service3:
            tech_support = st.selectbox(
                "Tech support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )

        st.divider()
        st.subheader("3 · Contract and billing")
        billing1, billing2, billing3 = st.columns(3)
        with billing1:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        with billing2:
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        with billing3:
            monthly_charges = st.number_input(
                "Monthly charges ($)", min_value=0.0, value=70.0, step=1.0
            )
            total_charges = st.number_input(
                "Total charges ($)", min_value=0.0, value=840.0, step=10.0
            )

        submitted = st.form_submit_button(
            "Estimate churn risk", type="primary", use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame(
            [
                {
                    "tenure": tenure,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                    "gender": gender,
                    "Partner": partner,
                    "Dependents": dependents,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "SeniorCitizen": senior_citizen,
                }
            ]
        )

        try:
            prediction = model.predict(customer)[0]
            prediction = prediction.item() if hasattr(prediction, "item") else prediction
            classes = [
                value.item() if hasattr(value, "item") else value
                for value in model.classes_
            ]

            churn_probability = None
            if hasattr(model, "predict_proba") and 1 in classes:
                probabilities = model.predict_proba(customer)[0]
                churn_probability = float(probabilities[classes.index(1)])

            st.divider()
            st.subheader("Prediction result")

            result_col, probability_col = st.columns([1, 1])
            if prediction == 1:
                with result_col:
                    st.error("### Higher churn concern\nThis customer is predicted to churn.")
                if churn_probability is not None:
                    with probability_col:
                        st.metric("Estimated churn probability", f"{churn_probability:.1%}")
                        st.progress(max(0.0, min(1.0, churn_probability)))
            else:
                with result_col:
                    st.success("### Customer likely to stay\nThe model predicts this customer will be retained.")
                if churn_probability is not None:
                    with probability_col:
                        st.metric("Estimated churn probability", f"{churn_probability:.1%}")
                        st.progress(max(0.0, min(1.0, churn_probability)))

            if churn_probability is not None:
                if churn_probability >= 0.70:
                    st.warning(
                        "The model assigns a high churn probability to this customer. "
                        "Consider reviewing their contract, service experience, and support needs."
                    )
                elif churn_probability >= 0.40:
                    st.info(
                        "The estimated churn probability is in a middle range. "
                        "Review this customer's experience before deciding on follow-up."
                    )
                else:
                    st.success(
                        "The model assigns a relatively low churn probability. "
                        "Continue normal customer engagement."
                    )

            with st.expander("View submitted customer details"):
                st.dataframe(customer, use_container_width=True, hide_index=True)

        except Exception as error:
            st.error(
                "Prediction failed. Check that the saved model matches the dataset "
                "and that the model file is the expected trained pipeline."
            )
            st.exception(error)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Customer Churn Analytics · Educational machine-learning project · "
    "Predictions are estimates, not guarantees."
)
