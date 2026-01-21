import streamlit as st
import pandas as pd
import pickle

# =====================================================
# Load trained objects
# =====================================================
model = pickle.load(open("loan_logreg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# =====================================================
# Page config & styling
# =====================================================
st.set_page_config(
    page_title="CreditWise Loan Approval",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main { padding: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #f5f7fa;
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# Header
# =====================================================
st.markdown("## 🏦 CreditWise – Intelligent Loan Approval System")
st.caption(
    "ML-powered system to assist banks in fast, consistent, and unbiased loan approval decisions."
)
st.divider()

# =====================================================
# SIDEBAR – QUICK SCENARIOS
# =====================================================
st.sidebar.header("⚡ Quick Scenarios")

scenario = st.sidebar.selectbox(
    "Load Example Profile",
    ["Custom", "Low Risk Applicant", "High Risk Applicant"]
)

# Default values
credit_score = 700
dti_ratio = 0.30
existing_loans = 0
applicant_income = 50000

if scenario == "Low Risk Applicant":
    credit_score = 780
    dti_ratio = 0.25
    existing_loans = 0
    applicant_income = 120000

elif scenario == "High Risk Applicant":
    credit_score = 550
    dti_ratio = 0.75
    existing_loans = 3
    applicant_income = 30000

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("👤 Applicant Details")

age = st.sidebar.number_input("Age", 18, 65)
education = st.sidebar.selectbox("Education Level", le.classes_.tolist())
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Single"])
dependents = st.sidebar.number_input("Dependents", 0, 10)

# -----------------------------------------------------
st.sidebar.header("💼 Employment & Income")

employment_status = st.sidebar.selectbox(
    "Employment Status",
    ["Salaried", "Self-Employed", "Business"]
)

employer_category = st.sidebar.selectbox(
    "Employer Category",
    ["Govt", "Private", "Self"]
)

applicant_income = st.sidebar.number_input(
    "Applicant Monthly Income",
    min_value=0,
    value=applicant_income
)

coapplicant_income = st.sidebar.number_input(
    "Coapplicant Monthly Income",
    min_value=0
)

# -----------------------------------------------------
st.sidebar.header("💳 Financial Profile")

credit_score = st.sidebar.number_input(
    "Credit Score",
    300, 900,
    value=credit_score,
    help="Higher score indicates better creditworthiness"
)

existing_loans = st.sidebar.number_input(
    "Existing Loans",
    0, 10,
    value=existing_loans
)

dti_ratio = st.sidebar.slider(
    "DTI Ratio",
    0.0, 1.0,
    value=dti_ratio,
    help="Debt-to-Income ratio (lower is better)"
)

savings = st.sidebar.number_input("Savings Balance", min_value=0)
collateral_value = st.sidebar.number_input("Collateral Value", min_value=0)

# -----------------------------------------------------
st.sidebar.header("🏠 Loan Details")

loan_amount = st.sidebar.number_input("Loan Amount Requested", min_value=0)
loan_term = st.sidebar.slider("Loan Term (Months)", 6, 360, 120)

loan_purpose = st.sidebar.selectbox(
    "Loan Purpose",
    ["Home", "Education", "Personal", "Business"]
)

property_area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semi-Urban", "Rural"]
)

# Reset button
st.sidebar.divider()
if st.sidebar.button("🔄 Reset All Inputs"):
    st.experimental_rerun()

# =====================================================
# BUILD INPUT DATAFRAME
# =====================================================
input_df = pd.DataFrame([{
    "Applicant_Income": applicant_income,
    "Coapplicant_Income": coapplicant_income,
    "Employment_Status": employment_status,
    "Age": age,
    "Marital_Status": marital_status,
    "Dependents": dependents,
    "Credit_Score": credit_score,
    "Existing_Loans": existing_loans,
    "DTI_Ratio": dti_ratio,
    "Savings": savings,
    "Collateral_Value": collateral_value,
    "Loan_Amount": loan_amount,
    "Loan_Term": loan_term,
    "Loan_Purpose": loan_purpose,
    "Property_Area": property_area,
    "Education_Level": education.strip(),
    "Gender": gender,
    "Employer_Category": employer_category
}])

# =====================================================
# Encode Education_Level
# =====================================================
input_df["Education_Level"] = le.transform(input_df["Education_Level"])

# =====================================================
# One-Hot Encoding
# =====================================================
cat_cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]

encoded = ohe.transform(input_df[cat_cols])
encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(cat_cols)
)

# =====================================================
# Feature Engineering
# =====================================================
input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"] ** 2
input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2

# =====================================================
# Final feature alignment
# =====================================================
input_df = pd.concat(
    [
        input_df.drop(columns=cat_cols + ["DTI_Ratio", "Credit_Score"]),
        encoded_df
    ],
    axis=1
)

input_df = input_df.reindex(columns=feature_columns, fill_value=0)
input_df = input_df.apply(pd.to_numeric)

# =====================================================
# Scaling
# =====================================================
input_scaled = scaler.transform(input_df.values)

# =====================================================
# MAIN OUTPUT
# =====================================================





predict_btn = st.button(
    "Evaluate Loan Application",
    use_container_width=True
)

if predict_btn:
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0d1b2a, #1b263b);
            padding: 28px;
            border-radius: 16px;
            text-align: center;
            color: #e0e1dd;
            margin-top: 10px;
        ">
            <div style="font-size: 14px; opacity: 0.8;">
                Approval Probability
            </div>
            <div style="font-size: 42px; font-weight: bold;">
                {probability:.2%}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(probability)
    if prediction == 1:
        st.info(
            "🟢 Applicant meets key financial and credit criteria. "
            "Final approval subject to document verification."
        )
    else:
        st.warning(
            "🟠 Applicant does not meet current risk thresholds. "
            "Manual review or revised loan terms may be considered."
        )

# =====================================================
# Explainability
# =====================================================
with st.expander("🧠 How does the system decide?"):
    st.write(
        """
        The model uses Logistic Regression to evaluate applicant income,
        credit score, debt-to-income ratio, existing liabilities,
        employment stability, and loan characteristics.

        Higher credit score, lower DTI ratio, stable income, and sufficient
        collateral increase approval probability.
        """
    )

st.divider()
st.caption(
    "© 2026 CreditWise • AI-assisted decision support system • For demo & educational use"
)
