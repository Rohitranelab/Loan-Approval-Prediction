import streamlit as st
import pandas as pd
import pickle


# Load model
with open("model/loan_approval.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("model/scaling.pkl", "rb") as file:
    scaler = pickle.load(file)


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦"
)

st.title("🏦 Loan Approval Prediction")


col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Income", min_value=0.0)
    credit_score = st.number_input("Credit Score", min_value=0.0)

with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    years_employed = st.number_input("Years Employed", min_value=0.0)


if st.button("Predict"):

    input_data = pd.DataFrame({
        "income": [income],
        "credit_score": [credit_score],
        "loan_amount": [loan_amount],
        "years_employed": [years_employed]
    })


    # Apply same scaling used during training
    input_scaled = scaler.transform(input_data)


    prediction = model.predict(input_scaled)


    probability = model.predict_proba(input_scaled)

    confidence = probability[0][prediction[0]] * 100


    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
