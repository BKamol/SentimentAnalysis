import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb


@st.cache_data
def predict(credit_score, geography, gender,
            age, tenure, balance, number_of_products,
            has_cr_card, is_active_member, estimated_salary):
    with open("./models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    lgb_model = lgb.Booster(model_file='./models/lgbm_model.txt')

    gender = 1 if gender == "Male" else 0
    input = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [number_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })
    input = pd.DataFrame(preprocessor.transform(input),
                         columns=preprocessor.get_feature_names_out())
    y_pred = lgb_model.predict(input)
    return "Client will exit" if y_pred >= 0.5 else "Client won't exit"


def main():
    st.title("Churn Prediction")
    st.subheader("Input information about client")

    credit_score = st.number_input("Credit score", 300, 900)
    age = st.number_input("Age", 18, 100)
    balance = st.number_input("Balance", 0, 500000)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.radio("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure", 0, 10)
    number_of_products = st.slider("Number of products", 1, 5)
    has_cr_card = st.checkbox("Has credit card", 1)
    is_active_member = st.checkbox("Is active member", 1)
    estimated_salary = st.number_input("Estimated salary", 10, 500000)

    y_pred = predict(credit_score, geography, gender,
                     age, tenure, balance, number_of_products,
                     has_cr_card, is_active_member, estimated_salary)
    st.subheader(y_pred)


main()
