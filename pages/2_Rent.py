import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import pickle


@st.cache_data
def predict(area_type, city, furn_status, tenant_preferred,
            point_of_contact, bhk, size, bathroom,
            floor, number_of_floors):
    model = CatBoostRegressor()
    model.load_model("./models/cb_regressor_model")
    with open("./models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    area_type += " Area"
    point_of_contact = "Contact " + point_of_contact
    numeric_data = scaler.transform(
        np.array([[bhk, size, bathroom, floor/number_of_floors]]))
    data = {"Area Type": [area_type],
            "City": [city],
            "Furnishing Status": [furn_status],
            "Tenant Preferred": [tenant_preferred],
            "Point of Contact": [point_of_contact],
            "BHK": [numeric_data[0, 0]],
            "Size": [numeric_data[0, 1]],
            "Bathroom": [numeric_data[0, 2]],
            "Floor": [numeric_data[0, 3]]}
    input_data = Pool(pd.DataFrame(data),
                      cat_features=['Area Type',
                                    'City',
                                    'Furnishing Status',
                                    'Tenant Preferred',
                                    'Point of Contact'])
    y_pred = model.predict(input_data)
    return round(y_pred[0])


def main():
    st.title("Predict rent price")
    st.subheader("Input information about apartment")

    area_type = st.selectbox("Pick area type",
                             ["Super", "Carpet", "Built"])

    city = st.selectbox("Choose a city",
                        ["Kolkata", "Mumbai", "Bangalore",
                         "Delhi", "Chennai", "Hyderabad"])

    furn_status = st.selectbox("Furnishing status",
                               ["Unfurnished", "Semi-Furnished", "Furnished"])

    tenant_preferred = st.selectbox("What type of tenant owner prefers?",
                                    ["Bachelors", "Family",
                                     "Bachelors/Family"])

    point_of_contact = st.selectbox(
        "Who should you contact for more information about apartment?",
        ["Owner", "Agent", "Builder"])

    bhk = st.slider("Pick number of Bedrooms, Hall, Kitchen", 1, 10)
    size = st.number_input("Input size of apartment in Square Feet", 10, 8000)
    bathroom = st.slider("Choose number of bathrooms", 1, 10)
    floor = st.number_input("On which fool is the apartment located?", 1, 100)
    number_of_floors = st.number_input(
        "How many floors are there in the building?", 1, 100)
    prediction = predict(area_type, city, furn_status, tenant_preferred,
                         point_of_contact, bhk, size, bathroom,
                         floor, number_of_floors)

    st.subheader(f"Rent price: **{prediction}**$")


main()
