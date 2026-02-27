"""
Streamlit App for ML Model Deployment
Restaurant Revenue Prediction & Classification
Author: Samuel Reid
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Restaurant Revenue Prediction App",
    page_icon="🍽️",
    layout="wide"
)


# =============================================================================
# LOAD MODELS
# =============================================================================

@st.cache_resource
def load_models():

    base_path = Path(__file__).parent.parent / "models"

    models = {}

    models['regression_model'] = joblib.load(base_path / "regression_model.pkl")
    models['regression_scaler'] = joblib.load(base_path / "regression_scaler.pkl")
    models['regression_features'] = joblib.load(base_path / "regression_features.pkl")

    models['classification_model'] = joblib.load(base_path / "classification_model.pkl")
    models['classification_scaler'] = joblib.load(base_path / "classification_scaler.pkl")
    models['classification_features'] = joblib.load(base_path / "classification_features.pkl")
    models['label_encoder'] = joblib.load(base_path / "label_encoder.pkl")

    try:
        models['binning_info'] = joblib.load(base_path / "binning_info.pkl")
    except:
        models['binning_info'] = None

    return models


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def make_regression_prediction(models, input_df):

    input_df = input_df[models['regression_features']]
    scaled = models['regression_scaler'].transform(input_df)

    prediction = models['regression_model'].predict(scaled)

    return prediction[0]


def make_classification_prediction(models, input_df):

    input_df = input_df[models['classification_features']]
    scaled = models['classification_scaler'].transform(input_df)

    pred = models['classification_model'].predict(scaled)

    label = models['label_encoder'].inverse_transform(pred)

    return label[0]


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Model",

    [
        "Home",
        "Regression",
        "Classification"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by Samuel Reid")


# =============================================================================
# HOME
# =============================================================================

if page == "Home":

    st.title("Restaurant Revenue Prediction")

    st.write("""

    This app predicts:

    • Annual Revenue (Regression)

    • Revenue Category (Classification)

    """)


# =============================================================================
# REGRESSION PAGE
# =============================================================================

elif page == "Regression":

    st.title("Regression Prediction")
    st.write("Predict restaurant revenue")


    models = load_models()


    st.subheader("Enter Features")


    col1, col2 = st.columns(2)

    input_values = {}


    # REQUIRED FEATURES

    with col1:

        input_values["Average Meal Price"] = st.number_input(

            "Average Meal Price ($)",
            min_value=0.0,
            value=25.0
        )

        input_values["Seating Capacity"] = st.number_input(

            "Seating Capacity",
            min_value=0,
            value=50
        )


        input_values["Total Reservations"] = st.number_input(

            "Total Reservations",
            min_value=0,
            value=200
        )


    with col2:

        input_values["Marketing Budget"] = st.number_input(

            "Marketing Budget",
            min_value=0.0,
            value=1000.0
        )


        input_values["Social Media Followers"] = st.number_input(

            "Social Media Followers",
            min_value=0,
            value=5000
        )


        input_values["Rating"] = st.slider(

            "Rating",
            0.0,
            5.0,
            4.0
        )


    if st.button("Predict Revenue"):

        df = pd.DataFrame([input_values])

        prediction = make_regression_prediction(models, df)

        st.success(f"Predicted Revenue: ${prediction:,.2f}")

        st.dataframe(df)



# =============================================================================
# CLASSIFICATION PAGE
# =============================================================================

elif page == "Classification":

    st.title("Revenue Category Classification")


    models = load_models()


    st.subheader("Enter Features")


    col1, col2 = st.columns(2)

    input_values = {}


    # NUMERIC FEATURES

    with col1:

        input_values["Average Meal Price"] = st.number_input(

            "Average Meal Price ($)",
            min_value=0.0,
            value=25.0
        )

        input_values["Seating Capacity"] = st.number_input(

            "Seating Capacity",
            min_value=0,
            value=50
        )


        input_values["Total Reservations"] = st.number_input(

            "Total Reservations",
            min_value=0,
            value=200
        )


    with col2:

        input_values["Rating"] = st.slider(

            "Rating",
            0.0,
            5.0,
            4.0
        )


    # LOCATION DROPDOWN

    location = st.selectbox(

        "Location",

        [

            "Urban",
            "Suburban",
            "Rural"

        ]
    )


    input_values["Location_Urban"] = 1 if location == "Urban" else 0
    input_values["Location_Suburban"] = 1 if location == "Suburban" else 0
    input_values["Location_Rural"] = 1 if location == "Rural" else 0


    # CUISINE DROPDOWN

    cuisine = st.selectbox(

        "Cuisine",

        [

            "American",
            "Italian",
            "Japanese",
            "Mexican",
            "French",
            "Indian",
            "Chinese"

        ]
    )


    input_values["Cuisine_American"] = 1 if cuisine == "American" else 0
    input_values["Cuisine_Italian"] = 1 if cuisine == "Italian" else 0
    input_values["Cuisine_Japanese"] = 1 if cuisine == "Japanese" else 0
    input_values["Cuisine_Mexican"] = 1 if cuisine == "Mexican" else 0
    input_values["Cuisine_French"] = 1 if cuisine == "French" else 0
    input_values["Cuisine_Indian"] = 1 if cuisine == "Indian" else 0
    input_values["Cuisine_Chinese"] = 1 if cuisine == "Chinese" else 0


    if st.button("Predict Category"):

        df = pd.DataFrame([input_values])

        prediction = make_classification_prediction(models, df)


        emoji_map = {

            "Low": "🔴",
            "Medium Low": "🟠",
            "Medium High": "🟡",
            "High": "🟢"

        }

        emoji = emoji_map.get(prediction, "")

        st.success(f"Predicted Category: {emoji} {prediction}")

        st.dataframe(df)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("Samuel Reid | Full Stack Academy AI & ML Bootcamp")