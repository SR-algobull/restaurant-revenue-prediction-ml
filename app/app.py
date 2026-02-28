"""
Streamlit App for ML Model Deployment
=====================================
(UNCHANGED HEADER — UI preserved)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =============================================================================
# PAGE CONFIGURATION (UNCHANGED)
# =============================================================================
st.set_page_config(
    page_title="Restaurant Revenue Forecasting & Classification App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
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

    return models


# =============================================================================
# SAFE FEATURE ALIGNMENT (FIXES YOUR KEYERROR)
# =============================================================================

def align_features(input_dict, feature_list):
    """
    Converts dropdown selections into correct one-hot columns
    and ensures dataframe EXACTLY matches model features.
    """

    df = pd.DataFrame([input_dict])

    # initialize all required columns to 0
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    # return correctly ordered dataframe
    return df[feature_list]


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def make_regression_prediction(models, input_dict):

    df = align_features(input_dict, models['regression_features'])

    scaled = models['regression_scaler'].transform(df)

    pred = models['regression_model'].predict(scaled)

    return pred[0]


def make_classification_prediction(models, input_dict):

    df = align_features(input_dict, models['classification_features'])

    scaled = models['classification_scaler'].transform(df)

    pred = models['classification_model'].predict(scaled)

    label = models['label_encoder'].inverse_transform(pred)

    return label[0]


# =============================================================================
# SIDEBAR (UNCHANGED)
# =============================================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a model:",
    ["🏠 Home", "📈 Regression Model", "🏷️ Classification Model"]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
This app deploys machine learning models trained on Restaurant_data.csv.

- Regression: Predicts Annual Restaurant Revenue
- Classification: Predicts Revenue Category
"""
)

st.sidebar.markdown("**Built by:** Samuel Reid")


# =============================================================================
# HOME PAGE (UNCHANGED)
# =============================================================================

if page == "🏠 Home":

    st.title("🤖 Restaurant Revenue Forecasting & Classification App")

    st.markdown("### Welcome!")

    st.write("Use the sidebar to navigate.")


# =============================================================================
# REGRESSION PAGE (3×3 GRID FIX)
# =============================================================================

elif page == "📈 Regression Model":

    st.title("📈 Regression Prediction")

    models = load_models()

    features = models['regression_features']


    st.markdown("### Enter Feature Values")


    # create 3 rows × 2 columns
    rows = [st.columns(2) for _ in range(3)]

    input_dict = {}

    for i, feature in enumerate(features):

        row = i // 2
        col = i % 2

        with rows[row][col]:

            input_dict[feature] = st.number_input(
                feature,
                value=0.0
            )


    st.markdown("---")

    if st.button("🔮 Make Regression Prediction", type="primary"):

        prediction = make_regression_prediction(models, input_dict)

        st.success(f"### Predicted Value: {prediction:,.2f}")

        with st.expander("View Input Summary"):
            st.write(input_dict)



# =============================================================================
# CLASSIFICATION PAGE (DROPDOWNS + FIXED)
# =============================================================================

elif page == "🏷️ Classification Model":

    st.title("🏷️ Classification Prediction")

    models = load_models()

    features = models['classification_features']


    cuisine_options = [
        "Japanese",
        "Mexican",
        "French",
        "Indian",
        "Italian"
    ]

    region_options = [
        "Urban",
        "Suburban",
        "Rural"
    ]


    rows = [st.columns(2) for _ in range(3)]

    input_dict = {}


    # Row 1
    with rows[0][0]:
        input_dict["Average Meal Price"] = st.number_input("Average Meal Price", value=25.0)

    with rows[0][1]:
        input_dict["Seating Capacity"] = st.number_input("Seating Capacity", value=50)


    # Row 2
    with rows[1][0]:

        region = st.selectbox("Region", region_options)

        input_dict[f"Location_{region}"] = 1


    with rows[1][1]:

        cuisine = st.selectbox("Cuisine", cuisine_options)

        input_dict[f"Cuisine_{cuisine}"] = 1


    # Row 3
    with rows[2][0]:

        input_dict["Total Reservations"] = st.number_input("Total Reservations", value=100)


    with rows[2][1]:

        input_dict["Rating"] = st.number_input("Rating", value=4.0)



    st.markdown("---")


    if st.button("🔮 Make Classification Prediction", type="primary"):

        label = make_classification_prediction(models, input_dict)


        color_map = {
            'Low': '🔴',
            'Medium Low': '🟠',
            'Medium High': '🟡',
            'High': '🟢'
        }

        emoji = color_map.get(label, '🔵')

        st.success(f"### Predicted Category: {emoji} {label}")

        with st.expander("View Input Summary"):
            st.write(input_dict)



# =============================================================================
# FOOTER (UNCHANGED)
# =============================================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align: center; color: gray;'>
Built by Samuel Reid | Full Stack Academy AI & ML Bootcamp
</div>
""",
unsafe_allow_html=True
)