
"""
Streamlit App for ML Model Deployment
=====================================

Restaurant Revenue Forecasting & Classification App

Author: Samuel Reid
Dataset: Restaurant_data.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go


# =============================================================================
# PAGE CONFIGURATION
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
# SAFE FEATURE ALIGNMENT (FIXES KEYERROR)
# =============================================================================

def align_features(input_dict, feature_list):

    aligned = {}

    for feature in feature_list:

        if feature in input_dict:
            aligned[feature] = input_dict[feature]
        else:
            aligned[feature] = 0

    return pd.DataFrame([aligned])


# =============================================================================
# PREDICTIONS
# =============================================================================

def make_regression_prediction(models, input_dict):

    df = align_features(input_dict, models['regression_features'])

    scaled = models['regression_scaler'].transform(df)

    prediction = models['regression_model'].predict(scaled)[0]

    return prediction


def make_classification_prediction(models, input_dict):

    df = align_features(input_dict, models['classification_features'])

    scaled = models['classification_scaler'].transform(df)

    prediction = models['classification_model'].predict(scaled)

    label = models['label_encoder'].inverse_transform(prediction)[0]

    return label


# =============================================================================
# GAUGE CHART
# =============================================================================

def revenue_gauge(prediction):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Predicted Annual Revenue"},
        gauge={
            'axis': {'range': [0, max(500000, prediction*1.2)]},

            'bar': {'color': "green"},

            'steps': [
                {'range': [0, 100000], 'color': "red"},
                {'range': [100000, 300000], 'color': "orange"},
                {'range': [300000, 500000], 'color': "yellow"},
                {'range': [500000, prediction*1.2], 'color': "green"},
            ],
        }
    ))

    fig.update_layout(height=400)

    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a model:",
    ["🏠 Home", "📈 Regression Model", "🏷️ Classification Model"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### About")

st.sidebar.info(
"""
This app deploys machine learning models trained on Restaurant_data.csv.

- **Regression**: Predicts Annual Restaurant Revenue  
- **Classification**: Predicts Revenue Category  
"""
)

st.sidebar.markdown("**Built by:** Samuel Reid")


# =============================================================================
# HOME PAGE
# =============================================================================

if page == "🏠 Home":

    st.title("🤖 Restaurant Revenue Forecasting & Classification App")

    st.markdown("### Welcome!")

    st.write(
"""
Use the sidebar to select a model.

📈 Regression → Predict revenue  
🏷️ Classification → Predict revenue category
"""
)


# =============================================================================
# REGRESSION PAGE
# =============================================================================

elif page == "📈 Regression Model":

    st.title("📈 Regression Prediction")

    st.write("Enter feature values to get a numerical prediction.")

    models = load_models()

    st.markdown("---")

    st.markdown("### Enter Feature Values")


    region_options = ["Urban", "Suburban", "Rural"]

    cuisine_options = ["Japanese", "Mexican", "French", "Indian", "Italian"]


    rows = [st.columns(2) for _ in range(3)]

    input_dict = {}


    with rows[0][0]:

        input_dict["Average Meal Price"] = st.slider(
            "Average Meal Price",
            5.0, 150.0, 25.0
        )

    with rows[0][1]:

        input_dict["Seating Capacity"] = st.slider(
            "Seating Capacity",
            10, 300, 50
        )


    with rows[1][0]:

        region = st.selectbox("Region", region_options)

        input_dict[f"Location_{region}"] = 1


    with rows[1][1]:

        cuisine = st.selectbox("Cuisine", cuisine_options)

        input_dict[f"Cuisine_{cuisine}"] = 1


    with rows[2][0]:

        input_dict["Total Reservations"] = st.slider(
            "Total Reservations",
            0, 1000, 200
        )


    with rows[2][1]:

        input_dict["Rating"] = st.slider(
            "Rating",
            1.0, 5.0, 4.0
        )


    st.markdown("---")


    if st.button("🔮 Make Regression Prediction", type="primary"):

        prediction = make_regression_prediction(models, input_dict)

        st.success(f"### Predicted Value: ${prediction:,.0f}")

        st.plotly_chart(revenue_gauge(prediction), use_container_width=True)

        with st.expander("View Input Summary"):

            st.dataframe(pd.DataFrame([input_dict]))


# =============================================================================
# CLASSIFICATION PAGE
# =============================================================================

elif page == "🏷️ Classification Model":

    st.title("🏷️ Classification Prediction")

    st.write("Enter feature values to get a category prediction.")

    models = load_models()

    st.markdown("---")

    st.markdown("### Enter Feature Values")


    region_options = ["Urban", "Suburban", "Rural"]

    cuisine_options = [
        "American",
        "Italian",
        "Japanese",
        "Mexican",
        "French",
        "Indian",
        "Chinese"
    ]


    rows = [st.columns(2) for _ in range(3)]

    input_dict = {}


    with rows[0][0]:

        input_dict["Average Meal Price"] = st.slider(
            "Average Meal Price",
            5.0, 150.0, 25.0
        )

    with rows[0][1]:

        input_dict["Seating Capacity"] = st.slider(
            "Seating Capacity",
            10, 300, 50
        )


    with rows[1][0]:

        region = st.selectbox("Region", region_options, key="class_region")

        input_dict[f"Location_{region}"] = 1


    with rows[1][1]:

        cuisine = st.selectbox("Cuisine", cuisine_options, key="class_cuisine")

        input_dict[f"Cuisine_{cuisine}"] = 1


    with rows[2][0]:

        input_dict["Total Reservations"] = st.slider(
            "Total Reservations",
            0, 1000, 200
        )


    with rows[2][1]:

        input_dict["Rating"] = st.slider(
            "Rating",
            1.0, 5.0, 4.0
        )


    st.markdown("---")


    if st.button("🔮 Make Classification Prediction", type="primary"):

        label = make_classification_prediction(models, input_dict)

        emoji_map = {
            "Low": "🔴",
            "Medium Low": "🟠",
            "Medium High": "🟡",
            "High": "🟢"
        }

        emoji = emoji_map.get(label, "🔵")

        st.success(f"### Predicted Category: {emoji} {label}")

        with st.expander("View Input Summary"):

            st.dataframe(pd.DataFrame([input_dict]))


# =============================================================================
# FOOTER
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