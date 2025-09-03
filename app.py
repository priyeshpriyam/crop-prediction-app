# --- Streamlit Web App ---

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Load Models and Data ---
# Load the Crop Recommendation model
recommendation_model = joblib.load('crop_recommendation_model.pkl')

# Load the Yield Prediction model
yield_model = joblib.load('yield_prediction_model.pkl')

# Load the raw data to get mappings for yield prediction inputs
# Make sure you have 'crop_production.csv' in the same folder
yield_df_raw = pd.read_csv('crop_production.csv')

# Create dictionaries to map text to the encoded numbers
# This is crucial for the yield prediction model's input
state_map = {state: i for i, state in enumerate(yield_df_raw['State_Name'].astype('category').cat.categories)}
season_map = {season: i for i, season in enumerate(yield_df_raw['Season'].astype('category').cat.categories)}
crop_map = {crop: i for i, crop in enumerate(yield_df_raw['Crop'].astype('category').cat.categories)}
# We don't need a district map for this version to keep it simple

# --- Web App Interface ---

st.set_page_config(page_title="AgriGenius", page_icon="🌾", layout="wide")
st.title("🌾 AgriGenius: Crop Recommendation & Yield Prediction")

# --- Recommendation Section ---
st.header("🌱 Crop Recommendation")
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input('Nitrogen (N) Content', min_value=0, max_value=140, value=90)
    P = st.number_input('Phosphorus (P) Content', min_value=5, max_value=145, value=42)
    K = st.number_input('Potassium (K) Content', min_value=5, max_value=205, value=43)
with col2:
    temperature = st.number_input('Temperature (°C)', min_value=8.0, max_value=44.0, value=21.0, format="%.2f")
    humidity = st.number_input('Humidity (%)', min_value=14.0, max_value=100.0, value=82.0, format="%.2f")
    ph = st.number_input('pH of Soil', min_value=3.5, max_value=10.0, value=6.5, format="%.2f")
with col3:
    rainfall = st.number_input('Rainfall (mm)', min_value=20.0, max_value=300.0, value=203.0, format="%.2f")

if st.button('Recommend Crop'):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = recommendation_model.predict(input_data)
    st.success(f"The recommended crop for these conditions is: **{prediction[0]}**")

st.markdown("---")

# --- Yield Prediction Section ---
st.header("📈 Crop Yield Prediction")
col4, col5, col6 = st.columns(3)
with col4:
    state_yield = st.selectbox('Select State', options=sorted(list(state_map.keys())))
    crop_yield = st.selectbox('Select Crop', options=sorted(list(crop_map.keys())))
with col5:
    season_yield = st.selectbox('Select Season', options=sorted(list(season_map.keys())))
    year_yield = st.number_input('Enter Year', min_value=1997, max_value=2025, value=2024)
with col6:
    # We need a placeholder for the district encoding. We'll use a common value like 0.
    # A more advanced version could have a district dropdown dependent on the state.
    district_yield_encoded = 0 

if st.button('Predict Yield'):
    # Get the encoded values from the dictionaries
    state_yield_encoded = state_map[state_yield]
    crop_yield_encoded = crop_map[crop_yield]
    season_yield_encoded = season_map[season_yield]
    
    # Prepare the input for the model
    # The order must match the training data: State, District, Year, Season, Crop
    input_data_yield = np.array([[state_yield_encoded, district_yield_encoded, year_yield, season_yield_encoded, crop_yield_encoded]])
    
    # Make the prediction
    prediction_yield = yield_model.predict(input_data_yield)
    
    # Display the result
    st.success(f"The predicted yield is approximately **{prediction_yield[0]:.2f} tons per hectare**.")