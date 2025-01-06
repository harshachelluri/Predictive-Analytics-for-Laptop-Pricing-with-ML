import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from sklearn.preprocessing import StandardScaler
import os

# Set the page configuration for Streamlit
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide") 

# Load the pre-trained model
model = joblib.load('optimized_laptop_price_model.pkl')

# Function to make predictions
def predict_laptop_price(features):
    prediction = model.predict(features)
    return np.exp(prediction)  # Take the exponential to reverse the log transformation

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Add background image using base64 encoding
image_path = r'laptop.jpg'  # Update path to your image

# Try loading the image and converting it to base64
try:
    image_base64 = image_to_base64(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{image_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.error("Image file not found. Please check the path and try again.")

# Streamlit interface
st.title('Laptop Price Prediction')

st.write("Welcome to the Laptop Price Prediction App. Enter the laptop details below:")

# Inputs for features - these should match the features used during training
company = st.selectbox('Company', ['Dell', 'HP', 'Lenovo', 'Acer', 'Asus', 'Apple'])
type_name = st.selectbox('Type', ['Laptop', 'Ultrabook', 'Gaming Laptop', 'Notebook'])
cpu_brand = st.selectbox('CPU Brand', ['Intel', 'AMD'])
gpu_brand = st.selectbox('GPU Brand', ['NVIDIA', 'AMD', 'Intel'])
os = st.selectbox('Operating System', ['Windows', 'Linux', 'macOS', 'Chrome OS'])
ram = st.number_input('RAM (GB)', min_value=4, max_value=64, value=8)
storage = st.number_input('Storage (GB)', min_value=128, max_value=2048, value=512)

# Middle column for touchscreen input
middle_column, right_column = st.columns(2)

with middle_column:
    # Touchscreen
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

with right_column:
    # IPS display
    ips = st.selectbox("IPS Display", ["No", "Yes"])

# Prepare the features for prediction (you might need to preprocess these same as during training)
input_data = {
    'Company': company,
    'TypeName': type_name,
    'Cpu brand': cpu_brand,
    'Gpu_brand': gpu_brand,
    'os': os,
    'Ram': ram,
    'Storage': storage,
    'Touchscreen': touchscreen,
    'IPS': ips
}

# Convert input_data to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess input data (apply transformations, e.g., one-hot encoding, scaling)
input_df_encoded = pd.get_dummies(input_df, columns=['Company', 'TypeName', 'Cpu brand', 'Gpu_brand', 'os', 'Touchscreen', 'IPS'], drop_first=True)

# Ensure the model receives the same feature columns as during training
missing_cols = set(model.feature_names_in_) - set(input_df_encoded.columns)
for col in missing_cols:
    input_df_encoded[col] = 0  # Add missing columns with value 0

input_df_encoded = input_df_encoded[model.feature_names_in_]  # Reorder columns to match model input order

# Prediction logic with a loading spinner
if st.button('Predict Price'):
    with st.spinner('Predicting...'):
        try:
            price = predict_laptop_price(input_df_encoded)
            st.write(f"The predicted price for the laptop is: RS:{price[0]:.2f}")
        except Exception as e:
            st.error(f"Error occurred during prediction: {str(e)}")
