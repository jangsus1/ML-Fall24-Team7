import streamlit as st
import pandas as pd
import pickle
from utils.preprocessing import preprocess_data
from utils.predictions import make_prediction

# Load pre-trained model
## TODO: Update model path
# model_path = 'models/model.pkl'
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)

# Title and description
st.title("CS 7641 Machine Learning Project - Fall 2024")
st.write("Upload your data in CSV format for prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Preprocess data
    preprocessed_data = preprocess_data(data)
    
    # Make prediction
    predictions = make_prediction(model, preprocessed_data)
    
    # Display results
    st.write("Predictions:")
    st.write(predictions)
else:
    st.write("Please upload a CSV file.")