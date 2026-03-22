import streamlit as st
import requests
import pandas as pd


API_URL = st.secrets["url"]

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction")
st.markdown("Enter student details to predict their **Math Score**.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["female", "male"])
    race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Education", [
        "some high school", "high school", "some college", 
        "associate's degree", "batchelor's degree", "master's degree"
    ])

with col2:
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Prep Course", ["none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=70)

if st.button("Predict Score 🚀"):
    payload = {
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "parental_level_of_education": parental_level_of_education,
        "lunch": lunch,
        "test_preparation_course": test_preparation_course,
        "reading_score": float(reading_score),
        "writing_score": float(writing_score)
    }

    try:
        with st.spinner('Calculating...'):
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                prediction = response.json().get("prediction")
                st.success(f"### Predicted Math Score: {round(prediction, 2)}")
                st.balloons()
            else:
                st.error(f"Error: {response.json().get('detail')}")
    except Exception as e:
        st.error(f"Could not connect to the API: {e}")
