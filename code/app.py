import streamlit as st
import pickle
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="💼 Salary Prediction App",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Load Trained Model ---
model = pickle.load(open(
    r"C:\Users\DELL\6months-AI-Roadmap.2025\practical\Salary Prediction App using SLR\linear_regression_model.pkl", "rb"
))

# --- Header Section ---
st.title("💼 Salary Prediction App")
st.markdown("___")

# --- User Input ---
years_experience = st.number_input(
    "Enter your experience in years", min_value=0.0, max_value=50.0, value=1.0, step=0.5
)

# --- Predict Button ---
if st.button("🚀 Predict Salary"):
    experience_input = np.array([[years_experience]])
    prediction = model.predict(experience_input)
    st.success(
        f"💰 **Estimated Salary:** ${prediction[0]:,.2f} for {years_experience} years of experience."
    )

# --- Model Info ---
st.markdown("___")
st.markdown(
    """
    🔍 **About this App:**  
    - A simple tool built with **Python**, **scikit-learn**, and **Streamlit** to predict salaries based on years of experience.  
    - The model is trained using **Simple Linear Regression (SLR)** on historical salary data.
    """
)
