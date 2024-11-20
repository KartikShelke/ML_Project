import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from streamlit_lottie import st_lottie

# Load Lottie animation from a URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load Lottie animations (you can replace the URLs with other animations)
lottie_celebrate = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_1pxqjqps.json")
lottie_credit_risk = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_sxk2ofwd.json")

# Streamlit App with Custom Styling and Animations
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://media.licdn.com/dms/image/D4D12AQE5tMSbHzL5Yw/article-cover_image-shrink_600_2000/0/1686909157848?e=2147483647&v=beta&t=QAErDuhf8HDQ3ojR9z9ajoGCCf3hpiF4wS-dQrsEc3Q');
        background-size: cover;
        color: white;
    }
    .title {
        font-size: 48px;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #ffd700;
        text-align: center;
    }
    .box-shadow {
        padding: 10px;
        box-shadow: 2px 2px 20px rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        background-color: rgba(0, 0, 0, 0.7);
    }
    </style>
""", unsafe_allow_html=True)

# Title with styling
st.markdown('<h1 class="title">🎉 Credit Default Prediction App 🎉</h1>', unsafe_allow_html=True)

# Lottie animation at the top
st_lottie(lottie_credit_risk, height=200, key="credit_risk")

# Input Form in a container with some shadow effect
st.markdown('<div class="box-shadow">', unsafe_allow_html=True)

# Inputs from user
income = st.number_input("Income", min_value=0, max_value=1000000, value=10000)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
loan = st.number_input("Loan", min_value=0, max_value=10000000, value=100000)

# Feature Engineering: Loan to Income ratio
loan_to_income = loan / income if income != 0 else 0

# Dataframe for prediction
input_data = pd.DataFrame([[income, age, loan, loan_to_income]], columns=['Income', 'Age', 'Loan', 'Loan to Income'])

# Prediction button with Lottie animation for celebration in case of no default
if st.button("Predict Default"):
    # Simulate loading with a progress bar
    with st.spinner('Analyzing...'):
        for i in range(100):
            st.progress(i + 1)
        
    # Make the prediction
    prediction = model.predict(input_data)

    # Output with an accompanying animation
    if prediction == 0:
        st.markdown("### 🎉 Prediction: **No Default** 🎉", unsafe_allow_html=True)
        st_lottie(lottie_celebrate, height=150, key="celebrate")
    else:
        st.markdown("### ❌ Prediction: **Default** ❌", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer Lottie animation at the bottom
st_lottie(
    lottie_credit_risk,
    height=150,
    key="footer_credit_risk"
)
