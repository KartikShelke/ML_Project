import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model (ensure 'model.pkl' is in the same directory or provide the full path)
model = pickle.load(open('model.pkl', 'rb'))

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Background container with blur */
    .blur-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        background-image: url('https://www.linkedin.com/pulse/credit-cards-ml-risks-methods-girish-mallya-cams-came');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        filter: blur(8px); /* Apply blur only to the image */
    }
    /* Center the content with a semi-transparent background */
    .main-content {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: auto;
        width: 60%;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTitle {
        color: #333;
        text-align: center;
        font-family: 'Arial', sans-serif;
        animation: fadeIn 2s ease-in-out;
    }
    /* Adding simple animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    </style>
    <div class="blur-background"></div>
    """,
    unsafe_allow_html=True,
)

# Streamlit App
st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.title("Credit Default Prediction")

# Inputs from user
income = st.number_input("Income", min_value=0, max_value=1000000, value=10000)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
loan = st.number_input("Loan", min_value=0, max_value=10000000, value=100000)

# Feature Engineering: Loan to Income ratio (assuming you want to include this as an engineered feature)
loan_to_income = loan / income if income != 0 else 0

# Dataframe for prediction
# Ensure column names are consistent with what the model expects
input_data = pd.DataFrame(
    [[income, age, loan, loan_to_income]], columns=["Income", "Age", "Loan", "Loan to Income"]
)

# Prediction button
if st.button("Predict Default"):
    # Make the prediction
    prediction = model.predict(input_data)

    # Output
    if prediction == 0:
        st.write("Prediction: **No Default**")
    else:
        st.write("Prediction: **Default**")

st.markdown("</div>", unsafe_allow_html=True)
