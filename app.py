import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model (ensure 'model.pkl' is in the same directory or provide the full path)
model = pickle.load(open('model.pkl', 'rb'))

# Load external CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
local_css("styles.css")
# Add the blurred background

# Add the background div
st.markdown(
    """
    <style>
    .blur-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        background-image: url('./background.jpg'); /* Path to your image */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        filter: blur(8px); /* Apply blur only to the background */
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
