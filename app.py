import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# Load the trained model (make sure 'model.pkl' is in the same directory or provide the full path)
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://media.licdn.com/dms/image/D4D12AQE5tMSbHzL5Yw/article-cover_image-shrink_600_2000/0/1686909157848?e=2147483647&v=beta&t=QAErDuhf8HDQ3ojR9z9ajoGCCf3hpiF4wS-dQrsEc3Q');
        background-size: cover;
        color: white;
    }
    h1 {
        text-align: center;
        font-size: 2.5em;
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .input-container {
        margin: 20px;
        padding: 20px;
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
    }
    .button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("Credit Default Prediction")

# Input containers
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Inputs from user
    income = st.number_input("Income", min_value=0, max_value=1000000, value=10000)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    loan = st.number_input("Loan", min_value=0, max_value=10000000, value=100000)

    # Feature Engineering: Loan to Income ratio
    loan_to_income = loan / income if income != 0 else 0

    # Dataframe for prediction
    input_data = pd.DataFrame([[income, age, loan, loan_to_income]], columns=['Income', 'Age', 'Loan', 'Loan to Income'])

    # Prediction button
    if st.button("Predict Default", key="predict", help="Click to predict whether there will be a default!"):
        with st.spinner("Making a prediction..."):
            time.sleep(2)  # Simulate a delay for loading
            
            # Make the prediction
            prediction = model.predict(input_data)

            # Output
            if prediction == 0:
                st.success("Prediction: **No Default**")
            else:
                st.error("Prediction: **Default**")

    st.markdown('</div>', unsafe_allow_html=True)
