import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model (make sure 'model.pkl' is in the same directory or provide the full path)
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App
st.title("Credit Default Prediction")

# Inputs from user
income = st.number_input("Income", min_value=0, max_value=1000000, value=10000)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
loan = st.number_input("Loan", min_value=0, max_value=10000000, value=100000)

# Feature Engineering: Loan to Income ratio (assuming you want to include this as an engineered feature)
loan_to_income = loan / income if income != 0 else 0

# Dataframe for prediction
# Ensure column names are consistent with what the model expects
input_data = pd.DataFrame([[income, age, loan, loan_to_income]], columns=['Income', 'Age', 'Loan', 'Loan to Income'])


# Prediction button
if st.button("Predict Default"):
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Output
    if prediction == 0:
        st.write("Prediction: **No Default**")
    else:
        st.write("Prediction: **Default**")
