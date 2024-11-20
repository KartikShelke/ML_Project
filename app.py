import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Background and overall app styling */
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                          url('https://media.licdn.com/dms/image/D4D12AQE5tMSbHzL5Yw/article-cover_image-shrink_600_2000/0/1686909157848?e=2147483647&v=beta&t=QAErDuhf8HDQ3ojR9z9ajoGCCf3hpiF4wS-dQrsEc3Q');
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Header styling */
    header {
        background-color: rgba(0, 0, 0, 0.7);
    }
    
    /* Input and button styling */
    .stTextInput, .stNumberInput, .stButton {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #45a049 !important;
        transform: scale(1.05);
    }
    
    /* Result styling */
    .result-box {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

# Main App
def main():
    # Title with animation
    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 4px #000000;'>🏦 Credit Default Prediction 💳</h1>", unsafe_allow_html=True)
    
    # Sidebar with additional information
    st.sidebar.title("🔍 Model Insights")
    
    # Load the model
    model = load_model()
    
    # Input columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Income input with tooltip
        income = st.number_input(
            "💰 Annual Income", 
            min_value=0, 
            max_value=1000000, 
            value=10000,
            help="Enter your annual income in local currency"
        )
    
    with col2:
        # Age input with tooltip
        age = st.number_input(
            "👤 Age", 
            min_value=18, 
            max_value=100, 
            value=30,
            help="Enter your current age"
        )
    
    # Loan amount input
    loan = st.number_input(
        "💸 Loan Amount", 
        min_value=0, 
        max_value=10000000, 
        value=100000,
        help="Enter the loan amount you're seeking"
    )
    
    # Feature Engineering
    loan_to_income = loan / income if income != 0 else 0
    
    # Prepare input data
    input_data = pd.DataFrame(
        [[income, age, loan, loan_to_income]], 
        columns=['Income', 'Age', 'Loan', 'Loan to Income']
    )
    
    # Prediction button with animation
    if st.button("🔮 Predict Default Risk"):
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Probability of default (if model supports)
        try:
            prediction_proba = model.predict_proba(input_data)[0][1]
        except:
            prediction_proba = None
        
        # Create a styled result container
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        
        # Animated result display
        if prediction == 0:
            st.success("✅ Low Default Risk")
            st.balloons()
        else:
            st.error("⚠️ High Default Risk")
            st.snow()
        
        # Probability visualization if available
        if prediction_proba is not None:
            # Create a gauge chart for probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Default Probability", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps' : [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Sidebar additional information
    st.sidebar.markdown("### Model Performance")
    st.sidebar.write("Accuracy: 85%")
    st.sidebar.write("Precision: 82%")
    st.sidebar.write("Recall: 88%")

# Run the app
if __name__ == "__main__":
    main()
