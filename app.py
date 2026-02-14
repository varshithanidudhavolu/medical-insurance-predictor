import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Medical Insurance Predictor", page_icon="🏥", layout="centered")

# 2. Load Model and Scaler
try:
    model = pickle.load(open('insurance_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'insurance_model.pkl' or 'scaler.pkl' not found. Please check file paths.")

# 3. Custom CSS for Professional Look & Glowing Result
st.markdown("""
    <style>
    /* Background setup */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)), 
                    url("https://www.hdfcergo.com/images/default-source/health-insurance/health-insurance-terminologies.jpg");
        background-size: cover;
    }

    /* Labels styling */
    label {
        font-size: 20px !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
    }

    /* Input Box focus effect */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px !important;
    }

    /* BUTTON STYLING */
    div.stButton > button {
        background: linear-gradient(45deg, #1565c0, #1e88e5);
        color: white;
        font-size: 20px;
        font-weight: bold;
        height: 3em;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(21, 101, 192, 0.4);
        background: linear-gradient(45deg, #1e88e5, #1565c0);
        color: white;
    }

    /* GLOWING RESULT BOX EFFECT */
    .result-container {
        background-color: white;
        border-radius: 15px;
        padding: 30px;
        margin-top: 25px;
        text-align: center;
        border: 2px solid #2e7d32;
        /* Glowing Animation */
        box-shadow: 0 0 15px rgba(46, 125, 50, 0.2);
        animation: glow 1.5s infinite alternate;
    }

    @keyframes glow {
        from { box-shadow: 0 0 10px rgba(46, 125, 50, 0.3); }
        to { box-shadow: 0 0 25px rgba(46, 125, 50, 0.6); }
    }

    .footer-text {
        font-size: 22px !important;
        font-weight: bold !important;
        color: #0d47a1 !important;
        text-align: center;
        margin-top: 50px;
        padding: 10px;
        background: rgba(255,255,255,0.6);
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 4. App Header
st.markdown("<h1 style='text-align: center; color: #0d47a1;'>🏥 Medical Insurance Cost Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter details accurately to get the most precise estimation.</p>", unsafe_allow_html=True)
st.markdown("---")

# 5. Input Fields
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=24.5)
        children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
        
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])     smoker = st.selectbox("Smoker", ["No", "Yes"])
        region = st.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

# 6. Feature Mapping
sex_val = 0 if sex == "Male" else 1
smoker_val = 1 if smoker == "Yes" else 0
region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
region_val = region_map[region]

# 7. Prediction Logic
st.write("") # Spacer
if st.button("Calculate Insurance Charges", use_container_width=True):
    try:
        cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        input_data = pd.DataFrame([[age, sex_val, bmi, children, smoker_val, region_val]], columns=cols)
        
        # Scaling and Prediction
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        final_output = max(0, prediction[0])
        
        # Result Display with Glowing Effect
        st.markdown(f"""
            <div class="result-container">
                <p style='color: #1565c0; font-size: 22px; font-weight: bold; margin-bottom: 5px;'>Estimated Annual Premium</p>
                <h1 style='color: #2e7d32; font-size: 45px; margin: 0;'>${final_output:,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Add success animation
        st.balloons()
        
    except Exception as e:
        st.error(f"Prediction logic lo issue undi: {e}")

# 8. Footer
st.markdown(f"<div class='footer-text'>Project by: Varshitha</div>", unsafe_allow_html=True)