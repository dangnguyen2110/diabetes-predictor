import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="centered"
)

# Load model and scaler
@st.cache_resource
def load_models():
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

try:
    model, scaler = load_models()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Title and description
st.title("🏥 Diabetes Risk Prediction")
st.markdown("""
This application predicts the likelihood of diabetes based on diagnostic measurements.
Please enter your health metrics below.
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120, step=1)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)

with col2:
    insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=900, value=80, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

# Predict button
if st.button("🔍 Predict Diabetes Risk", type="primary"):
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, dpf, age]])
    
    # Standardize the input
    standardized_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(standardized_input)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error("⚠️ **High Risk**: The model indicates a high likelihood of diabetes. Bro cần phải đi gym nhiều hơn!!!")
    else:
        st.success("✅ **Low Risk**: The model indicates a low likelihood of diabetes. Ní này còn cú được yên tâm nha")
    
    # Display input summary
    with st.expander("📊 View Input Summary"):
        input_df = pd.DataFrame({
            'Feature': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                       'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'],
            'Value': [pregnancies, glucose, blood_pressure, skin_thickness, 
                     insulin, bmi, dpf, age]
        })
        st.dataframe(input_df, use_container_width=True)
    
    # Medical disclaimer
    st.warning("""
    **⚠️ Medical Disclaimer**: This tool is for educational purposes only and should NOT be used 
    as a substitute for professional medical advice, diagnosis, or treatment. Always consult 
    with a qualified healthcare provider for medical concerns. Made by anh Đăng đẹp trai
    """)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This application uses a Support Vector Machine (SVM) classifier to predict diabetes risk 
    based on the Pima Indians Diabetes Database features.
    
    **Input Features:**
    - Pregnancies
    - Glucose Level
    - Blood Pressure
    - Skin Thickness
    - Insulin Level
    - BMI (Body Mass Index)
    - Diabetes Pedigree Function
    - Age
    """)
    
    st.header("Normal Ranges")
    st.markdown("""
    - **Glucose**: 70-125 mg/dL (fasting)
    - **Blood Pressure**: 80-120 mm Hg
    - **BMI**: 18.5-24.9 (normal weight)
    - **Insulin**: 16-166 μU/mL (fasting)
    """)
