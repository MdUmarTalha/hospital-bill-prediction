import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(page_title="Hospital Bill Predictor", page_icon="🏥")

# --- Load the Model ---
@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_model():
    return joblib.load("best_hospital_bill_model.joblib")

try:
    model = load_model()
except:
    st.error("Model file not found! Please run your training script first.")
    st.stop()

# --- UI Layout ---
st.title(" Medical Cost Predictor")
st.markdown("Enter the patient's details below to estimate their annual hospital charges.")

# Create two columns for a cleaner look
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.slider("Number of Children", 0, 10, 0)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Is the patient a smoker?", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# --- Prediction Logic ---
if st.button("Predict Medical Charges", type="primary"):
    # Create a DataFrame for the model
    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }])

    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.divider()
    st.subheader(f"Estimated Bill: :blue[${prediction:,.2f}]")
    
    # Optional: Display a helpful tip based on the prediction
    if smoker == "yes":
        st.warning("Note: Smoking status significantly increases the predicted medical costs.")