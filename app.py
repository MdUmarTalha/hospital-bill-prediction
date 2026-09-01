import os
import subprocess
import sys
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hospital Bill Predictor")

MODEL_PATH = "best_hospital_bill_model.joblib"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        # Runs model.py using the active Python environment
        result = subprocess.run(
            [sys.executable, "model.py"], capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Model training failed:\n{result.stderr}")

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load or train model: {e}")
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
    
    st.divider()
    st.subheader(f"Estimated Bill: :blue[${prediction:,.2f}]")
    
    if smoker == "yes":
        st.warning("Note: Smoking status significantly increases the predicted medical costs.")
