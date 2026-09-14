import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("social_ads_model.keras", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

st.title("Social Network Ads - Purchase Predictor")
st.write("Enter customer demographic details to predict likelihood of purchase.")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input(
    "Estimated Salary ($)",
    min_value=10000,
    max_value=200000,
    value=50000,
    step=1000
)

if st.button("Predict"):
    gender_encoded = 0 if gender == "Male" else 1

    input_data = np.array([[gender_encoded, age, salary]])
    scaled_input = scaler.transform(input_data)

    prob = float(model.predict(scaled_input, verbose=0)[0][0])
    prediction = int(prob >= 0.5)

    st.subheader("Result")

    if prediction == 1:
        st.success(f"Purchased (Probability: {prob:.2%})")
    else:
        st.error(f"Not Purchased (Probability: {prob:.2%})")
