import streamlit as st
import joblib
import numpy as np
import os

# Pastikan joblib sudah terinstall
try:
    import joblib
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "joblib"])
    import joblib

# Cek apakah file model ada
model_path = "iris_model.pkl"
if not os.path.exists(model_path):
    st.error(f"File {model_path} tidak ditemukan. Pastikan sudah diunggah ke repo.")
else:
    # Load model
    model = joblib.load(model_path)

def main():
    st.title("Machine Learning Model Deployment")

    # Input untuk fitur model
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.5)
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.3)

    if st.button("Make Prediction"):
        features = [sepal_length, sepal_width, petal_length, petal_width]
        result = make_prediction(features)
        st.success(f"The prediction is: {result}")

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == "__main__":
    main()
