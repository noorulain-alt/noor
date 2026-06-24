import streamlit as st
import pickle
import numpy as np

# Load model
with open("heart disease model .pkl", "rb") as file:
   heart disease model = pickle.load(file)

st.title("Machine Learning Prediction App")

st.write("Enter feature values below:")

# Example inputs (change according to your model)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

if st.button("Predict"):

    input_data = np.array(
        [[feature1, feature2, feature3, feature4]]
    )

    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")
