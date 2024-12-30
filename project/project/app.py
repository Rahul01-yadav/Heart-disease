import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the pre-trained model
def load_model():
    return joblib.load("C:/Users/DELL/Downloads/project/project/rf_model.pkl")

# Feature columns for input
feature_columns = [
    "age", "gender", "chest pain", "resting blood pressure", "cholestrol", "fasting blood sugar", "electrocardiographic",
    "max. heart rate", "exang", "oldpeak", "slope", "coronary artery", "thalassemia"
]

# Application code
def main():
    st.title("Heart Disease Prediction")
    st.write("Input your health data to predict the likelihood of heart disease.")

    # Collect user input
    # inputs = {}
    # for col in feature_columns:
    #     if col in ["gender", "chest pain", "fasting blood sugar", "electrocardiographic", "exang", "slope", "coronary artery", "thalassemia"]:
    #         # Discrete features
    #         inputs[col] = st.selectbox(f"{col.capitalize()}", [0, 1, 2, 3], index=0)
    #     else:
    #         # Continuous features
    #         inputs[col] = st.slider(f"{col.capitalize()}", 0, 300, 100)
    # Collect user input
    inputs = {}
    for col in feature_columns:
        if col in ["gender", "chest pain", "fasting blood sugar", "electrocardiographic", "exang", "slope", "coronary artery", "thalassemia"]:
            # Discrete features
            if col == "chest pain":
                inputs[col] = st.selectbox(f"{col.capitalize()} (cp type)", [0, 1, 2, 3], index=0)
            elif col == "thalassemia":
                inputs[col] = st.selectbox(f"{col.capitalize()} (Thal)", [0, 1, 2, 3], index=0)
            elif col == "coronary artery":
                inputs[col] = st.selectbox(f"{col.capitalize()} (Number of vessels)", [0, 1, 2, 3], index=0)
            else:
                inputs[col] = st.selectbox(f"{col.capitalize()}", [0, 1], index=0)
        else:
            # Continuous features
            if col in ["age", "resting blood pressure", "cholestrol", "max. heart rate"]:
                inputs[col] = st.slider(f"{col.capitalize()}", min_value=0, max_value=300, value=100)
            elif col == "oldpeak":
                inputs[col] = st.slider(f"{col.capitalize()} (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    user_data = pd.DataFrame([inputs])

    # Load the model and make prediction
    rf_model = load_model()
    prediction = rf_model.predict(user_data)
    prediction_proba = rf_model.predict_proba(user_data)

    # Display results
    if st.button("Predict"):
        if prediction[0] == 1:
            st.success("The model predicts you have heart disease.")
        else:
            st.success("The model predicts you do NOT have heart disease.")

        st.write(f"Prediction confidence: {prediction_proba[0][prediction[0]] * 100:.2f}%")

if __name__ == "__main__":
    main()
