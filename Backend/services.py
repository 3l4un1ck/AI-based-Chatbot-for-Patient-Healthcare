import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load the model
@st.cache_resource
def load_model():
    with open("app/decision_tree_classifier.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# Load symptom data (features)
training_data = pd.read_csv("Data/Training.csv")
symptom_list = training_data.columns[:-1]  # Get symptom names
prognosis_list = training_data['prognosis'].unique()  # Possible diagnoses

# Streamlit app layout
st.title("Disease Prediction ChatBot")
st.write("Interact with the AI to predict possible diseases based on symptoms.")

# User inputs for symptoms
st.subheader("Select the symptoms you are experiencing:")
selected_symptoms = st.multiselect("Symptoms:", symptom_list)

# Predict button
if st.button("Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Prepare input vector for the model
        input_vector = np.zeros(len(symptom_list))
        for symptom in selected_symptoms:
            index = list(symptom_list).index(symptom)
            input_vector[index] = 1

        # Make a prediction
        prediction = model.predict([input_vector])[0]
        st.success(f"Prediction: {prediction}")

        # Display possible treatments or precautions
        st.subheader("Recommended Precautions:")
        precautions = {
            # Add your symptom-to-precaution mappings here
            "Disease1": ["Take rest", "Stay hydrated", "Consult a doctor"],
            "Disease2": ["Avoid allergens", "Use prescribed medication", "Monitor symptoms"],
        }
        if prediction in precautions:
            for i, precaution in enumerate(precautions[prediction], 1):
                st.write(f"{i}. {precaution}")
        else:
            st.write("No specific precautions available.")
