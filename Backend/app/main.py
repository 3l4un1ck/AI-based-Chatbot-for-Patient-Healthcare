from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Disease Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les origines (à adapter en production)
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Autoriser tous les en-têtes
)

# Load the model
def load_model():
    with open("app/decision_tree_classifier.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# Load symptoms list
training_data = pd.read_csv("app/Data/Training.csv")
symptom_list = training_data.columns[:-1].tolist()  # Symptom names
prognosis_list = training_data['prognosis'].unique().tolist()  # Possible diagnoses


# Pydantic model for input validation
class SymptomInput(BaseModel):
    symptoms: list[str]


# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Disease Prediction API!"}


# Endpoint to get all symptoms
@app.get("/symptoms/")
def get_symptoms():
    return {"symptoms": symptom_list}


# Endpoint to predict disease
@app.post("/predict/")
def predict_disease(input_data: SymptomInput):
    # Validate input
    user_symptoms = input_data.symptoms
    if not user_symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided.")

    # Prepare input vector for the model
    input_vector = np.zeros(len(symptom_list))
    for symptom in user_symptoms:
        if symptom not in symptom_list:
            raise HTTPException(status_code=400, detail=f"Invalid symptom: {symptom}")
        index = symptom_list.index(symptom)
        input_vector[index] = 1

    # Predict disease
    prediction = model.predict([input_vector])[0]

    # avec le rowPredict retrouver la bonne description
    dfDescription = pd.read_csv("app/MasterData/symptom_Description.csv")
    result = pd.Series(dfDescription.iloc[:, 1].values, index=dfDescription.iloc[:, 0]).to_dict()
    try:
        # Find the description for the predicted disease
        description = result[prediction]
    except KeyError:
        raise HTTPException(status_code=404, detail="Description not found for the predicted disease.")

    # avec le pred on retrouve les précautions à prendre
    dfPred = pd.read_csv("app/MasterData/symptom_precaution.csv")

    dfPred.columns = ["Disease", "Precaution 1", "Precaution 2", "Precaution 3", "Precaution 4"]
    dfPred.head()

    row = dfPred[dfPred["Disease"] == prediction]
    precautions = row.iloc[0, 1:].dropna().tolist()
    print(precautions)

    return {
        "predicted_disease": prediction,
        "description": description,
        "precautions": precautions,
    }
