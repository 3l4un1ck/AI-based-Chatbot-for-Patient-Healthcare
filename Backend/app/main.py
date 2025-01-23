import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.middleware.cors import CORSMiddleware
from app.models import SymptomInput

with open("app/decision_tree_classifier.pkl", "rb") as f:
    decision_tree_model_pkl = pickle.load(f)
    print(decision_tree_model_pkl)
app = FastAPI(
    title="Project 3 - Healthcare ChatBot",
    description="This is a Healthcare ChatBot API",
    version="1.0.0",
    docs_url=None
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ReDoc will be displayed by default
@app.get("/", include_in_schema=False)
async def redoc():
    return get_redoc_html(openapi_url="/openapi.json", title="ReDoc - Auth Service")


# Swagger docs at /docs
@app.get("/docs", include_in_schema=False)
async def swagger_docs():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Swagger - Auth Service")


@app.post("/predict")
def predict_disease(symptom_input: SymptomInput):
    # Here, you would process the symptom data (e.g., convert to a feature vector)
    # For simplicity, we'll assume the model uses the number of symptoms as a feature:
    num_symptoms = len(symptom_input.symptoms)

    # Create feature array (example, modify based on your actual model input)
    input_data = np.array([[num_symptoms]])  # Assuming the model takes the number of symptoms

    # Make prediction
    prediction = decision_tree_model_pkl.predict(input_data)

    # Return the prediction
    return {"prediction": int(prediction[0])}
    # return {"message": f"Hello World{symptom_input.symptoms}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8999)
