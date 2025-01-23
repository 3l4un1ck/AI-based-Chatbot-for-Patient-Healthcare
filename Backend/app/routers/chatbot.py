from fastapi import APIRouter
from app.models import SymptomInput
from app.services import clf, le, reduced_data, description_list, precautionDictionary, sec_predict, calc_condition

router = APIRouter()

@router.post("/predict")
def predict_disease(symptom_input: SymptomInput):
    symptoms_exp = symptom_input.symptoms
    num_days = symptom_input.days

    input_vector = [0] * len(symptoms_exp)
    for symptom in symptoms_exp:
        input_vector[symptoms_exp.index(symptom)] = 1

    present_disease = clf.predict([input_vector])
    present_disease = le.inverse_transform(present_disease)[0]

    symptoms_given = reduced_data.columns[reduced_data.loc[present_disease].values[0].nonzero()]
    second_prediction = sec_predict(symptoms_exp)

    condition = calc_condition(symptoms_exp, num_days)
    if present_disease == second_prediction[0]:
        disease_info = {
            "disease": present_disease,
            "description": description_list[present_disease],
            "precautions": precautionDictionary[present_disease],
            "condition": condition
        }
    else:
        disease_info = {
            "disease": f"{present_disease} or {second_prediction[0]}",
            "description": f"{description_list[present_disease]} or {description_list[second_prediction[0]]}",
            "precautions": precautionDictionary[present_disease],
            "condition": condition
        }

    return disease_info