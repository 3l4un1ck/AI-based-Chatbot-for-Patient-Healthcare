import pickle

# Load the pre-trained model
with open("decision_tree_classifier.pkl", "rb") as f:
    decision_tree_model_pkl = pickle.load(f)
    print(decision_tree_model_pkl)
