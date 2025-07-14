import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap

def train_lab_model():
    data = pd.DataFrame({
        'WBC': [6.2, 11.4, 3.5, 9.1, 15.0, 4.2],
        'Hb': [13.4, 9.2, 15.1, 12.8, 7.5, 16.0],
        'Na': [140, 135, 132, 145, 130, 142],
        'Dx': ['Normal', 'Infection', 'Anemia', 'Normal', 'Anemia', 'Normal']
    })

    X = data.drop("Dx", axis=1)
    y = data["Dx"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, X

def predict_lab_result(model, input_dict):
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]
    return prediction

def explain_prediction(model, background_data, input_dict):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([input_dict]))
    return explainer, shap_values
