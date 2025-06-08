from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# Exemple de définition des données attendues
class ClientData(BaseModel):
    age: int
    revenu: float
    anciennete: int
    nb_incidents: int
    score_credit: int

# Chargement du modèle et de l'explainer (à adapter selon ton code)
model = joblib.load("app/model.pkl")
explainer = joblib.load("app/explainer.pkl")

@app.post("/predict")
def predict(data: ClientData):
    # Convertir en dataframe
    df = pd.DataFrame([data.dict()])

    # Prédiction du score (probabilité de la classe positive)
    score = model.predict_proba(df)[0][1]

    # Calcul des valeurs SHAP
    raw_shap_output = explainer.shap_values(df)
    print(f"Avertissement: Structure de shap_values inattendue: {type(raw_shap_output)}, dim: {getattr(raw_shap_output, 'ndim', 'N/A')}")

    # Extraction robuste des shap_values correspondant à la classe positive
    if isinstance(raw_shap_output, list):
        # Si classification binaire avec shap_values pour chaque classe
        if len(raw_shap_output) == 2:
            shap_values = raw_shap_output[1][0]  # Classe positive, premier échantillon
        else:
            shap_values = raw_shap_output[0]  # Cas multi-classes ou autre, prendre le premier set
    elif isinstance(raw_shap_output, np.ndarray):
        if raw_shap_output.ndim == 3:
            # Exemple shape (1, 2, n_features) -> on prend [0, 1, :]
            shap_values = raw_shap_output[0, 1, :]
        elif raw_shap_output.ndim == 2:
            # Exemple shape (1, n_features)
            shap_values = raw_shap_output[0]
        elif raw_shap_output.ndim == 1:
            shap_values = raw_shap_output
        else:
            shap_values = raw_shap_output[0]
    else:
        # Autre type inattendu, tenter d’extraire le premier élément
        shap_values = raw_shap_output[0] if isinstance(raw_shap_output, list) else raw_shap_output

    feature_names = df.columns.tolist()
    print("feature_names:", feature_names)
    print("shap_values.shape:", shap_values.shape)

    shap_impacts = {}
    for i, feature in enumerate(feature_names):
        if i >= len(shap_values):
            print(f"Attention: shap_values index {i} hors limites")
            break
        shap_impacts[feature] = shap_values[i]

    # Tri des impacts absolus décroissants
    sorted_impacts = sorted(shap_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    # Construction d’une liste d’explications simplifiées
    explanation_list = []
    for feature, impact in sorted_impacts:
        sign = "positivement" if impact > 0 else "négativement"
        explanation_list.append(f"La feature '{feature}' impacte {sign} la prédiction avec un poids de {impact:.3f}")

    return {
        "score": score,
        "shap_values": shap_impacts,
        "explanations": explanation_list
    }
