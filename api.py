from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os # Pour vérifier l'existence des fichiers

app = FastAPI()

class ClientData(BaseModel):
    age: int
    revenu: float
    anciennete: int
    nb_incidents: int
    score_credit: int

# Définir le chemin de vos fichiers de modèle et d'explainer
MODEL_PATH = "app/model.pkl"
EXPLAINER_PATH = "app/explainer.pkl"

# Vérifier si les fichiers existent avant de les charger
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Fichier modèle non trouvé à : {MODEL_PATH}")
if not os.path.exists(EXPLAINER_PATH):
    raise FileNotFoundError(f"Fichier explainer non trouvé à : {EXPLAINER_PATH}")

# Chargement du modèle et de l'explainer
try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    print("✅ Modèle et Explainer chargés avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle ou de l'explainer : {e}")
    # Quitter ou gérer l'erreur de manière appropriée si le chargement échoue
    exit(1)

@app.post("/predict")
def predict(data: ClientData):
    print("✅ Données reçues :", data.dict())

    # Convertir en dataframe
    # TRÈS IMPORTANT : Assurez-vous que l'ordre des colonnes dans le DataFrame correspond
    # à l'ordre des données d'entraînement de votre modèle et de votre explainer.
    # Vous DEVEZ adapter cette liste pour qu'elle corresponde à l'ordre réel de vos features.
    expected_feature_order = ['age', 'revenu', 'anciennete', 'nb_incidents', 'score_credit'] 
    
    df = pd.DataFrame([data.dict()])
    
    # Réindexer le DataFrame pour garantir le bon ordre des colonnes
    try:
        df = df[expected_feature_order]
        print(f"✅ DataFrame réindexé avec l'ordre attendu : {df.columns.tolist()}")
    except KeyError as e:
        print(f"❌ Erreur de clé lors de la réindexation du DataFrame : {e}. "
              f"Assurez-vous que toutes les fonctionnalités de `expected_feature_order` sont présentes dans les données reçues.")
        return {"error": f"Fonctionnalités manquantes dans les données d'entrée : {e}"}

    # Prédiction du score (probabilité de la classe positive)
    score = model.predict_proba(df)[0][1]

    # Calcul des valeurs SHAP
    raw_shap_output = explainer.shap_values(df)
    
    # --- DÉBOGAGE DE LA SORTIE SHAP ---
    print(f"\n--- DÉBOGAGE DE LA SORTIE SHAP ---")
    print(f"Type de raw_shap_output : {type(raw_shap_output)}")
    if isinstance(raw_shap_output, list):
        print(f"raw_shap_output est une liste. Longueur : {len(raw_shap_output)}")
        for i, item in enumerate(raw_shap_output):
            if isinstance(item, np.ndarray):
                print(f"  Élément {i} (ndarray) : Forme : {item.shape}, Dimensions : {item.ndim}")
            else:
                print(f"  Élément {i} : Type : {type(item)}")
    elif isinstance(raw_shap_output, np.ndarray):
        print(f"raw_shap_output est un ndarray. Forme : {raw_shap_output.shape}, Dimensions : {raw_shap_output.ndim}")
    print(f"--- FIN DU DÉBOGAGE DE LA SORTIE SHAP ---\n")

    # Extraction robuste des shap_values correspondant à la classe positive
    shap_values = None
    if isinstance(raw_shap_output, list):
        # Cas courant pour TreeExplainer (classification binaire), où raw_shap_output[0] est pour la classe 0, raw_shap_output[1] pour la classe 1
        if len(raw_shap_output) == 2 and isinstance(raw_shap_output[1], np.ndarray):
            # Pour la classification binaire, le deuxième élément est généralement pour la classe positive
            if raw_shap_output[1].ndim == 2 and raw_shap_output[1].shape[0] > 0:
                shap_values = raw_shap_output[1][0]  # Prendre les valeurs SHAP du premier échantillon
            elif raw_shap_output[1].ndim == 1:
                shap_values = raw_shap_output[1] # Si c'est déjà 1D
            else:
                print(f"Avertissement : raw_shap_output[1] a un nombre de dimensions inattendu : {raw_shap_output[1].ndim} ou est vide.")
        elif len(raw_shap_output) > 0 and isinstance(raw_shap_output[0], np.ndarray):
            # Fallback si ce n'est pas binaire ou si la structure est différente, essayer de prendre le premier élément.
            if raw_shap_output[0].ndim == 2 and raw_shap_output[0].shape[0] > 0:
                shap_values = raw_shap_output[0][0]
            elif raw_shap_output[0].ndim == 1:
                shap_values = raw_shap_output[0]
            else:
                print(f"Avertissement : raw_shap_output[0] a un nombre de dimensions inattendu : {raw_shap_output[0].ndim} ou est vide.")
    elif isinstance(raw_shap_output, np.ndarray):
        # Pour KernelExplainer ou d'autres types qui peuvent renvoyer un seul tableau
        if raw_shap_output.ndim == 3: # ex: (n_samples, n_classes, n_features)
            # Supposons n_samples=1 et que nous voulons la classe 1 (positive)
            shap_values = raw_shap_output[0, 1, :]
        elif raw_shap_output.ndim == 2: # ex: (n_samples, n_features)
            shap_values = raw_shap_output[0, :] # Prendre le premier échantillon
        elif raw_shap_output.ndim == 1: # ex: (n_features,)
            shap_values = raw_shap_output
    
    if shap_values is None:
        print(f"❌ Impossible d'extraire les shap_values de la structure inattendue. Veuillez vérifier le type et la forme de raw_shap_output.")
        return {"error": "Échec de l'extraction des valeurs SHAP en raison d'une structure inattendue."}

    feature_names = df.columns.tolist()
    print("feature_names (issues du DataFrame) :", feature_names)
    print("shap_values.shape (extraites) :", shap_values.shape)
    
    if len(feature_names) != len(shap_values):
        print(f"❌ ERREUR MAJEURE : Le nombre de noms de fonctionnalités ({len(feature_names)}) ne correspond pas au nombre de valeurs SHAP ({len(shap_values)}).")
        print("Cela indique une incohérence entre les fonctionnalités utilisées pour l'entraînement de l'explainer et les données actuelles.")
        # Vous devriez potentiellement lever une HTTPException ici pour signaler l'erreur au client.
        return {"error": "Incohérence entre les fonctionnalités attendues et les valeurs SHAP calculées."}

    shap_impacts = {}
    for i, feature in enumerate(feature_names):
        # Vérifier si l'index est valide pour shap_values avant d'accéder
        if i < len(shap_values):
            shap_impacts[feature] = float(shap_values[i]) # Convertir en float pour une sérialisation JSON sûre
        else:
            print(f"Avertissement : La fonctionnalité '{feature}' n'a pas de valeur SHAP correspondante (index {i} hors limites de shap_values).")

    # Tri des impacts absolus décroissants
    sorted_impacts = sorted(shap_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    # Construction d’une liste d’explications simplifiées
    explanation_list = []
    for feature, impact in sorted_impacts:
        sign = "positivement" if impact > 0 else "négativement"
        explanation_list.append(f"La fonctionnalité '{feature}' impacte {sign} la prédiction avec un poids de {impact:.3f}")
    
    print("✅ Score renvoyé :", score)
    print("✅ Explications :", explanation_list)

    return {
        "score": score,
        "shap_values": shap_impacts,
        "explanations": explanation_list
    }
