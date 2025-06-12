from fastapi import FastAPI, HTTPException # Import HTTPException pour les erreurs claires
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os # Pour vérifier l'existence des fichiers

app = FastAPI(
    title="API de Prédiction de Risque Client Neo-Banque",
    description="API pour prédire le risque client et fournir des explications SHAP."
)

# 1. Définition des données attendues pour la requête
class ClientData(BaseModel):
    age: int
    revenu: float
    anciennete: int # Ancienneté du client
    nb_incidents: int # Nombre d'incidents (par exemple, de paiement)
    score_credit: int # Score de crédit actuel du client (ex: FICO, interne)

# 2. Définition des chemins vers les fichiers du modèle et de l'explainer
MODEL_PATH = "app/model.pkl"
EXPLAINER_PATH = "app/explainer.pkl"

# Initialisation des variables globales pour le modèle et l'explainer
model = None
explainer = None
# Ordre attendu des fonctionnalités - C'EST CRUCIAL !
# CETTE LISTE DOIT CORRESPONDRE EXACTEMENT À L'ORDRE ET AUX NOMS DES COLONNES
# QUI ONT ÉTÉ UTILISÉES POUR ENTRAÎNER VOTRE MODÈLE ET CRÉER VOTRE EXPLAINER SHAP.
expected_feature_order = ['age', 'revenu', 'anciennete', 'nb_incidents', 'score_credit']


# 3. Chargement du modèle et de l'explainer au démarrage de l'application
@app.on_event("startup")
def load_resources():
    global model, explainer # Permet de modifier les variables globales

    print("Tentative de chargement des ressources (modèle et explainer)...")

    # Vérification de l'existence des fichiers
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Fichier modèle non trouvé à : {MODEL_PATH}. L'application ne peut pas démarrer.")
    if not os.path.exists(EXPLAINER_PATH):
        raise RuntimeError(f"Fichier explainer non trouvé à : {EXPLAINER_PATH}. L'application ne peut pas démarrer.")

    # Chargement
    try:
        model = joblib.load(MODEL_PATH)
        explainer = joblib.load(EXPLAINER_PATH)
        print("✅ Modèle et Explainer chargés avec succès.")
    except Exception as e:
        # En cas d'erreur de chargement, lever une exception pour empêcher le démarrage de l'API
        raise RuntimeError(f"❌ Erreur lors du chargement du modèle ou de l'explainer : {e}. L'application ne peut pas démarrer.")

# 4. Point de terminaison de prédiction
@app.post("/predict")
def predict(data: ClientData):
    print(f"✅ Données reçues pour prédiction : {data.dict()}")

    # Convertir les données reçues en DataFrame Pandas
    df = pd.DataFrame([data.dict()])

    # --- Étape CRUCIALE : Réindexer le DataFrame pour assurer l'ordre correct des colonnes ---
    # Cela garantit que les fonctionnalités sont passées à l'explainer et au modèle
    # dans le même ordre que celui utilisé pendant l'entraînement.
    try:
        df = df[expected_feature_order]
        print(f"✅ DataFrame réindexé avec l'ordre attendu : {df.columns.tolist()}")
    except KeyError as e:
        # Si une fonctionnalité attendue est manquante dans les données d'entrée
        error_msg = (f"❌ Erreur de clé lors de la réindexation du DataFrame : {e}. "
                     f"Assurez-vous que toutes les fonctionnalités de `expected_feature_order` sont présentes dans les données reçues.")
        print(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # --- DÉBOGAGE PRÉ-PRÉDICTION ---
    print("\n--- DÉBOGAGE PRÉ-PRÉDICTION ---")
    print("DataFrame à utiliser pour la prédiction et SHAP :")
    print(df)
    print(f"Forme du DataFrame (df.shape) : {df.shape}")
    print(f"Colonnes du DataFrame (df.columns) : {df.columns.tolist()}")
    print("--- FIN DÉBOGAGE PRÉ-PRÉDICTION ---\n")

    # 5. Prédiction du score (probabilité de la classe positive)
    score = None # Initialiser le score à None
    try:
        # model.predict_proba renvoie généralement un tableau numpy de forme (n_samples, n_classes)
        # Pour une classification binaire, c'est (n_samples, 2).
        # [0][1] prend la probabilité du premier échantillon pour la classe positive (index 1).
        prediction_proba = model.predict_proba(df)
        print(f"DEBUG: Type de sortie de model.predict_proba : {type(prediction_proba)}")
        print(f"DEBUG: Forme de sortie de model.predict_proba : {prediction_proba.shape}")
        
        if prediction_proba.ndim == 2 and prediction_proba.shape[1] > 1:
            score = prediction_proba[0][1] # Probabilité de la classe positive
        elif prediction_proba.ndim == 1:
            # Si le modèle renvoie un tableau 1D (ex: pour régression ou proba directe)
            score = prediction_proba[0]
        else:
            raise ValueError(f"Forme inattendue de predict_proba: {prediction_proba.shape}")
            
        print(f"✅ Score de prédiction calculé : {score:.4f}")

    except Exception as e:
        error_msg = f"❌ Erreur lors du calcul du score de prédiction : {e}. Vérifiez le modèle ou les données d'entrée."
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # 6. Calcul des valeurs SHAP pour l'explicabilité
    raw_shap_output = None # Initialiser raw_shap_output
    try:
        raw_shap_output = explainer.shap_values(df)
    except Exception as e:
        error_msg = f"❌ Erreur lors du calcul des valeurs SHAP : {e}. L'explainer est-il correct ?"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # --- DÉBOGAGE DE LA SORTIE SHAP BRUTE ---
    print(f"\n--- DÉBOGAGE DE LA SORTIE SHAP BRUTE (raw_shap_output) ---")
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
    else:
        print(f"Type de raw_shap_output inattendu : {type(raw_shap_output)}")
    print(f"--- FIN DU DÉBOGAGE DE LA SORTIE SHAP BRUTE ---\n")

    # 7. Extraction robuste des shap_values pour la classe positive
    shap_values = None
    if isinstance(raw_shap_output, list):
        # Cas le plus courant pour la classification binaire (TreeExplainer) :
        # raw_shap_output[0] = valeurs SHAP pour la classe 0 (négative)
        # raw_shap_output[1] = valeurs SHAP pour la classe 1 (positive)
        if len(raw_shap_output) == 2 and isinstance(raw_shap_output[1], np.ndarray):
            if raw_shap_output[1].ndim == 2 and raw_shap_output[1].shape[0] > 0:
                shap_values = raw_shap_output[1][0]  # Prendre les valeurs SHAP du premier échantillon
            elif raw_shap_output[1].ndim == 1:
                shap_values = raw_shap_output[1] # Si c'est déjà 1D
            else:
                print(f"Avertissement : raw_shap_output[1] a un nombre de dimensions inattendu : {raw_shap_output[1].ndim}.")
        # Cas fallback ou pour d'autres configurations
        elif len(raw_shap_output) > 0 and isinstance(raw_shap_output[0], np.ndarray):
            if raw_shap_output[0].ndim == 2 and raw_shap_output[0].shape[0] > 0:
                shap_values = raw_shap_output[0][0]
            elif raw_shap_output[0].ndim == 1:
                shap_values = raw_shap_output[0]
            else:
                print(f"Avertissement : raw_shap_output[0] a un nombre de dimensions inattendu : {raw_shap_output[0].ndim}.")
    elif isinstance(raw_shap_output, np.ndarray):
        # Pour des explainers qui renvoient un tableau NumPy direct (ex: KernelExplainer)
        if raw_shap_output.ndim == 3: # Forme (n_samples, n_classes, n_features)
            # Supposons 1 échantillon et que nous voulons la classe 1 (positive)
            if raw_shap_output.shape[0] > 0 and raw_shap_output.shape[1] > 1:
                shap_values = raw_shap_output[0, 1, :]
            else:
                print(f"Avertissement : raw_shap_output 3D n'a pas la forme attendue pour l'extraction.")
        elif raw_shap_output.ndim == 2: # Forme (n_samples, n_features)
            if raw_shap_output.shape[0] > 0:
                shap_values = raw_shap_output[0, :] # Prendre le premier échantillon
            else:
                print(f"Avertissement : raw_shap_output 2D est vide.")
        elif raw_shap_output.ndim == 1: # Forme (n_features,)
            shap_values = raw_shap_output
    
    # Vérification finale si shap_values a été extrait
    if shap_values is None:
        error_msg = "❌ Impossible d'extraire les valeurs SHAP de la structure renvoyée par l'explainer. La structure de raw_shap_output est inattendue."
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # 8. Vérification de la cohérence entre le nombre de features et les valeurs SHAP
    feature_names = df.columns.tolist() # Utiliser les noms de colonnes du DataFrame réindexé
    print(f"feature_names (issues du DataFrame réindexé) : {feature_names}")
    print(f"shap_values.shape (extraites) : {shap_values.shape}")
    
    if len(feature_names) != len(shap_values):
        error_msg = (f"❌ ERREUR MAJEURE : Le nombre de noms de fonctionnalités ({len(feature_names)}) "
                     f"ne correspond pas au nombre de valeurs SHAP ({len(shap_values)}). "
                     f"Cela indique une INCOHÉRENCE entre l'explainer SHAP et les données. "
                     f"Veuillez vérifier comment votre `explainer.pkl` a été créé.")
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # 9. Création du dictionnaire d'impacts SHAP
    shap_impacts = {}
    for i, feature in enumerate(feature_names):
        shap_impacts[feature] = float(shap_values[i]) # Convertir en float pour la sérialisation JSON

    # 10. Tri des impacts par valeur absolue décroissante
    sorted_impacts = sorted(shap_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    # 11. Construction de la liste d'explications lisibles
    explanation_list = []
    for feature, impact in sorted_impacts:
        sign = "positivement" if impact > 0 else "négativement"
        explanation_list.append(f"La fonctionnalité '{feature}' impacte {sign} la prédiction avec un poids de {impact:.3f}")
    
    print("✅ Prédiction et explications SHAP générées avec succès.")

    # 12. Retour de la réponse
    return {
        "score": score,
        "shap_values": shap_impacts,
        "explanations": explanation_list
    }
