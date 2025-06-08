import streamlit as st
import pandas as pd
import requests
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Dashboard Néo-Banque", layout="centered")

# === En-tête stylisé ===
st.markdown(
    """
    <h1 style='text-align: center; color: #2e86c1;'>🏦 Dashboard - Néo-Banque</h1>
    <h4 style='text-align: center; color: gray;'>Analyse d'éligibilité au prêt client</h4>
    """,
    unsafe_allow_html=True
)

# === Logo optionnel (ajoute un logo.png dans data/) ===
try:
    logo = Image.open("data/logo.png")
    st.image(logo, width=100)
except:
    pass

# === Chargement des données clients ===
clients = pd.read_csv("data/clients.csv").reset_index().rename(columns={"index": "id"})

st.markdown("### 👤 Sélection d'un client")
selected_id = st.selectbox("Choisir un client", clients.index)
client = clients.loc[selected_id]

# === Affichage des infos client ===
st.markdown("### 📋 Informations client")
st.write(client)

# === Bouton de prédiction ===
if st.button("📤 Envoyer pour scoring"):
    input_data = {
        "age": int(client["age"]),
        "revenu": float(client["revenu"]),
        "anciennete": int(client["anciennete"]),
        "nb_incidents": int(client["nb_incidents"]),
        "score_credit": float(client["score_credit"]),
    }

    try:
        # URL de l'API en ligne – assure-toi qu'elle est correcte
        res = requests.post("https://neo-banque-dashboard.onrender.com/predict", json=input_data)
        response_data = res.json()

        # Affiche la réponse brute en debug
        #st.write("🔍 Réponse API :", response_data)

        # === Affichage du score ===
        score = response_data["score"]
        st.markdown("### 📤 Résultat du scoring")
        st.metric("💯 Score d’éligibilité", f"{score * 100:.1f} %")

        if score > 0.5:
            st.success("✅ Ce client est probablement **éligible au prêt**.")
        else:
            st.warning("⚠️ Ce client semble **à risque** ou **inéligible**.")

        # === Affichage des explications ===
        st.markdown("### 🔍 Explication du score")

        explanations = response_data.get("explanations", [])
        if explanations:
            for explanation in explanations:
                if "positivement" in explanation:
                    st.markdown(f"<span style='color:green;'>⬆️ {explanation}</span>", unsafe_allow_html=True)
                elif "négativement" in explanation:
                    st.markdown(f"<span style='color:red;'>⬇️ {explanation}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"➡️ {explanation}")
        else:
            st.info("ℹ️ Aucune explication n’a pu être générée pour ce client.")

    except requests.exceptions.ConnectionError:
        st.error("🚫 Erreur de connexion à l’API. Vérifiez que l'API est bien en ligne.")
    except Exception as e:
        st.error(f"❌ Une erreur inattendue s'est produite : {e}")
