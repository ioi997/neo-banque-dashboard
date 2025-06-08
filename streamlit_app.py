import streamlit as st
import pandas as pd
import requests
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Néo-Banque", layout="wide")

# === Header stylisé ===
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 48px;
            color: #2e86c1;
            font-weight: bold;
        }
        .section-header {
            font-size: 24px;
            margin-top: 20px;
            color: #1f4e79;
        }
        .card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
    </style>
    <h1 class='main-title'>🏦 Dashboard – Néo-Banque</h1>
    """,
    unsafe_allow_html=True
)

# === Logo ===
try:
    logo = Image.open("data/logo.png")
    st.sidebar.image(logo, width=150)
except:
    st.sidebar.markdown("# Néo-Banque")

st.sidebar.markdown("## 🌐 Menu")
st.sidebar.info("Choisissez un client pour analyser son profil de crédit.")

# === Chargement des données clients ===
clients = pd.read_csv("data/clients.csv").reset_index().rename(columns={"index": "id"})

# === Sélection du client ===
client_id = st.sidebar.selectbox("Sélectionnez un client", clients.index)
client = clients.loc[client_id]

# === Affichage des infos et bouton ===
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div class='section-header'>💼 Informations client</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'>" + client.to_frame().to_html(header=False) + "</div>", unsafe_allow_html=True)

with col2:
    if st.button("📤 Envoyer pour scoring"):
        input_data = {
            "age": int(client["age"]),
            "revenu": float(client["revenu"]),
            "anciennete": int(client["anciennete"]),
            "nb_incidents": int(client["nb_incidents"]),
            "score_credit": float(client["score_credit"]),
        }

        try:
            res = requests.post("https://neo-banque-dashboard.onrender.com/predict", json=input_data)
            data = res.json()

            if "score" not in data:
                st.error("L'API n'a pas retourné de score.")
                #st.write("Réponse brute :", data)
            else:
                st.markdown("<div class='section-header'>🌐 Résultat</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class='card'>
                    <h3>Score d'éligibilité : <span style='color:{'green' if data['score'] > 0.5 else 'red'};'>{data['score']*100:.1f}%</span></h3>
                    <p>{'Client probablement éligible ✅' if data['score'] > 0.5 else 'Client potentiellement à risque ⚠️'}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<div class='section-header'>🔍 Explication de la décision</div>", unsafe_allow_html=True)
                for explanation in data.get("explanations", []):
                    if "positivement" in explanation:
                        st.markdown(f"<div class='card' style='border-left: 5px solid green;'>{explanation}</div>", unsafe_allow_html=True)
                    elif "négativement" in explanation:
                        st.markdown(f"<div class='card' style='border-left: 5px solid red;'>{explanation}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='card'>{explanation}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur lors de la requête à l'API : {e}")
