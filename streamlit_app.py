#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import requests
import os
import logging
from streamlit_echarts import st_echarts

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Dashboard Néo-Banque", layout="centered")

# Ajout de la section RGPD dans la sidebar
with st.sidebar.expander("🔐 Données & RGPD"):
    st.markdown("""
    **Conformité RGPD**

    - Ce dashboard traite des données **pseudonymisées**.
    - Aucune donnée personnelle (nom, email...) n’est utilisée.
    - Les données sont utilisées uniquement à des fins de **scoring de prêt**.
    - Le traitement est **explicable** grâce aux outils SHAP.

    👉 Ce traitement respecte les principes du RGPD :
    - Licéité, transparence, finalité, minimisation.
    - Pas de stockage ni de profilage automatisé externe.
    """)

# Charger les données des clients
try:
    clients = pd.read_csv("data/clients.csv").reset_index().rename(columns={"index": "id"})
except FileNotFoundError:
    st.error("Erreur : Le fichier 'data/clients.csv' est introuvable. Veuillez le placer dans le répertoire 'data'.")
    st.stop()

st.title("📊 Dashboard conseiller")

# Sélection du client
selected_id = st.selectbox("Choisir un client", clients.index, key="client_selector")

client = clients.loc[selected_id]

# Utilisation de st.columns pour un agencement comme sur l'image
col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.subheader("Informations Client")
    
    # Création d'un DataFrame pour les informations du client
    client_info_df = pd.DataFrame({
        "Caractéristique": [
            "ID Client",
            "Âge",
            "Revenu annuel",
            "Ancienneté",
            "Incidents de paiement",
            "Score de crédit initial"
        ],
        "Valeur": [
            str(client['id']),
            f"{int(client['age'])} ans",
            f"{float(client['revenu']):,.0f} €",
            f"{int(client['anciennete'])} ans",
            str(int(client['nb_incidents'])),
            f"{float(client['score_credit']):.1f}"
        ]
    })
    
    # Afficher le DataFrame
    st.dataframe(client_info_df, hide_index=True) # hide_index=True pour ne pas afficher l'index du DataFrame


API_URL = os.getenv("API_URL", "https://neo-api-jigt.onrender.com/predict")

# La logique de scoring et d'affichage est maintenant exécutée à chaque changement de client sélectionné
input_data = {
    "age": int(client["age"]),
    "revenu": float(client["revenu"]),
    "anciennete": int(client["anciennete"]),
    "nb_incidents": int(client["nb_incidents"]),
    "score_credit": float(client["score_credit"]),
}

try:
    logging.info(f"Envoi de la requête à l'API pour le client ID: {client['id']}")
    res = requests.post(API_URL, json=input_data, timeout=10)
    res.raise_for_status()
    response_data = res.json()
    score = response_data["score"]
    explanations_from_api = response_data.get("explanations", [])

    with col2:
        st.subheader("Score de Crédit")
        credit_score_percentage = round(score * 100, 1)

        options = {
            "series": [
                {
                    "type": "gauge",
                    "axisLine": {
                        "lineStyle": {
                            "width": 10,
                            "color": [
                                [0.5, "#ea4521"],
                                [0.8, "#f7bb10"],
                                [1, "#269f67"]
                            ]
                        }
                    },
                    "pointer": {"show": False},
                    "axisTick": {"show": False},
                    "splitLine": {"show": False},
                    "axisLabel": {"show": False},
                    "detail": {
                        "show": True,
                        "offsetCenter": [0, "-10%"],
                        "valueAnimation": True,
                        "formatter": "{value}%",
                        "fontSize": 30,
                        "fontWeight": "bolder",
                        "color": "#333",
                    },
                    "title": {
                        "show": True,
                        "offsetCenter": [0, "120%"],
                        "fontSize": 14,
                        "color": "#333",
                        "formatter": ""
                    },
                    "data": [{"value": credit_score_percentage}],
                    "progress": {
                        "show": True,
                        "width": 10
                    },
                    "splitNumber": 0,
                    "radius": "80%",
                    "center": ["50%", "50%"],
                    "min": 0,
                    "max": 100,
                    "anchor": {"show": False},
                    "itemStyle": {"color": "#269f67"},
                }
            ]
        }
        st_echarts(options=options, height="200px")

        if score > 0.8:
            st.markdown("<p style='text-align: center; color: #269f67; font-weight: bold;'>✅ Éligible</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Score excellent pour l'octroi de crédit</p>", unsafe_allow_html=True)
        elif score > 0.5:
            st.markdown("<p style='text-align: center; color: #f7bb10; font-weight: bold;'>⚠️ Potentiellement éligible</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Score bon pour l'octroi de crédit, mais à étudier</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align: center; color: #ea4521; font-weight: bold;'>❌ Inéligible probable</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Score faible pour l'octroi de crédit</p>", unsafe_allow_html=True)

    st.subheader("Comprendre le score (Facteurs d'influence SHAP)")
    if explanations_from_api:
        for explanation_text in explanations_from_api:
            if "positivement" in explanation_text:
                st.write(f"⬆️ {explanation_text}")
            elif "négativement" in explanation_text:
                st.write(f"⬇️ {explanation_text}")
            else:
                st.write(f"➡️ {explanation_text}")
    else:
        st.info("Aucune explication détaillée disponible pour le moment ou une erreur s'est produite côté API.")

except requests.exceptions.ConnectionError:
    st.error(f"Erreur de connexion à l’API. Vérifiez que l'API est accessible à l'adresse {API_URL}.")
except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors de la communication avec l’API : {e}. Vérifiez l'URL et la configuration de l'API.")
except Exception as e:
    st.error(f"Une erreur inattendue s'est produite : {e}")
