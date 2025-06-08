#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Dashboard Néo-Banque", layout="centered")

# Ajout de la section RGPD dans la sidebar
with st.sidebar.expander("🔐 Données & RGPD"):
    st.markdown("""
    **Conformité RGPD**

    - Ce dashboard traite des données **pseudonymisées**
    - Aucune donnée personnelle (nom, email...) n’est utilisée
    - Les données sont utilisées uniquement à des fins de **scoring de prêt**
    - Le traitement est **explicable** grâce aux outils SHAP

    👉 Ce traitement respecte les principes du RGPD :
    - Licéité, transparence, finalité, minimisation
    - Pas de stockage ni de profilage automatisé externe
    """)

clients = pd.read_csv("data/clients.csv").reset_index().rename(columns={"index": "id"})

st.title("📊 Dashboard conseiller")
selected_id = st.selectbox("Choisir un client", clients.index)

client = clients.loc[selected_id]
st.subheader("Informations client")
st.write(client)

API_URL = os.getenv("API_URL", "https://neo-banque-dashboard.onrender.com/predict")

if st.button("📤 Envoyer pour scoring"):
    input_data = {
        "age": int(client["age"]),
        "revenu": float(client["revenu"]),
        "anciennete": int(client["anciennete"]),
        "nb_incidents": int(client["nb_incidents"]),
        "score_credit": float(client["score_credit"]),
    }

    try:
        logging.info(f"Envoi de la requête à l'API : {input_data}")
        res = requests.post(API_URL, json=input_data, timeout=10)
        res.raise_for_status()  # Lève une exception pour les codes HTTP d'erreur
        response_data = res.json()
        #st.write("🔍 Réponse brute de l’API :", res.json())
        score = response_data["score"]
        explanations_from_api = response_data.get("explanations", [])

        st.metric("Score d’éligibilité au prêt", f"{score * 100:.1f} %")

        if score > 0.5:
            st.success("✅ Client éligible probable au prêt.")
        else:
            st.warning("⚠️ Client potentiellement inéligible ou profil à risque.")

        st.subheader("Comprendre le score (facteurs clés)")
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
